"""Metric grounding of an annotated track.

Given a ground-truth bounding box and the radar sweep nearest the frame, this
module returns the quantities the paper's state tuple carries: range, bearing,
approach rate, and the quantised band and zone.  Range is estimated by the
RCS-compensated weighted mean of equation (3),

    r_hat = sum_i omega_i r_i / sum_i omega_i ,   omega_i = rho_i r_i^4 ,

which removes the systematic under-estimate produced by weighting returns with
raw echo power under the r^-4 dependence of the radar equation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config as C
from .dataset import RadarSweeps

R = RadarSweeps


@dataclass
class TargetState:
    """One target at one instant, in the own-vessel frame."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    range_m: float
    bearing_deg: float          # + to starboard
    approach_rate_mps: float    # negative = closing
    n_returns: int
    power_med_db: float
    range_band: str
    bearing_zone: str
    approaching: bool
    world_radial_mps: float = 0.0   # target's own radial velocity, ego motion removed
    motion: str = "static"          # static | oncoming | receding
    vessel_class: str = "powered"
    range_source: str = "radar_rcs"

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = [round(float(v), 1) for v in self.bbox]
        for k in ("range_m", "bearing_deg", "approach_rate_mps", "power_med_db"):
            d[k] = round(float(d[k]), 3)
        return d


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------

def range_band(r: float) -> str:
    if r < C.D_CLOSE:
        return "close"
    if r < C.D_MID:
        return "mid"
    if r <= C.D_MAX:
        return "far"
    return "beyond"


def bearing_zone(b: float) -> str:
    for name, lo, hi in C.BEARING_ZONES:
        if lo <= b < hi:
            return name
    return "port_quarter" if b < 0 else "starboard_quarter"


# --------------------------------------------------------------------------
# Association
# --------------------------------------------------------------------------

def points_in_bbox(sweep: np.ndarray, bbox, pad: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    u, v = sweep[:, R.U], sweep[:, R.V]
    m = (u >= x1 - pad) & (u <= x2 + pad) & (v >= y1 - pad) & (v <= y2 + pad)
    return sweep[m]


def _mpaf_core(pts: np.ndarray, cfg: C.PerceptionConfig) -> np.ndarray:
    """Largest mutually-similar group under the joint similarity of eq. (1).

    The similarity combines range, bearing, echo power and Doppler; returns are
    joined when it exceeds ``T_edge``.  Connected components are found with a
    plain union-find, which is adequate at the few tens of returns that fall
    inside one bounding box.
    """
    n = len(pts)
    if n <= cfg.min_points:
        return pts
    r, az = pts[:, R.RANGE], pts[:, R.AZ]
    rho, dop = pts[:, R.POWER], pts[:, R.DOPPLER]
    dr = np.abs(r[:, None] - r[None, :]) / cfg.sigma_r
    da = np.abs(az[:, None] - az[None, :]) / cfg.sigma_theta
    dp = np.abs(rho[:, None] - rho[None, :]) / cfg.sigma_rho
    dv = np.abs(dop[:, None] - dop[None, :]) / cfg.sigma_v
    sim = np.exp(-0.25 * (dr ** 2 + da ** 2 + dp ** 2 + dv ** 2))
    adj = sim > cfg.t_edge

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(adj, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # prefer the group with the largest total RCS-compensated weight, which
    # favours a coherent rigid target over a diffuse clutter patch
    best, best_w = None, -1.0
    for idx in groups.values():
        if len(idx) < cfg.min_points:
            continue
        w = float(np.sum(pts[idx, R.POWER] * pts[idx, R.RANGE] ** cfg.rcs_exponent))
        if w > best_w:
            best, best_w = idx, w
    return pts[best] if best is not None else pts


def rcs_compensated_range(pts: np.ndarray, cfg: C.PerceptionConfig) -> float:
    w = pts[:, R.POWER] * pts[:, R.RANGE] ** cfg.rcs_exponent
    w = np.clip(w, 1e-9, None)
    return float(np.sum(w * pts[:, R.RANGE]) / np.sum(w))


def power_weighted_range(pts: np.ndarray) -> float:
    w = np.clip(pts[:, R.POWER], 1e-9, None)
    return float(np.sum(w * pts[:, R.RANGE]) / np.sum(w))


def extract_target_state(
    sweep: np.ndarray,
    bbox,
    track_id: int,
    cfg: Optional[C.PerceptionConfig] = None,
    vessel_class: str = "powered",
) -> Optional[TargetState]:
    """Metric state of one boxed target, or ``None`` if unobservable."""
    cfg = cfg or C.DEFAULT.perception
    pts = points_in_bbox(sweep, bbox, cfg.bbox_pad_px)
    if len(pts) == 0:
        return None
    keep = (pts[:, R.RANGE] < cfg.range_gate_m) & (pts[:, R.POWER] > cfg.power_floor_db)
    pts = pts[keep]
    if len(pts) < cfg.min_points:
        return None
    core = _mpaf_core(pts, cfg)
    if len(core) == 0:
        return None
    rng = rcs_compensated_range(core, cfg)
    brg = float(np.median(core[:, R.AZ]))
    # raw Doppler is the range rate relative to the own vessel and is what the
    # collision risk depends on; the ego-compensated value describes the
    # target's own motion and is not used here
    appr = float(np.median(core[:, R.DOPPLER]))
    # the ego-compensated Doppler is the target's own radial velocity in the
    # world frame, and separates a moored or fixed object from a vessel under
    # way; it is used for the aspect, never for the risk
    world_v = float(np.median(core[:, R.CV]))
    return TargetState(
        track_id=track_id,
        bbox=tuple(float(v) for v in bbox),
        range_m=rng,
        bearing_deg=brg,
        approach_rate_mps=appr,
        n_returns=int(len(core)),
        power_med_db=float(np.median(core[:, R.POWER])),
        range_band=range_band(rng),
        bearing_zone=bearing_zone(brg),
        approaching=bool(appr < -C.APPROACH_EPS),
        world_radial_mps=world_v,
        motion=classify_motion(world_v),
        vessel_class=vessel_class,
    )


def classify_motion(world_radial: float, eps: float = 0.35) -> str:
    if world_radial < -eps:
        return "oncoming"
    if world_radial > eps:
        return "receding"
    return "static"


# --------------------------------------------------------------------------
# Multi-sweep reference (Appendix A protocol, automated part)
# --------------------------------------------------------------------------

def accumulated_reference_range(
    sweeps: RadarSweeps,
    t: float,
    bbox,
    n_acc: int = 9,
    cfg: Optional[C.PerceptionConfig] = None,
) -> Optional[Tuple[float, int]]:
    """Range from ``n_acc`` accumulated sweeps, the reference of Appendix A.

    Accumulation is what the online estimator cannot do: it spans about
    0.6 s at the 15 Hz sweep rate and multiplies the number of returns on the
    target, so the reference is not a relabelled copy of the method output.
    Ghost and clutter rejection is left to the operator review stage.
    """
    cfg = cfg or C.DEFAULT.perception
    win = sweeps.window(t, n_acc)
    pts = points_in_bbox(win, bbox, cfg.bbox_pad_px)
    keep = (pts[:, R.RANGE] < cfg.range_gate_m) & (pts[:, R.POWER] > cfg.power_floor_db)
    pts = pts[keep]
    if len(pts) < cfg.min_points * 2:
        return None
    core = _mpaf_core(pts, cfg)
    if len(core) < cfg.min_points:
        return None
    return float(np.median(core[:, R.RANGE])), int(len(core))


# --------------------------------------------------------------------------
# Track-level series
# --------------------------------------------------------------------------

def track_series(
    sweeps: RadarSweeps,
    frame_times: np.ndarray,
    boxes: Dict[int, np.ndarray],
    track_id: int,
    stride: int = 1,
    cfg: Optional[C.PerceptionConfig] = None,
    n_sweeps: int = 3,
) -> List[Tuple[int, float, TargetState]]:
    """``[(frame_index, timestamp, state), ...]`` over the life of one track.

    ``n_sweeps`` consecutive sweeps are pooled before association.  Three
    sweeps span 0.2 s at the 15 Hz sweep rate, well inside one control period,
    and are needed because a single sweep leaves only a handful of returns on
    a target of inland size.
    """
    cfg = cfg or C.DEFAULT.perception
    out = []
    for fr in sorted(boxes)[::stride]:
        if fr - 1 >= len(frame_times):
            continue
        t = float(frame_times[fr - 1])
        sweep = sweeps.window(t, n_sweeps) if n_sweeps > 1 else sweeps.nearest(t)[1]
        st = extract_target_state(sweep, boxes[fr], track_id, cfg)
        if st is not None:
            out.append((fr, t, st))
    return out


def smooth_series(series: List[Tuple[int, float, TargetState]],
                  win_s: float = 1.0) -> List[Tuple[int, float, TargetState]]:
    """Median-smooth the per-frame estimates along a track.

    The approach rate is the measured Doppler, not a derivative of the range
    series: a rigid target inside one bounding box carries only a few returns,
    so differentiating its range amplifies the association noise by an order of
    magnitude, whereas the Doppler is the range rate the sensor measures
    directly.  A running median over about a second removes the outliers left
    by an occasional wrong cluster without lagging the encounter.
    """
    if len(series) < 5:
        return series
    ts = np.array([t for _, t, _ in series])
    dt = float(np.median(np.diff(ts)))
    w = max(3, int(round(win_s / max(dt, 1e-3))) | 1)
    pad = w // 2

    def med(a):
        p = np.pad(a, pad, mode="edge")
        return np.array([np.median(p[i:i + w]) for i in range(len(a))])

    rs = med(np.array([s.range_m for _, _, s in series]))
    bs = med(np.array([s.bearing_deg for _, _, s in series]))
    ap = med(np.array([s.approach_rate_mps for _, _, s in series]))
    wv = med(np.array([s.world_radial_mps for _, _, s in series]))
    out = []
    for k, (fr, t, s) in enumerate(series):
        s2 = TargetState(**{**s.__dict__})
        s2.range_m = float(rs[k])
        s2.bearing_deg = float(bs[k])
        s2.approach_rate_mps = float(ap[k])
        s2.world_radial_mps = float(wv[k])
        s2.motion = classify_motion(s2.world_radial_mps)
        s2.range_band = range_band(s2.range_m)
        s2.bearing_zone = bearing_zone(s2.bearing_deg)
        s2.approaching = bool(s2.approach_rate_mps < -C.APPROACH_EPS)
        out.append((fr, t, s2))
    return out
