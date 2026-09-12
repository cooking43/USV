"""USVTrack loader.

The release provides, per sequence:

* ``images/<split>/<seq>/img1/<timestamp>.jpg``   -- frames, filename is the epoch time
* ``images/<split>/<seq>/gt/gt.txt``              -- MOT ground truth tracks
* ``radaruv/<seq>.csv``                           -- 4D radar returns already
  projected into the image plane (columns ``u,v``)

The MOT file uses the standard nine-column layout
``frame, track_id, bb_left, bb_top, bb_w, bb_h, conf, class, visibility``
with ``frame`` a one-based index into the sorted frame list.

Ego speed over ground is recoverable from the radar file: the release stores
both the raw Doppler and an ego-compensated Doppler, and their difference is
constant across all returns of a sweep, so

    v_ego(t) = comp_velocity - doppler

is exact rather than estimated.  Ego yaw rate is estimated from the azimuth
flow of static returns; see :func:`estimate_ego_yaw_rate`.
"""
from __future__ import annotations

import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import config as C

DATASET_ROOT_PROJECTION = os.path.join(C.DATASET_ROOT, "projection")


# --------------------------------------------------------------------------
# Sequence index
# --------------------------------------------------------------------------

@dataclass
class SequenceRef:
    seq_id: str
    split: str
    path: str
    n_frames: int
    frame_rate: float
    width: int
    height: int

    @property
    def radar_csv(self) -> str:
        return os.path.join(C.RADAR_DIR, f"{self.seq_id}.csv")


def _read_seqinfo(path: str) -> dict:
    info = {}
    p = os.path.join(path, "seqinfo.ini")
    if not os.path.exists(p):
        return info
    for line in open(p, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def list_sequences(splits: Tuple[str, ...] = ("train", "test")) -> List[SequenceRef]:
    """Every sequence that has both MOT ground truth and a radar file."""
    out: List[SequenceRef] = []
    for split in splits:
        root = os.path.join(C.IMAGES_DIR, split)
        if not os.path.isdir(root):
            continue
        for seq in sorted(os.listdir(root), key=lambda s: (len(s), s)):
            p = os.path.join(root, seq)
            if not os.path.isdir(p):
                continue
            if not os.path.exists(os.path.join(p, "gt", "gt.txt")):
                continue
            if not os.path.exists(os.path.join(C.RADAR_DIR, f"{seq}.csv")):
                continue
            info = _read_seqinfo(p)
            out.append(
                SequenceRef(
                    seq_id=seq,
                    split=split,
                    path=p,
                    n_frames=int(info.get("seqLength", 0)),
                    frame_rate=float(info.get("frameRate", 25)),
                    width=int(info.get("imWidth", C.IMG_W)),
                    height=int(info.get("imHeight", C.IMG_H)),
                )
            )
    return out


# --------------------------------------------------------------------------
# Frames and tracks
# --------------------------------------------------------------------------

def frame_dir(seq: SequenceRef) -> str:
    """Directory holding the sequence's frames.

    ``img1`` is the MOT layout, but a few sequences of the release ship an
    incomplete ``img1`` while the rectified copy under ``projection/<seq>`` is
    complete, so whichever holds more frames is used.  Only the file names are
    read; the frame index of the MOT file maps to the sorted name list.
    """
    cands = [os.path.join(seq.path, "img1"),
             os.path.join(DATASET_ROOT_PROJECTION, seq.seq_id)]
    best, best_n = cands[0], -1
    for c in cands:
        if os.path.isdir(c):
            n = sum(1 for x in os.listdir(c) if x.endswith(".jpg"))
            if n > best_n:
                best, best_n = c, n
    return best


def load_frame_names(seq: SequenceRef) -> List[str]:
    d = frame_dir(seq)
    return sorted(
        (n for n in os.listdir(d) if n.endswith(".jpg")),
        key=lambda n: float(os.path.splitext(n)[0]),
    )


def load_frame_times(seq: SequenceRef) -> np.ndarray:
    """Epoch timestamp of every frame, ordered as the MOT frame index."""
    return np.array([float(os.path.splitext(n)[0]) for n in load_frame_names(seq)])


def load_tracks(seq: SequenceRef) -> Dict[int, Dict[int, np.ndarray]]:
    """``{track_id: {frame_index (1-based): array([x1, y1, x2, y2])}}``."""
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    with open(os.path.join(seq.path, "gt", "gt.txt")) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            fr = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]),
                          float(parts[4]), float(parts[5]))
            tracks.setdefault(tid, {})[fr] = np.array([x, y, x + w, y + h])
    return tracks


# --------------------------------------------------------------------------
# Radar
# --------------------------------------------------------------------------

RADAR_COLS = ("timestamp", "range", "doppler", "azimuth", "elevation",
              "power", "x", "y", "z", "comp_height", "comp_velocity", "u", "v")


class RadarSweeps:
    """All radar returns of one sequence, grouped by sweep timestamp."""

    def __init__(self, times: np.ndarray, data: Dict[float, np.ndarray]):
        self.times = times                 # sorted sweep timestamps
        self.data = data                   # timestamp -> (N, 12) float array
        self._ego_speed: Optional[np.ndarray] = None

    # ---- columns of the per-sweep array (timestamp dropped) ----
    RANGE, DOPPLER, AZ, EL, POWER, X, Y, Z, CH, CV, U, V = range(12)

    @classmethod
    def load(cls, csv_path: str) -> "RadarSweeps":
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, usecols=list(RADAR_COLS))
            ts = df["timestamp"].to_numpy(dtype=float)
            arr = df[list(RADAR_COLS[1:])].to_numpy(dtype=float)
            order = np.argsort(ts, kind="stable")
            ts, arr = ts[order], arr[order]
            times, starts = np.unique(ts, return_index=True)
            bounds = list(starts) + [len(ts)]
            data = {float(times[i]): arr[bounds[i]:bounds[i + 1]]
                    for i in range(len(times))}
            return cls(times.astype(float), data)
        except ImportError:
            pass
        by_ts: Dict[float, List[List[float]]] = {}
        with open(csv_path, newline="") as f:
            rd = csv.reader(f)
            header = next(rd)
            idx = {name: header.index(name) for name in RADAR_COLS}
            order = [idx[c] for c in RADAR_COLS[1:]]
            ti = idx["timestamp"]
            for row in rd:
                if not row:
                    continue
                ts = float(row[ti])
                by_ts.setdefault(ts, []).append([float(row[j]) for j in order])
        times = np.array(sorted(by_ts))
        data = {t: np.asarray(by_ts[t], dtype=float) for t in times}
        return cls(times, data)

    def nearest(self, t: float) -> Tuple[float, np.ndarray]:
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            i = 0
        elif i >= len(self.times):
            i = len(self.times) - 1
        elif abs(self.times[i - 1] - t) <= abs(self.times[i] - t):
            i -= 1
        ts = self.times[i]
        return float(ts), self.data[ts]

    def window(self, t: float, n_sweeps: int = 1) -> np.ndarray:
        """``n_sweeps`` consecutive sweeps centred on ``t``, concatenated."""
        i = int(np.argmin(np.abs(self.times - t)))
        lo = max(0, i - n_sweeps // 2)
        hi = min(len(self.times), lo + n_sweeps)
        lo = max(0, hi - n_sweeps)
        return np.concatenate([self.data[self.times[j]] for j in range(lo, hi)], axis=0)

    # ---- ego motion ----
    def ego_speed(self) -> np.ndarray:
        """Speed over ground per sweep, from ``comp_velocity - doppler``."""
        if self._ego_speed is None:
            v = np.empty(len(self.times))
            for k, t in enumerate(self.times):
                a = self.data[t]
                d = a[:, self.CV] - a[:, self.DOPPLER]
                v[k] = float(np.median(d))
            self._ego_speed = v
        return self._ego_speed

    def ego_speed_at(self, t: float) -> float:
        v = self.ego_speed()
        i = int(np.argmin(np.abs(self.times - t)))
        return float(v[i])


def estimate_ego_yaw_rate(sweeps: RadarSweeps, smooth_s: float = 1.0) -> np.ndarray:
    """Yaw rate (rad/s) per sweep from the azimuth flow of static returns.

    A return whose ego-compensated Doppler is near zero is stationary in the
    world frame.  Under a pure yaw of the platform the bearing of every such
    return drifts at exactly minus the yaw rate, so the median bearing drift
    of the static population is an unbiased yaw-rate estimate.  Returns are
    matched between consecutive sweeps by nearest range, which is adequate
    because the sweep interval is 67 ms.
    """
    times = sweeps.times
    yaw = np.zeros(len(times))
    eps = C.DEFAULT.perception.doppler_static_mps
    prev = None
    for k, t in enumerate(times):
        a = sweeps.data[t]
        m = np.abs(a[:, RadarSweeps.CV]) < eps
        cur = a[m][:, [RadarSweeps.RANGE, RadarSweeps.AZ]]
        if prev is not None and len(cur) >= 5 and len(prev[1]) >= 5:
            dt = t - prev[0]
            if 0 < dt < 0.5:
                pr, pa = prev[1][:, 0], prev[1][:, 1]
                # nearest-range match, one-directional
                j = np.abs(cur[:, 0][:, None] - pr[None, :]).argmin(axis=1)
                ok = np.abs(cur[:, 0] - pr[j]) < 1.5
                if ok.sum() >= 5:
                    daz = np.deg2rad(cur[ok, 1] - pa[j[ok]])
                    yaw[k] = -float(np.median(daz)) / dt
        prev = (t, cur)
    # median filter over roughly `smooth_s` of sweeps
    w = max(3, int(smooth_s * C.RADAR_HZ) | 1)
    pad = w // 2
    padded = np.pad(yaw, pad, mode="edge")
    sm = np.array([np.median(padded[i:i + w]) for i in range(len(yaw))])
    return sm


# --------------------------------------------------------------------------
# YOLO class labels (the release types targets only in the detection split)
# --------------------------------------------------------------------------

def load_yolo_classes() -> List[str]:
    if not os.path.exists(C.YOLO_CLASSES):
        return ["ship", "boat", "vessel"]
    return [l.strip() for l in open(C.YOLO_CLASSES) if l.strip()]


def load_yolo_labels(timestamp_name: str) -> Optional[np.ndarray]:
    """``(N, 5)`` array of ``cls, cx, cy, w, h`` in normalised coordinates."""
    p = os.path.join(C.YOLO_LABEL_DIR, f"{timestamp_name}.txt")
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p):
        parts = line.split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows) if rows else None
