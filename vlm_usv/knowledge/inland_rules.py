"""Inland Rules of the People's Republic of China: encoded corpus, deterministic
annotator, and the independent rule checker.

Each article is stored as a precondition over measurable attributes and an
action set.  The same object serves three purposes, which is why they live
together: it is the corpus retrieved by the rule branch, the generator of the
IWDB candidate annotation, and the checker that decides admission to the
experience space.  The checker is independent of the reasoning model in the
sense that matters, namely that nothing it computes passes through the model.

Observability note.  A forward-looking radar and a monocular camera measure
range, bearing and range rate, and the ego-compensated Doppler separates a
vessel under way from a fixed or moored object.  They do not measure the
target's heading, its draught, or its operational status.  Articles 14, 17, 18
and 21 turn on those, so they are carried in the corpus but are not used to
generate annotations from this release; ``ARTICLES_ANNOTATABLE`` records the
subset that is.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import math

from .. import config as C
from ..perception.mpaf import TargetState


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------

@dataclass
class Article:
    number: int
    short: str
    text: str
    precondition: str
    action_set: str


ARTICLES: Dict[int, Article] = {
    6: Article(6, "lookout",
               "Every vessel shall at all times maintain a proper look-out by "
               "sight and hearing and by all available means.",
               "always", "observe"),
    7: Article(7, "safe speed",
               "Every vessel shall at all times proceed at a safe speed so "
               "that she can take proper and effective action to avoid "
               "collision and be stopped within a distance appropriate to the "
               "prevailing circumstances.",
               "always", "hold | reduce | half | stop"),
    8: Article(8, "navigation principle",
               "Vessels shall navigate on the starboard side of the fairway "
               "where practicable and shall not impede the passage of others.",
               "under way", "hold | alter to starboard"),
    9: Article(9, "avoidance principle",
               "Any action taken to avoid collision shall be positive, made in "
               "ample time and be readily apparent to another vessel; a "
               "succession of small alterations shall be avoided.",
               "risk of collision exists",
               "alteration >= 10 deg | speed reduction"),
    10: Article(10, "head-on",
                "When two power-driven vessels are meeting on reciprocal or "
                "nearly reciprocal courses each shall alter her course to "
                "starboard so that each shall pass on the port side of the "
                "other.",
                "target within 20 deg of the bow, oncoming, range closing",
                "alter to starboard"),
    11: Article(11, "overtaking",
                "An overtaking vessel shall keep out of the way of the vessel "
                "being overtaken, shall signal her intention and shall not "
                "overtake until the vessel ahead consents.",
                "target forward of the beam, receding in the world frame, "
                "range closing",
                "alter to port | alter to starboard | reduce, with signal"),
    12: Article(12, "crossing",
                "When two power-driven vessels are crossing so as to involve "
                "risk of collision, the vessel which has the other on her own "
                "starboard side shall keep out of the way and shall avoid "
                "crossing ahead of the other vessel.",
                "target on the starboard bow or beam, oncoming, range closing",
                "alter to starboard | reduce | stop, passing astern"),
    13: Article(13, "stand-on",
                "Where one vessel is required to keep out of the way the other "
                "shall keep her course and speed, but may take action when it "
                "becomes apparent that the give-way vessel is not acting.",
                "target on the port side, oncoming, range closing",
                "hold | alter to starboard when close"),
    14: Article(14, "ferry",
                "A ferry crossing the fairway shall keep out of the way of "
                "vessels proceeding along the fairway.",
                "target class is ferry", "hold"),
    17: Article(17, "engineering vessel",
                "Vessels shall keep clear of engineering vessels at work and "
                "shall pass at reduced speed.",
                "target class is engineering vessel",
                "alter away | reduce"),
    18: Article(18, "draught-constrained",
                "Vessels shall not impede the passage of a vessel constrained "
                "by her draught navigating within the deep-water channel.",
                "target class is draught-constrained", "alter away | reduce"),
    21: Article(21, "non-powered craft",
                "Non-powered craft shall navigate outside the fairway used by "
                "power-driven vessels and shall not impede their passage.",
                "target class is non-powered", "hold"),
    23: Article(23, "restricted visibility",
                "In restricted visibility every vessel shall proceed at a safe "
                "speed adapted to the prevailing circumstances and shall be "
                "ready to take avoiding action.",
                "visibility restricted", "reduce | half"),
}

#: Articles the geometry of this release is sufficient to ground.
ARTICLES_ANNOTATABLE = (8, 9, 10, 11, 12, 13)

ENCOUNTERS = ("head_on", "crossing_give_way", "crossing_stand_on",
              "overtaking", "static_hazard", "no_encounter")

ROLES = ("give_way", "stand_on", "both_give_way", "none")

TURNS = ("port", "starboard", "either", "none")

SPEEDS = ("hold", "reduce", "half", "stop")


# --------------------------------------------------------------------------
# Risk ordering
# --------------------------------------------------------------------------

def time_to_range_zero(t: TargetState) -> float:
    """Seconds until the range reaches zero at the present closing rate."""
    rate = -t.approach_rate_mps
    if rate <= 1e-3:
        return math.inf
    return t.range_m / rate


def governing_target(targets: Sequence[TargetState],
                     d_m: float = C.D_MAX) -> Optional[TargetState]:
    """The target that governs the decision.

    Ordering is by time to range zero rather than by range, so a distant target
    closing fast outranks a near one opening.  Targets beyond the described
    scene or not closing are ignored.
    """
    cand = [t for t in targets
            if t.range_m <= d_m and t.approach_rate_mps < -C.APPROACH_EPS]
    if not cand:
        return None
    return min(cand, key=time_to_range_zero)


# --------------------------------------------------------------------------
# Encounter classification
# --------------------------------------------------------------------------

HEAD_ON_HALF_ANGLE = 20.0
FORWARD_HALF_ANGLE = 67.5
BEAM_HALF_ANGLE = 112.5


def classify_encounter(t: Optional[TargetState]) -> Tuple[str, str, int]:
    """``(encounter, ego_role, governing_article)`` for one governing target."""
    if t is None:
        return "no_encounter", "none", 8

    b = t.bearing_deg
    if t.motion == "static":
        return "static_hazard", "give_way", 9

    if t.motion == "receding" and abs(b) <= FORWARD_HALF_ANGLE:
        # the target draws away in the world frame while the range closes, so
        # the own vessel is the faster one coming up from astern of it
        return "overtaking", "give_way", 11

    if abs(b) <= HEAD_ON_HALF_ANGLE:
        return "head_on", "both_give_way", 10

    if b > 0:
        return "crossing_give_way", "give_way", 12
    return "crossing_stand_on", "stand_on", 13


# --------------------------------------------------------------------------
# Admissible action sets
# --------------------------------------------------------------------------

@dataclass
class ActionSet:
    turn_direction: str                 # one of TURNS
    alteration_min_deg: float
    alteration_max_deg: float
    speeds: Tuple[str, ...]
    note: str = ""

    def contains(self, turn_deg: float, speed: str) -> bool:
        if speed not in self.speeds:
            return False
        mag = abs(turn_deg)
        if self.turn_direction == "none":
            return mag <= self.alteration_max_deg
        if mag < self.alteration_min_deg or mag > self.alteration_max_deg:
            return False
        if self.turn_direction == "either":
            return True
        if self.turn_direction == "starboard":
            return turn_deg > 0
        return turn_deg < 0

    def to_dict(self):
        d = asdict(self)
        d["speeds"] = list(self.speeds)
        return d


def admissible_actions(encounter: str, t: Optional[TargetState]) -> ActionSet:
    """The set of manoeuvres the governing article permits.

    Lower bounds follow Article 9, which requires an alteration large enough to
    be readily apparent; upper bounds keep the manoeuvre within what a narrow
    inland fairway allows.
    """
    close = t is not None and t.range_m < C.D_CLOSE

    if encounter == "head_on":
        return ActionSet("starboard", 10.0, 45.0, ("hold", "reduce"),
                         "Article 10 requires each vessel to alter to starboard")
    if encounter == "crossing_give_way":
        return ActionSet("starboard", 15.0, 45.0, ("hold", "reduce", "half", "stop"),
                         "Article 12 forbids crossing ahead, so the resolution "
                         "passes astern")
    if encounter == "crossing_stand_on":
        if close:
            return ActionSet("starboard", 10.0, 45.0, ("hold", "reduce", "half", "stop"),
                             "Article 13 permits action once it is apparent the "
                             "give-way vessel is not acting")
        return ActionSet("none", 0.0, 5.0, ("hold",),
                         "Article 13 requires course and speed to be kept")
    if encounter == "overtaking":
        return ActionSet("either", 15.0, 45.0, ("hold", "reduce"),
                         "Article 11 leaves the side free but requires the "
                         "overtaking vessel to keep clear")
    if encounter == "static_hazard":
        if t is None or abs(t.bearing_deg) <= 5.0:
            return ActionSet("either", 10.0, 45.0, ("reduce", "half", "stop"),
                             "an obstruction dead ahead admits either side but "
                             "requires a speed reduction")
        side = "port" if t.bearing_deg > 0 else "starboard"
        return ActionSet(side, 10.0, 45.0, ("hold", "reduce", "half"),
                         "the alteration is away from the obstruction")
    return ActionSet("none", 0.0, 5.0, ("hold",), "no risk of collision")


# --------------------------------------------------------------------------
# Preferred action
# --------------------------------------------------------------------------

def predicted_min_distance(range_m: float, bearing_deg: float,
                           range_rate: float, turn_deg: float,
                           speed_factor: float, ego_speed: float,
                           horizon: float, dt: float,
                           v_perp: float) -> float:
    """Minimum clearance over the horizon under a constant-rate extrapolation.

    The target is carried forward along the measured line of sight at the
    measured range rate, and the disc around it grows at ``v_perp`` to bound
    the cross-range motion the radar does not resolve.  The own vessel is
    carried forward on the commanded course at the commanded speed.
    """
    b = math.radians(bearing_deg)
    tx, ty = range_m * math.cos(b), range_m * math.sin(b)
    ux, uy = math.cos(b), math.sin(b)
    psi = math.radians(turn_deg)
    v = ego_speed * speed_factor
    worst = math.inf
    n = int(horizon / dt)
    for k in range(n + 1):
        tau = k * dt
        px, py = tx + ux * range_rate * tau, ty + uy * range_rate * tau
        ex, ey = v * math.cos(psi) * tau, v * math.sin(psi) * tau
        d = math.hypot(px - ex, py - ey) - v_perp * tau
        worst = min(worst, d)
    return worst


SPEED_FACTOR = {"hold": 1.0, "reduce": 0.7, "half": 0.5, "stop": 0.0}


def expansion_rate(t: TargetState, cfg: Optional[C.ShieldConfig] = None) -> float:
    """Growth rate of the disc enclosing one target.

    The disc grows because the sensor resolves the radial component of the
    target's velocity and not the cross-range component.  A target whose
    ego-compensated Doppler places it below the static threshold is not under
    way, and bounding its cross-range motion by the speed of a vessel under way
    would inflate every moored boat and pier into an obstruction the vessel
    cannot pass, so the bound falls back to the static threshold itself.
    """
    cfg = cfg or C.DEFAULT.shield
    if t.motion == "static":
        return C.DEFAULT.perception.doppler_static_mps
    return cfg.v_perp


def preferred_action(t: Optional[TargetState], action_set: ActionSet,
                     ego_speed: float,
                     cfg: Optional[C.ShieldConfig] = None) -> Dict:
    """Smallest compliant manoeuvre that reaches the required passing margin.

    Ties are broken toward the smaller course change and then toward the
    higher speed, so the annotation never prefers an unnecessarily disruptive
    manoeuvre over an adequate one.
    """
    cfg = cfg or C.DEFAULT.shield
    need = C.DEFAULT.vessel.R0 + cfg.d_safe
    if t is None:
        return {"turn_deg": 0.0, "speed": "hold", "clears": True,
                "predicted_min_distance_m": None}

    best = None
    grid = sorted({0.0, *[float(a) for a in cfg.alter_grid_deg]},
                  key=lambda a: (abs(a), a))
    for turn in grid:
        for speed in ("hold", "reduce", "half", "stop"):
            if not action_set.contains(turn, speed):
                continue
            d = predicted_min_distance(
                t.range_m, t.bearing_deg, t.approach_rate_mps, turn,
                SPEED_FACTOR[speed], ego_speed, cfg.T_h, cfg.dt_pred,
                expansion_rate(t, cfg))
            key = (0 if d >= need else 1, abs(turn),
                   -SPEED_FACTOR[speed], -d)
            if best is None or key < best[0]:
                best = (key, turn, speed, d)
    if best is None:
        return {"turn_deg": 0.0, "speed": "hold", "clears": False,
                "predicted_min_distance_m": None}
    _, turn, speed, d = best
    return {"turn_deg": float(turn), "speed": speed,
            "clears": bool(d >= need),
            "predicted_min_distance_m": round(float(d), 2)}


# --------------------------------------------------------------------------
# The independent checker
# --------------------------------------------------------------------------

@dataclass
class CheckResult:
    accepted: bool
    precondition_ok: bool
    action_ok: bool
    encounter_ok: bool
    role_ok: bool
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def check_precondition(article: int, t: Optional[TargetState]) -> bool:
    """Does the measured geometry satisfy the cited article's precondition?"""
    if article in (6, 7, 8):
        return True
    if t is None:
        return article == 8
    b, closing = t.bearing_deg, t.approach_rate_mps < -C.APPROACH_EPS
    if article == 9:
        return closing
    if article == 10:
        return closing and abs(b) <= HEAD_ON_HALF_ANGLE and t.motion == "oncoming"
    if article == 11:
        return closing and abs(b) <= FORWARD_HALF_ANGLE and t.motion == "receding"
    if article == 12:
        return closing and 0 < b <= BEAM_HALF_ANGLE and t.motion == "oncoming"
    if article == 13:
        return closing and -BEAM_HALF_ANGLE <= b < 0 and t.motion == "oncoming"
    return False


def check_decision(decision: Dict, targets: Sequence[TargetState],
                   ego_speed: float) -> CheckResult:
    """Test a justification record against the geometry it claims to describe.

    Two independent tests are applied.  The precondition test asks whether the
    measured attributes satisfy the precondition of the article the record
    cites.  The action-set test asks whether the emitted manoeuvre lies in the
    set that article prescribes.  Neither consults the model that produced the
    record.
    """
    gt = governing_target(targets)
    true_enc, true_role, true_art = classify_encounter(gt)

    art = int(decision.get("article", 0) or 0)
    enc = str(decision.get("encounter", ""))
    role = str(decision.get("role", ""))
    turn = float(decision.get("turn_deg", 0.0))
    speed = str(decision.get("speed", "hold"))

    pre_ok = check_precondition(art, gt)
    aset = admissible_actions(true_enc, gt)
    act_ok = aset.contains(turn, speed)
    enc_ok = (enc == true_enc)
    role_ok = (role == true_role)

    accepted = pre_ok and act_ok and enc_ok and role_ok
    reason = ""
    if not enc_ok:
        reason = f"encounter asserted {enc!r}, geometry gives {true_enc!r}"
    elif not pre_ok:
        reason = f"precondition of Article {art} not satisfied"
    elif not role_ok:
        reason = f"role asserted {role!r}, geometry gives {true_role!r}"
    elif not act_ok:
        reason = (f"action ({turn:+.0f} deg, {speed}) outside the set "
                  f"prescribed by Article {true_art}")
    return CheckResult(accepted, pre_ok, act_ok, enc_ok, role_ok, reason)


def violates_rules(decision: Dict, targets: Sequence[TargetState]) -> bool:
    """Rule violation: the manoeuvre is outside the governing action set.

    A violation is a property of the manoeuvre alone.  A record that cites the
    wrong article but still turns the right way is unfaithful, not unsafe, and
    is counted by the verification faithfulness metric instead.
    """
    gt = governing_target(targets)
    enc, _, _ = classify_encounter(gt)
    aset = admissible_actions(enc, gt)
    return not aset.contains(float(decision.get("turn_deg", 0.0)),
                             str(decision.get("speed", "hold")))


# --------------------------------------------------------------------------
# Reference annotation for one frame
# --------------------------------------------------------------------------

def annotate_frame(targets: Sequence[TargetState], ego_speed: float) -> Dict:
    """Deterministic candidate annotation for one frame."""
    gt = governing_target(targets)
    enc, role, art = classify_encounter(gt)
    aset = admissible_actions(enc, gt)
    pref = preferred_action(gt, aset, ego_speed)
    return {
        "governing_target_id": None if gt is None else gt.track_id,
        "encounter": enc,
        "ego_role": role,
        "governing_article": art,
        "article_short": ARTICLES[art].short,
        "admissible_action": aset.to_dict(),
        "preferred_action": pref,
    }
