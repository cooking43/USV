"""Metric definitions shared by the open-loop and closed-loop experiments."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

from . import config as C
from .knowledge import inland_rules as IR


# --------------------------------------------------------------------------
# Closed loop
# --------------------------------------------------------------------------

def aggregate_episodes(results: Sequence) -> Dict:
    """Collision rate, margin violation rate, clearance quantiles, PLR, RVR."""
    n = len(results)
    if n == 0:
        return {}
    dmin = np.array([r.d_min for r in results])
    plr = np.array([r.path_length_ratio for r in results])
    return {
        "n_episodes": n,
        "collision_rate_pct": round(100.0 * sum(r.collision for r in results) / n, 1),
        "margin_violation_rate_pct":
            round(100.0 * sum(r.margin_violation for r in results) / n, 1),
        "d_min_median_m": round(float(np.median(dmin)), 1),
        "d_min_p5_m": round(float(np.percentile(dmin, 5)), 1),
        "path_length_ratio_median": round(float(np.median(plr)), 3),
        "rule_violation_rate_pct":
            round(100.0 * sum(r.rule_violation for r in results) / n, 1),
        "decision_rule_violation_pct": round(
            100.0 * sum(r.n_rule_violations for r in results)
            / max(sum(r.n_decisions for r in results), 1), 1),
        "shield_intervention_pct": round(
            100.0 * sum(r.n_shield_interventions for r in results)
            / max(sum(r.n_decisions for r in results), 1), 1),
        "n_decisions": int(sum(r.n_decisions for r in results)),
    }


# --------------------------------------------------------------------------
# Open loop, against IWDB
# --------------------------------------------------------------------------

def score_decision(pred: Dict, record: Dict,
                   targets: Sequence[IR.TargetState]) -> Dict:
    """Per-frame scores of one decision against one IWDB record."""
    a = record["annotation"]
    adm = a["admissible_action"]
    aset = IR.ActionSet(
        turn_direction=adm["turn_direction"],
        alteration_min_deg=adm["alteration_min_deg"],
        alteration_max_deg=adm["alteration_max_deg"],
        speeds=tuple(adm["speeds"]), note=adm.get("note", ""))
    ref = a["preferred_action"]

    turn = float(pred.get("turn_deg", 0.0))
    speed = str(pred.get("speed", "hold"))
    admissible = aset.contains(turn, speed)
    ref_turn = float(ref["turn_deg"])
    dir_ok = (np.sign(turn) == np.sign(ref_turn)) or \
             (abs(ref_turn) < 1e-6 and abs(turn) <= 5)
    chk = IR.check_decision(pred, targets, record["ego"]["speed_mps"])
    return {
        "encounter_correct": pred.get("encounter") == a["encounter"],
        "role_correct": pred.get("role") == a["ego_role"],
        "article_correct": int(pred.get("article", 0) or 0) == a["governing_article"],
        "admissible": bool(admissible),
        "direction_consistent": bool(dir_ok),
        "alteration_error_deg": abs(turn - ref_turn),
        "rule_violation": not bool(admissible),
        "verified": bool(chk.accepted),
    }


def aggregate_decisions(scores: Sequence[Dict]) -> Dict:
    if not scores:
        return {}
    def pct(k):
        return round(100.0 * float(np.mean([s[k] for s in scores])), 1)
    return {
        "n": len(scores),
        "ECA": pct("encounter_correct"),
        "RIA": pct("role_correct"),
        "AGA": pct("article_correct"),
        "AAR": pct("admissible"),
        "TDC": pct("direction_consistent"),
        "MAE_deg": round(float(np.mean([s["alteration_error_deg"] for s in scores])), 1),
        "RVR": pct("rule_violation"),
        "VF": pct("verified"),
    }


# --------------------------------------------------------------------------
# Paired significance tests, no SciPy dependency
# --------------------------------------------------------------------------

def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Dict:
    """Two-sided exact McNemar test on paired binary outcomes."""
    a, b = list(map(bool, a)), list(map(bool, b))
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    # exact binomial tail at p = 0.5
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "p": float(min(1.0, 2 * tail))}


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_boot: int = 10000, seed: int = 0) -> Dict:
    """Bootstrap the paired mean difference ``a - b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "mean_diff": round(float(d.mean()), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "p_two_sided": round(float(2 * min((means <= 0).mean(),
                                           (means >= 0).mean())), 4),
    }


def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """Holm step-down correction over a family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = {"p": round(float(p), 5), "p_holm": round(float(adj), 5),
                  "significant": bool(adj < alpha)}
    return out
