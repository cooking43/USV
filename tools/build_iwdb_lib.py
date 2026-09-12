"""Inland Waterway Decision Benchmark: schema and builder.

The release annotates *where* targets are; it does not annotate *what the own
vessel should do*.  IWDB adds the second layer on top of the first without
inventing any geometry.

Construction has three stages.

1.  Metric grounding.  For every ground-truth track box the radar returns
    falling inside it give range, bearing, range rate, and, from the
    ego-compensated Doppler, whether the target is under way.  Nothing here is
    a judgement.

2.  Candidate annotation.  A deterministic engine over those attributes emits
    the encounter category, the own-vessel role, the governing article, the
    admissible action set and the preferred action.  The engine is the code in
    ``inland_rules.py``; it is auditable and reproducible, and it is the same
    object used as the checker at run time.

3.  Human review.  Every candidate is exported to a review worklist carrying
    the frame, the measured attributes and the proposed label.  A reviewer
    accepts, corrects or rejects.  Only reviewed frames enter the released
    benchmark, and the disagreement between the engine and the reviewers is
    reported as the automation-agreement figure.

Frames are sampled rather than taken wholesale, because consecutive frames of
one transit are near-duplicates: one frame per ``sample_period_s`` of each
track's observable span, stratified over encounter category.
"""
from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from vlm_usv import config as C
from vlm_usv.perception import dataset as D
from vlm_usv.perception import mpaf as G
from vlm_usv.knowledge import inland_rules as IR


SCHEMA_VERSION = "iwdb-1.0"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "description": (
        "Inland Waterway Decision Benchmark. Each record is one frame of one "
        "USVTrack sequence, carrying the measured state of every annotated "
        "target and the decision-level label for that instant."
    ),
    "frame_record": {
        "frame_uid": "str, stable hash of sequence and frame index",
        "sequence_id": "str, USVTrack sequence",
        "split": "str, train or test as released",
        "frame_index": "int, one-based index into the sorted frame list",
        "timestamp": "float, epoch seconds, from the frame file name",
        "image": "str, path relative to the dataset root",
        "ego": {
            "speed_mps": "float, speed over ground from comp_velocity - doppler",
            "yaw_rate_dps": "float, from the azimuth flow of static returns",
        },
        "targets": [{
            "track_id": "int, MOT ground-truth identity",
            "bbox_xyxy": "list[float], ground-truth box in pixels",
            "range_m": "float, RCS-compensated weighted range",
            "bearing_deg": "float, positive to starboard of the bow",
            "approach_rate_mps": "float, negative when the range closes",
            "world_radial_mps": "float, target radial velocity, ego motion removed",
            "motion": "str, static | oncoming | receding",
            "range_band": "str, close | mid | far | beyond",
            "bearing_zone": "str, one of the five described zones",
            "approaching": "bool",
            "n_returns": "int, radar returns supporting the estimate",
            "vessel_class": "str, powered by default; the release does not type "
                            "engineering, ferry, draught-constrained or "
                            "non-powered craft",
        }],
        "annotation": {
            "governing_target_id": "int or null",
            "encounter": "str, one of " + " | ".join(IR.ENCOUNTERS),
            "ego_role": "str, one of " + " | ".join(IR.ROLES),
            "governing_article": "int, article of the Inland Rules",
            "admissible_action": {
                "turn_direction": "str, port | starboard | either | none",
                "alteration_min_deg": "float",
                "alteration_max_deg": "float",
                "speeds": "list[str], subset of hold | reduce | half | stop",
                "note": "str, the clause the set follows from",
            },
            "preferred_action": {
                "turn_deg": "float, signed, positive to starboard",
                "speed": "str",
                "clears": "bool, whether the required margin is reached",
                "predicted_min_distance_m": "float",
            },
            "source": "str, auto for the engine output, reviewed once a "
                      "reviewer has accepted or corrected it",
            "review": {
                "status": "str, pending | accepted | corrected | rejected",
                "reviewer": "str",
                "corrected_fields": "list[str]",
                "comment": "str",
            },
        },
    },
    "header": {
        "n_frames": "int",
        "n_sequences": "int",
        "sample_period_s": "float",
        "engine_version": "str",
        "agreement": "filled in after review: per-field agreement between the "
                     "engine and the reviewers, and the inter-reviewer kappa",
    },
}


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def _uid(seq_id: str, frame: int) -> str:
    return hashlib.sha1(f"{seq_id}:{frame}".encode()).hexdigest()[:12]


def build_sequence(seq: D.SequenceRef, sample_period_s: float = 2.0,
                   cfg: Optional[C.PerceptionConfig] = None) -> List[Dict]:
    cfg = cfg or C.DEFAULT.perception
    times = D.load_frame_times(seq)
    names = D.load_frame_names(seq)
    tracks = D.load_tracks(seq)
    if len(times) == 0 or not tracks:
        return []
    sweeps = D.RadarSweeps.load(seq.radar_csv)
    yaw = D.estimate_ego_yaw_rate(sweeps)

    fps = len(times) / max(times[-1] - times[0], 1e-6)
    stride = max(1, int(round(sample_period_s * fps)))

    # metric series per track, evaluated on the sampled frames only and
    # smoothed along the track
    series: Dict[int, Dict[int, G.TargetState]] = {}
    for tid, boxes in tracks.items():
        s = G.track_series(sweeps, times, boxes, tid, stride=stride, cfg=cfg)
        s = G.smooth_series(s)
        series[tid] = {fr: st for fr, _, st in s}

    records: List[Dict] = []
    for fr in range(1, len(times) + 1, stride):
        targets = [series[tid][fr] for tid in sorted(series) if fr in series[tid]]
        if not targets:
            continue
        t = float(times[fr - 1])
        ego_v = sweeps.ego_speed_at(t)
        i = int(np.argmin(np.abs(sweeps.times - t)))
        ann = IR.annotate_frame(targets, ego_v)
        ann["source"] = "auto"
        ann["review"] = {"status": "pending", "reviewer": "",
                         "corrected_fields": [], "comment": ""}
        records.append({
            "frame_uid": _uid(seq.seq_id, fr),
            "sequence_id": seq.seq_id,
            "split": seq.split,
            "frame_index": fr,
            "timestamp": round(t, 5),
            "image": os.path.relpath(os.path.join(D.frame_dir(seq), names[fr - 1]),
                                     C.DATASET_ROOT).replace("\\", "/"),
            "ego": {
                "speed_mps": round(float(ego_v), 3),
                "yaw_rate_dps": round(float(np.rad2deg(yaw[i])), 3),
            },
            "targets": [s.to_dict() for s in targets],
            "annotation": ann,
        })
    return records


def stratify(records: Sequence[Dict], per_class: int = 0,
             seed: int = 0) -> List[Dict]:
    """Cap the number of frames per encounter category.

    Without a cap the benchmark is dominated by the ``no_encounter`` and
    ``static_hazard`` frames that make up most of a transit, and the
    give-way categories the paper is about would be a few percent of it.
    """
    if per_class <= 0:
        return list(records)
    rng = np.random.default_rng(seed)
    by: Dict[str, List[Dict]] = {}
    for r in records:
        by.setdefault(r["annotation"]["encounter"], []).append(r)
    out: List[Dict] = []
    for enc, rs in by.items():
        if len(rs) <= per_class:
            out.extend(rs)
        else:
            idx = rng.choice(len(rs), per_class, replace=False)
            out.extend(rs[i] for i in sorted(idx))
    out.sort(key=lambda r: (r["sequence_id"], r["frame_index"]))
    return out


def summarise(records: Sequence[Dict]) -> Dict:
    enc: Dict[str, int] = {}
    art: Dict[str, int] = {}
    role: Dict[str, int] = {}
    band: Dict[str, int] = {}
    zone: Dict[str, int] = {}
    n_tgt = []
    for r in records:
        a = r["annotation"]
        enc[a["encounter"]] = enc.get(a["encounter"], 0) + 1
        art[str(a["governing_article"])] = art.get(str(a["governing_article"]), 0) + 1
        role[a["ego_role"]] = role.get(a["ego_role"], 0) + 1
        n_tgt.append(len(r["targets"]))
        for t in r["targets"]:
            band[t["range_band"]] = band.get(t["range_band"], 0) + 1
            zone[t["bearing_zone"]] = zone.get(t["bearing_zone"], 0) + 1
    return {
        "n_frames": len(records),
        "n_sequences": len({r["sequence_id"] for r in records}),
        "targets_per_frame_mean": round(float(np.mean(n_tgt)), 3) if n_tgt else 0,
        "encounter_counts": dict(sorted(enc.items(), key=lambda kv: -kv[1])),
        "role_counts": dict(sorted(role.items(), key=lambda kv: -kv[1])),
        "article_counts": dict(sorted(art.items(), key=lambda kv: -kv[1])),
        "range_band_counts": dict(sorted(band.items(), key=lambda kv: -kv[1])),
        "bearing_zone_counts": dict(sorted(zone.items(), key=lambda kv: -kv[1])),
    }


def write_review_worklist(records: Sequence[Dict], path: str) -> None:
    """Flat CSV a reviewer can work through and edit in place."""
    import csv as _csv
    cols = ["frame_uid", "sequence_id", "frame_index", "image", "n_targets",
            "gov_track_id", "gov_range_m", "gov_bearing_deg",
            "gov_approach_mps", "gov_motion",
            "encounter", "ego_role", "article",
            "adm_turn", "adm_min_deg", "adm_max_deg", "adm_speeds",
            "pref_turn_deg", "pref_speed", "pref_clears",
            "review_status", "reviewer", "corrected_encounter",
            "corrected_role", "corrected_article", "corrected_turn_deg",
            "corrected_speed", "comment"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f)
        w.writerow(cols)
        for r in records:
            a = r["annotation"]
            gid = a["governing_target_id"]
            g = next((t for t in r["targets"] if t["track_id"] == gid), None)
            adm, pref = a["admissible_action"], a["preferred_action"]
            w.writerow([
                r["frame_uid"], r["sequence_id"], r["frame_index"], r["image"],
                len(r["targets"]),
                gid if gid is not None else "",
                g["range_m"] if g else "", g["bearing_deg"] if g else "",
                g["approach_rate_mps"] if g else "", g["motion"] if g else "",
                a["encounter"], a["ego_role"], a["governing_article"],
                adm["turn_direction"], adm["alteration_min_deg"],
                adm["alteration_max_deg"], "|".join(adm["speeds"]),
                pref["turn_deg"], pref["speed"], pref["clears"],
                "pending", "", "", "", "", "", "", "",
            ])


def apply_review(records: List[Dict], worklist_csv: str) -> Dict:
    """Merge a completed worklist back into the records.

    Returns the agreement report the paper needs: how often the reviewer
    accepted the engine output unchanged, per field.
    """
    import csv as _csv
    by_uid = {r["frame_uid"]: r for r in records}
    fields = {"encounter": 0, "ego_role": 0, "governing_article": 0,
              "preferred_action": 0}
    seen = 0
    for row in _csv.DictReader(open(worklist_csv, encoding="utf-8-sig")):
        r = by_uid.get(row["frame_uid"])
        if r is None or row.get("review_status", "pending") == "pending":
            continue
        seen += 1
        a = r["annotation"]
        corrected = []
        if row.get("corrected_encounter"):
            a["encounter"] = row["corrected_encounter"]; corrected.append("encounter")
        else:
            fields["encounter"] += 1
        if row.get("corrected_role"):
            a["ego_role"] = row["corrected_role"]; corrected.append("ego_role")
        else:
            fields["ego_role"] += 1
        if row.get("corrected_article"):
            a["governing_article"] = int(row["corrected_article"])
            corrected.append("governing_article")
        else:
            fields["governing_article"] += 1
        if row.get("corrected_turn_deg") or row.get("corrected_speed"):
            if row.get("corrected_turn_deg"):
                a["preferred_action"]["turn_deg"] = float(row["corrected_turn_deg"])
            if row.get("corrected_speed"):
                a["preferred_action"]["speed"] = row["corrected_speed"]
            corrected.append("preferred_action")
        else:
            fields["preferred_action"] += 1
        a["source"] = "reviewed"
        a["review"] = {"status": row.get("review_status", "accepted"),
                       "reviewer": row.get("reviewer", ""),
                       "corrected_fields": corrected,
                       "comment": row.get("comment", "")}
    return {"n_reviewed": seen,
            "engine_agreement": {k: (round(v / seen, 4) if seen else None)
                                 for k, v in fields.items()}}
