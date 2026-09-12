"""Build the initial experience space E0 from reviewed IWDB records.

    python -m tools.build_seed_cases --out assets/cases/initial_experience_space.json

A seed case is a scene together with the least-disruptive compliant manoeuvre
the rule engine derives for it, so the initial space is correct by
construction and what the run-time updates add to it can be measured against
a known starting point.  Cases are spread over the range bands of each
encounter category, which covers every category without covering every
configuration inside one; filling in those configurations is what the
experience space does at run time.

The sequences used to build E0 must be disjoint from those used for
evaluation.  ``--sequences`` takes the list explicitly so the partition is
recorded in the command rather than buried in a default.
"""
from __future__ import annotations

import os
import json
import argparse
import collections
from typing import Dict, List

from vlm_usv.knowledge import inland_rules as IR
from vlm_usv.knowledge.experience_space import scene_text
from vlm_usv.perception.mpaf import TargetState


def as_state(t: Dict) -> TargetState:
    return TargetState(
        track_id=t["track_id"], bbox=tuple(t["bbox"]), range_m=t["range_m"],
        bearing_deg=t["bearing_deg"], approach_rate_mps=t["approach_rate_mps"],
        n_returns=t["n_returns"], power_med_db=t["power_med_db"],
        range_band=t["range_band"], bearing_zone=t["bearing_zone"],
        approaching=t["approaching"], world_radial_mps=t["world_radial_mps"],
        motion=t["motion"], vessel_class=t["vessel_class"])


def sequence_of(record: Dict) -> str:
    parts = record["image"].replace("\\", "/").split("/")
    return parts[2] if len(parts) > 2 else parts[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations",
                    default=os.path.join("assets", "iwdb",
                                         "iwdb_annotations.json"))
    ap.add_argument("--sequences", default="",
                    help="comma-separated sequence ids to draw seeds from; "
                         "empty uses every sequence, which is only correct "
                         "when nothing else is evaluated")
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(
        "assets", "cases", "initial_experience_space.json"))
    args = ap.parse_args()

    records = json.load(open(args.annotations, encoding="utf-8"))["records"]
    keep = set(filter(None, args.sequences.split(","))) or None
    records = [r for r in records if r["targets"]
               and (keep is None or sequence_of(r) in keep)]

    by_class: Dict[str, List[Dict]] = collections.defaultdict(list)
    for r in records:
        st = [as_state(t) for t in r["targets"]]
        gt = IR.governing_target(st)
        if gt is None:
            continue
        enc, role, art = IR.classify_encounter(gt)
        pref = IR.preferred_action(gt, IR.admissible_actions(enc, gt),
                                   r["ego"]["speed_mps"]) or {}
        by_class[enc].append({
            "encounter": enc, "role": role, "article": int(art),
            "turn_deg": float(pref.get("turn_deg", 0.0)),
            "speed": str(pref.get("speed", "hold")),
            "scene_text": scene_text(st, r["ego"]["speed_mps"]),
            "band": gt.range_band,
            "source_frame": r["frame_uid"],
        })

    out: List[Dict] = []
    for k, v in sorted(by_class.items()):
        by_band: Dict[str, List[Dict]] = collections.defaultdict(list)
        for c in v:
            by_band[c["band"]].append(c)
        take, i = [], 0
        target = min(args.per_class, len(v))
        while len(take) < target:
            added = False
            for b in sorted(by_band):
                if i < len(by_band[b]):
                    take.append(by_band[b][i])
                    added = True
                    if len(take) >= target:
                        break
            if not added:
                break
            i += 1
        out += take

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(out), "per_class": args.per_class,
                   "sequences": sorted(keep) if keep else "all",
                   "composition": dict(collections.Counter(
                       c["encounter"] for c in out)),
                   "cases": out}, f, indent=1, ensure_ascii=False)
    print(f"{len(out)} seed cases -> {args.out}")
    for k, v in sorted(collections.Counter(c["encounter"] for c in out).items()):
        print(f"  {k:<20s} {v}")


if __name__ == "__main__":
    main()
