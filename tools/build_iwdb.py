"""Build the IWDB annotation file and the human-review worklist.

    python -m tools.build_iwdb --sample-period 2.0 --per-class 300

Outputs, under ``assets/iwdb/``:

    iwdb_annotations.json   full benchmark, one record per sampled frame
    iwdb_review.csv         reviewer worklist, one row per frame
    iwdb_schema.json        the schema, for the data-availability statement
    iwdb_summary.json       distribution over encounter, article, band, zone
"""
from __future__ import annotations

import os
import json
import argparse

from vlm_usv import config as C
from vlm_usv.perception import dataset as D
from tools import build_iwdb_lib as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-period", type=float, default=2.0,
                    help="seconds between sampled frames within a sequence")
    ap.add_argument("--per-class", type=int, default=0,
                    help="cap on frames per encounter category (0 = no cap)")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--out", default=os.path.join(C.OUTPUT_ROOT, "iwdb"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    seqs = D.list_sequences(tuple(args.splits.split(",")))
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"building IWDB from {len(seqs)} sequences")

    records = []
    for i, s in enumerate(seqs, 1):
        try:
            r = B.build_sequence(s, sample_period_s=args.sample_period)
        except Exception as exc:                        # noqa: BLE001
            print(f"  [{i}/{len(seqs)}] {s.split}/{s.seq_id}  FAILED: {exc}")
            continue
        records.extend(r)
        print(f"  [{i}/{len(seqs)}] {s.split}/{s.seq_id:>4s}  frames={len(r)}")

    full = B.summarise(records)
    records = B.stratify(records, per_class=args.per_class)
    summary = B.summarise(records)

    os.makedirs(args.out, exist_ok=True)
    header = {
        "schema_version": B.SCHEMA_VERSION,
        "dataset": "USVTrack",
        "sample_period_s": args.sample_period,
        "per_class_cap": args.per_class,
        "engine": "vlm_usv.knowledge.inland_rules",
        "annotatable_articles": list(__import__(
            "vlm_usv.knowledge.inland_rules", fromlist=["x"]).ARTICLES_ANNOTATABLE),
        "before_stratification": full,
        "after_stratification": summary,
        "review": {
            "status": "pending",
            "protocol": "every record is exported to iwdb_review.csv and is "
                        "accepted, corrected or rejected by a reviewer before "
                        "the benchmark is used; agreement between the engine "
                        "and the reviewers is reported by "
                        "tools.build_iwdb_lib.apply_review",
        },
    }
    with open(os.path.join(args.out, "iwdb_annotations.json"), "w",
              encoding="utf-8") as f:
        json.dump({"header": header, "records": records}, f, indent=1)
    with open(os.path.join(args.out, "iwdb_schema.json"), "w",
              encoding="utf-8") as f:
        json.dump(B.SCHEMA, f, indent=1)
    with open(os.path.join(args.out, "iwdb_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"before_stratification": full,
                   "after_stratification": summary}, f, indent=1)
    B.write_review_worklist(records, os.path.join(args.out, "iwdb_review.csv"))

    print("\n=== summary ===")
    print(json.dumps(summary, indent=1))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
