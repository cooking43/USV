"""Run the pipeline on one frame and print every intermediate quantity.

    python demo.py --frame 0
    python demo.py --frame 12 --no-experience     # rule-only configuration

The point of the demo is that nothing is hidden: the fused target state, the
articles retrieved, the precedents retrieved with their similarities, the
prompt exactly as the backbone received it, the raw reply, the parsed
decision and whether the execution check replaced it are all printed. The
reviewed reference is shown last, and only for comparison; it never enters
the pipeline.

Requires the USVTrack images (see the README for the download) and a local
copy of Qwen2-VL-2B-Instruct.
"""
from __future__ import annotations

import os
import json
import argparse

from vlm_usv import config as C
from vlm_usv.knowledge import inland_rules as IR
from vlm_usv.knowledge.experience_space import (ExperienceSpace, SceneEncoder,
                                                scene_text)
from vlm_usv.pipeline import NavigationPipeline, load_backbone
from vlm_usv.reasoning import prompt as PR
from tools.build_seed_cases import as_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="assets/iwdb/iwdb_annotations.json")
    ap.add_argument("--cases", default="assets/cases/initial_experience_space.json")
    ap.add_argument("--dataset-root", default=None,
                    help="USVTrack root; defaults to config.DATASET_ROOT")
    ap.add_argument("--model", default=None, help="path to Qwen2-VL-2B-Instruct")
    ap.add_argument("--frame", type=int, default=0,
                    help="index into the frames that carry at least one target")
    ap.add_argument("--visual-tokens", type=int, default=128)
    ap.add_argument("--no-experience", action="store_true",
                    help="retrieve articles only, which is the rule-only ablation")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = args.dataset_root or C.DATASET_ROOT
    records = [r for r in json.load(open(args.annotations,
                                         encoding="utf-8"))["records"]
               if r["targets"]]
    rec = records[args.frame % len(records)]
    targets = [as_state(t) for t in rec["targets"]]
    ego = rec["ego"]["speed_mps"]

    experience = encoder = None
    if not args.no_experience:
        encoder = SceneEncoder()
        seeds = json.load(open(args.cases, encoding="utf-8"))["cases"]
        # the origin of the similarity space is fixed on the seed corpus,
        # because a frozen encoder places every scene built from one template
        # inside a narrow cone and the raw cosine cannot rank them
        encoder.fit_center([c["scene_text"] for c in seeds])
        experience = ExperienceSpace(encoder)
        experience.seed(seeds)

    backbone = load_backbone(args.model, device=args.device,
                             visual_tokens=args.visual_tokens)
    pipe = NavigationPipeline(backbone, encoder, experience, admit=False)

    from PIL import Image
    image = Image.open(os.path.join(root, rec["image"])).convert("RGB")
    out = pipe.step(image, targets, ego)

    gt = IR.governing_target(targets)
    enc, role, art = IR.classify_encounter(gt)
    aset = IR.admissible_actions(enc, gt)
    pref = IR.preferred_action(gt, aset, ego) or {}

    line = "=" * 74
    print(f"\n{line}\nframe {rec['frame_uid']}  sequence {rec['sequence_id']}"
          f"  {rec['image']}\n{line}")
    print(f"\n-- fused perception state --\nown speed {ego:.1f} m/s")
    print(f"targets   {PR.structured_scene(targets)}")
    print(f"\n-- retrieval --")
    print("rule space       " + ", ".join(
        f"Art.{a} {IR.ARTICLES[a].short}" for a in out["articles"]))
    if experience is not None:
        q = encoder.one(scene_text(targets, ego))
        print(f"experience space ({len(out['cases'])} returned, "
              f"|E|={len(experience)})")
        for s_, c in experience.retrieve(q):
            print(f"    sim {s_:.3f}  {c['encounter']} / {c['role']} / "
                  f"Art.{c['article']} -> turn {c['turn_deg']:+.0f} deg, "
                  f"speed {c['speed']}")
    else:
        print("experience space  disabled")
    print(f"\n-- prompt as sent --\n{out['prompt']}")
    print(f"\n-- raw reply --\n{out['raw'].strip()!r}")
    d = out["decision"]
    print(f"\n-- parsed decision --")
    print("parse failed" if d is None else
          f"encounter {d['encounter']} | role {d['role']} | "
          f"article {d['article']} | turn {d['turn_deg']:+.0f} deg | "
          f"speed {d['speed']}")
    e = out["executed"]
    print(f"\n-- after the execution check --")
    print(f"replaced {out['shield_replaced']}" + ("" if e is None else
          f"  ->  turn {e['turn_deg']:+.0f} deg, speed {e['speed']}"))
    print(f"\n-- reviewed reference (not used by the pipeline) --")
    print(f"encounter {enc} | role {role} | governing article {art}")
    print(f"admissible turn direction {aset.turn_direction}, "
          f"max alteration {aset.alteration_max_deg:.0f} deg, "
          f"speeds {list(aset.speeds)}")
    print(f"preferred action turn {float(pref.get('turn_deg', 0.0)):+.0f} deg, "
          f"speed {pref.get('speed', 'hold')}\n")

    if hasattr(backbone, "release"):
        backbone.release()


if __name__ == "__main__":
    main()
