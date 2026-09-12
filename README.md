# VLM-USV

Dynamic knowledge-augmented vision-language reasoning for unmanned surface vehicle navigation in inland waterways.

A vision-language model is asked to produce a navigation command that is both **operationally admissible** and **traceable to a statute**. It cannot do either from an image alone: a general model enumerates small waterborne targets unreliably, a monocular image carries no metric scale, and inland navigation regulations are almost absent from pre-training corpora. This repository supplies the three things it is missing and keeps them separate.

---

## Contents

```
vlm_usv/
  config.py                 all thresholds and constants in one place
  pipeline.py               the four stages wired together, one decision per call
  metrics.py                AGA, AAR, VF and the rest, with their exact predicates
  perception/
    mpaf.py                 multi-physical-feature adaptive filtering, RCS-compensated range
    dataset.py              USVTrack sequence, frame, track and radar-sweep access
  knowledge/
    inland_rules.py         Articles 7-13 as applicability conditions and action sets
    experience_space.py     scene encoding, retrieval, admission gate, redundancy filter
  reasoning/
    prompt.py               prompt template, output schema, parser
    backend.py              Qwen2-VL wrapper with CUDA-graph decoding
  safety/
    execution_check.py      one-step clearance test and command replacement
tools/
  convert_usvtrack_to_yolo.py   USVTrack MOT annotations -> YOLO format
  train_yolov11.py              detector fine-tuning
  dataset.yaml                  detector data configuration
  download_qwen2vl.py           fetch the backbone
  build_iwdb.py                 build the decision benchmark from the release
  build_seed_cases.py           build the initial experience space E0
assets/                                      local runtime assets (not tracked)
  rules/inland_rules.json                    generated rule corpus
  cases/initial_experience_space.json        generated seed cases
  iwdb/                                      generated decision benchmark
  weights/yolo11s_usvtrack.pt                locally supplied detector weights
demo.py                       one frame, every intermediate quantity printed
```

The experiment drivers that produced the tables in the paper are not included. Everything needed to run the method, rebuild the benchmark and score a decision is.

---

## Installation

Python 3.10, CUDA 12.1, one GPU with at least 12 GB. The reported results were produced on an NVIDIA RTX 3060 (12 GB).

```bash
git clone https://github.com/cooking43/USV.git
cd USV
conda create -n vlmusv python=3.10 -y
conda activate vlmusv

# torch first, from the PyTorch index
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`transformers` must be at least 4.45; earlier versions have no `Qwen2VLForConditionalGeneration`.

---

## Data

### 1. USVTrack

The method is evaluated on **USVTrack**, a 4D-radar and camera tracking dataset for inland waterways. It is not redistributed here. Download it from the official page and cite the authors:

- Dataset page: <https://usvtrack.github.io>
- Paper: <https://arxiv.org/abs/2506.18737>

Expected layout after extraction:

```
<USVTRACK_ROOT>/
  images/<split>/<sequence>/img1/<timestamp>.jpg
  images/<split>/<sequence>/gt/gt.txt          MOT ground-truth tracks
  images/<split>/<sequence>/seqinfo.ini
  radar/<split>/<sequence>.csv                 4D radar sweeps
```

Point the code at it either by editing `DATASET_ROOT` in `vlm_usv/config.py` or by exporting:

```bash
export USVTRACK_ROOT=/path/to/USVTrack
```

Only the 54 sequences carrying both tracking ground truth and radar are used; of the 91 annotated tracks, 73 are observable in the radar for at least ten states.

### 2. Backbone

```bash
python tools/download_qwen2vl.py            # Qwen2-VL-2B-Instruct -> models/
```

Roughly 4.5 GB. The 2B model is the one all reported numbers use.

### 3. Sentence encoder

The experience space is keyed by `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, frozen):

```bash
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir models/sentence-transformers/all-MiniLM-L6-v2
```

If `sentence-transformers` is not installed the encoder falls back to plain `transformers` with mean pooling and L2 normalisation, which yields the same vectors.

---

## Training the detector

Target enumeration is delegated to a detector rather than to the visual pathway of the backbone, because on the released sequences the VLM recovers 66.7 % of annotated instances against 95.5 % for a fine-tuned YOLOv11-s.

```bash
# 1. convert USVTrack MOT annotations to YOLO format
python tools/convert_usvtrack_to_yolo.py \
    --usvtrack-root $USVTRACK_ROOT \
    --out datasets/usvtrack_yolo

# 2. fine-tune, starting from the COCO-pretrained yolo11s checkpoint
python tools/train_yolov11.py \
    --data tools/dataset.yaml \
    --model yolo11s.pt \
    --epochs 100 --imgsz 640 --batch 16
```

Edit the `path:` field of `tools/dataset.yaml` to point at `datasets/usvtrack_yolo` before training. Weights land under `runs/detect/<name>/weights/best.pt`.

After training, copy the selected checkpoint to `assets/weights/yolo11s_usvtrack.pt`, or update the detector path in `vlm_usv/config.py`.

---

## Running

```bash
python demo.py --frame 0
```

prints the fused target state, the retrieved articles, the retrieved precedents with their similarities, the prompt exactly as sent, the raw reply, the parsed decision, whether the execution check replaced it, and — last, for comparison only — the reviewed reference.

To use the method in your own loop:

```python
from PIL import Image
from vlm_usv.knowledge.experience_space import ExperienceSpace, SceneEncoder
from vlm_usv.pipeline import NavigationPipeline, load_backbone
import json

seeds = json.load(open("assets/cases/initial_experience_space.json"))["cases"]
encoder = SceneEncoder()
encoder.fit_center([c["scene_text"] for c in seeds])   # fix the similarity origin
experience = ExperienceSpace(encoder, gate="independent", t_red=0.995)
experience.seed(seeds)

pipe = NavigationPipeline(load_backbone(), encoder, experience, admit=True)
out = pipe.step(Image.open("frame.jpg"), targets, ego_speed=1.5)
print(out["executed"])
```

`targets` is a list of `vlm_usv.perception.mpaf.TargetState`, which `tools/build_seed_cases.as_state` constructs from an IWDB record and `perception.mpaf.track_series` constructs from raw radar and a detection.

### Configuration that matters

| Symbol | Meaning | Value |
|---|---|---|
| $K_r$ | articles retrieved per decision | 3 |
| $K_c$ | precedents retrieved per decision | 2 |
| $T_{sim}$ | retrieval similarity threshold | 0.75 |
| $T_{red}$ | redundancy threshold of the admission gate | 0.995 |
| visual tokens | pixel budget the frame is resized to | 128 |
| $R_0 + d_{safe}$ | required clearance | 3 m + 8 m |
| $T_h$ | prediction horizon of the execution check | twice the yaw time constant |

Similarities are the normalised cosine $(1+\cos)/2$. The origin of the embedding space is fixed on the construction corpus before any similarity is taken; a frozen encoder places every scene built from one template inside a narrow cone, where the raw cosine has a median of about 0.95 and cannot rank one scene against another.

---

## Citing

If you use this code, please cite the paper:

```bibtex
@misc{vlmusv2026,
  title   = {Dynamic Knowledge-Augmented Vision-Language Reasoning for
             Unmanned Surface Vehicle Navigation in Inland Waterways},
  author  = {Xie, Daoshun and Yang, Shenhua and Chen, Guoquan and Zhu, Hong
             and Wang, Weijun and Huang, Zeyang},
  note    = {Manuscript},
  year    = {2026}
}
```

The evaluation depends on USVTrack, which must be cited separately:

```bibtex
@inproceedings{yao2025usvtrack,
  title     = {USVTrack: USV-Based 4D Radar-Camera Tracking Dataset for
               Autonomous Driving in Inland Waterways},
  author    = {Yao, Shanliang and Guan, Runwei and Ni, Yi and Xu, Sen and
               Yue, Yong and Zhu, Xiaohui and Liu, Ryan Wen},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and
               Systems (IROS)},
  year      = {2025},
  eprint    = {2506.18737},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

Third-party components: Qwen2-VL (Alibaba, Apache-2.0), YOLOv11 via Ultralytics (AGPL-3.0 — note that this is copyleft and applies to the detector training path), and all-MiniLM-L6-v2 (Apache-2.0).

## License

MIT, see [LICENSE](LICENSE). The dataset, backbone and detector framework carry their own licences.
