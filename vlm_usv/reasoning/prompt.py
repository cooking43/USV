"""Prompt construction and output parsing.

The decision is emitted as one delimited line of five fields.  Decoding is
sequential in the number of tokens generated and batching cannot amortise it,
so the length of the answer is the dominant term in single-frame latency:
``crossing_give_way|give_way|12|25|hold`` costs about fifteen tokens whereas
``CG|G|12|25|H`` costs about nine.  The parser expands the codes, so nothing
downstream ever sees an abbreviation.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

from ..knowledge import inland_rules as IR
from ..perception.mpaf import TargetState


ENCOUNTER_CODE = {
    "HO": "head_on", "CG": "crossing_give_way", "CS": "crossing_stand_on",
    "OV": "overtaking", "SH": "static_hazard", "NO": "no_encounter",
}
ROLE_CODE = {"G": "give_way", "S": "stand_on", "B": "both_give_way",
             "N": "none"}
SPEED_CODE = {"H": "hold", "R": "reduce", "F": "half", "P": "stop"}

ENCOUNTER_TO_CODE = {v: k for k, v in ENCOUNTER_CODE.items()}
ROLE_TO_CODE = {v: k for k, v in ROLE_CODE.items()}
SPEED_TO_CODE = {v: k for k, v in SPEED_CODE.items()}

SYSTEM_PROMPT = (
    "You are the navigation decision layer of an unmanned surface vessel on a "
    "Chinese inland waterway. Answer with one line of five fields separated "
    "by |, nothing else:\n"
    "encounter|role|article|turn|speed\n"
    "encounter: HO head-on, CG crossing give-way, CS crossing stand-on, "
    "OV overtaking, SH static hazard, NO none\n"
    "role: G give-way, S stand-on, B both give-way, N none\n"
    "article: number\n"
    "turn: degrees, negative to port, positive to starboard, 0 to keep course\n"
    "speed: H hold, R reduce, F half, P stop\n"
    "A negative approach rate means the range is closing.\n"
    "Which side the other vessel is on decides the obligation: one crossing "
    "from your starboard side makes you give-way (CG, role G); one crossing "
    "from your port side makes you stand-on (CS, role S). A target dead ahead "
    "closing head to head is HO and both give way (role B). A target you are "
    "coming up on from astern is OV. Anything not under way is SH."
)

FEWSHOT = [
    ("own speed 1.6 m/s\n"
     "targets: [starboard_bow mid 55m -1.8m/s oncoming]\n"
     "articles: Art.12 crossing, Art.9 avoidance principle\n"
     "cases: none",
     "CG|G|12|25|H"),
    ("own speed 1.1 m/s\n"
     "targets: [port_bow close 21m -0.4m/s static]\n"
     "articles: Art.9 avoidance principle, Art.7 safe speed\n"
     "cases: (+15deg,reduce)",
     "SH|G|9|15|R"),
]

FEWSHOT_USER, FEWSHOT_ASSISTANT = FEWSHOT[0]

RETRY_SUFFIX = (
    "\n\nAnswer with one line only, five fields separated by |, "
    "for example CG|G|12|25|H"
)

STRUCTURED_TEMPLATE = (
    "own speed {speed:.1f} m/s\n"
    "targets: {targets}\n"
    "articles: {articles}\n"
    "cases: {cases}"
)


def structured_scene(targets: Sequence[TargetState]) -> str:
    if not targets:
        return "none"
    out = []
    for t in targets:
        out.append(f"[{t.bearing_zone} {t.range_band} {t.range_m:.0f}m "
                   f"{t.approach_rate_mps:+.1f}m/s {t.motion}]")
    return " ".join(out)


VOCAB = {
    "encounter": set(IR.ENCOUNTERS),
    "role": set(IR.ROLES),
    "speed": set(IR.SPEEDS),
}

FIELD_RE = {
    "encounter": re.compile(r"encounter\s*[:=]?\s*\**\s*([a-z_]+)", re.I),
    "role": re.compile(r"role\s*[:=]?\s*\**\s*([a-z_]+)", re.I),
    # the model writes the article back in the form the corpus uses it,
    # "Art.9 avoidance principle", so the prefix has to be tolerated
    "article": re.compile(
        r"article\s*[:=]?\s*\**\s*(?:art(?:icle)?\s*\.?\s*|no\.?\s*|\u7b2c\s*)?(\d+)",
        re.I),
    "turn_deg": re.compile(r"turn(?:_deg|ing)?\s*[:=]?\s*\**\s*([+-]?\d+(?:\.\d+)?)",
                           re.I),
    "speed": re.compile(r"speed\s*[:=]?\s*\**\s*([a-z_]+)", re.I),
}


COMPACT_RE = re.compile(
    r"([A-Za-z_]+)\s*\|\s*([A-Za-z_]+)\s*\|\s*(?:art(?:icle)?\.?\s*)?(\d+)\s*\|"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*\|\s*([A-Za-z_]+)", re.I)


def _expand(raw: str, code_map: Dict[str, str], vocab) -> Optional[str]:
    """Resolve a field written either as a short code or as the full word."""
    if raw in code_map:
        return code_map[raw]
    up = raw.upper()
    if up in code_map:
        return code_map[up]
    low = raw.lower()
    return low if low in vocab else None


def _parse_compact(text: str):
    """The five fields as one delimited line, which is what the prompt asks for."""
    m = COMPACT_RE.search(text or "")
    if not m:
        return None, ["compact"]
    r_enc, r_role, art, turn, r_speed = m.groups()
    missing = []
    enc = _expand(r_enc, ENCOUNTER_CODE, VOCAB["encounter"])
    if enc is None:
        missing.append("encounter")
        enc = "no_encounter"
    role = _expand(r_role, ROLE_CODE, VOCAB["role"])
    if role is None:
        missing.append("role")
        role = "none"
    speed = _expand(r_speed, SPEED_CODE, VOCAB["speed"])
    if speed is None:
        missing.append("speed")
        speed = "hold"
    return {"encounter": enc, "role": role, "article": int(art),
            "turn_deg": float(turn), "speed": speed}, missing


def parse_decision(text: str, strict: bool = False):
    """Parse a reply into a decision, or return ``None``.

    Each field is matched independently rather than as one ordered pattern, so
    a reply that gets four fields right and wanders on the fifth still yields
    four, and the caller learns which one was missing instead of discarding
    the whole generation.  A value outside the permitted vocabulary counts as
    missing; a model that invents a category has not answered the question.
    """
    compact, cmiss = _parse_compact(text)
    if compact is not None and "speed" not in cmiss:
        return compact, cmiss

    got, missing = {}, []
    for field, rx in FIELD_RE.items():
        m = rx.search(text or "")
        if not m:
            missing.append(field)
            continue
        raw = m.group(1).lower()
        if field in VOCAB:
            if raw not in VOCAB[field]:
                missing.append(field)
                continue
            got[field] = raw
        elif field == "article":
            got[field] = int(raw)
        else:
            got[field] = float(raw)
    if missing and (strict or "turn_deg" in missing or "speed" in missing):
        return None, missing
    out = {
        "encounter": got.get("encounter", "no_encounter"),
        "role": got.get("role", "none"),
        "article": got.get("article", 8),
        "turn_deg": got.get("turn_deg", 0.0),
        "speed": got.get("speed", "hold"),
    }
    return out, missing


# --------------------------------------------------------------------------
# Model call
# --------------------------------------------------------------------------

MODEL_PATHS = {"2b": "models/Qwen2-VL-2B-Instruct",
               "7b": "models/Qwen2-VL-7B-Instruct"}


def build_prompt(targets: Sequence[TargetState], ego_speed: float,
                 articles: Sequence[int], cases: Sequence[Dict]) -> str:
    """The per-frame part of the prompt.

    The instruction block is constant across frames and lives in
    ``SYSTEM_PROMPT``, so this string is what varies with the scene and what a
    deployment with prefix caching pays for on every frame.
    """
    return STRUCTURED_TEMPLATE.format(
        speed=ego_speed,
        targets=structured_scene(targets),
        articles=", ".join(f"Art.{a} {IR.ARTICLES[a].short}" for a in articles)
        or "none",
        cases=", ".join(f"({c['turn_deg']:+.0f}deg,{c['speed']})"
                        for c in cases) or "none")
