"""One decision, end to end.

The pipeline is four stages and they are deliberately separable, because each
one is falsifiable on its own.

    perception   radar returns inside the detected box are associated by
                 ``perception.mpaf`` and reduced to a target state carrying a
                 metric range, a bearing, a range rate and a motion class
    knowledge    the governing article is derived from that state by the rule
                 space, and the nearest precedents are retrieved from the
                 experience space by the scene embedding
    reasoning    both are written into a compact prompt, the backbone answers
                 with one delimited line, and the parser expands it
    safety       the command is propagated to the next decision instant and
                 replaced if it would breach the required clearance

Nothing here consults a benchmark label. The reviewed annotations under
``assets/iwdb`` are used for scoring only, after the fact.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from . import config as C
from .knowledge import inland_rules as IR
from .knowledge.experience_space import ExperienceSpace, SceneEncoder, scene_text
from .perception.mpaf import TargetState
from .reasoning import prompt as PR
from .safety.execution_check import Shield


class NavigationPipeline:
    """The deployed decision layer.

    ``experience`` may be ``None``, in which case only the rule space is
    retrieved and the case field of the prompt reads ``none``; that is the
    rule-only configuration of the ablation.  ``admit`` controls whether an
    executed decision is offered back to the experience space, which is what
    makes the knowledge base dynamic.
    """

    def __init__(self, backbone, encoder: Optional[SceneEncoder] = None,
                 experience: Optional[ExperienceSpace] = None,
                 shield: Optional[Shield] = None,
                 k_rules: int = 3, admit: bool = True):
        self.backbone = backbone
        self.encoder = encoder
        self.experience = experience
        self.shield = shield if shield is not None else Shield()
        self.k_rules = k_rules
        self.admit = admit

    # ---- retrieval ----
    def retrieve(self, targets: Sequence[TargetState], ego_speed: float):
        """Articles from the rule space, precedents from the experience space."""
        gt = IR.governing_target(targets)
        _enc, _role, art = IR.classify_encounter(gt)
        articles = ([art] + [a for a in (9, 7) if a != art])[: self.k_rules]
        cases: List[Dict] = []
        if self.experience is not None and self.encoder is not None:
            q = self.encoder.one(scene_text(targets, ego_speed))
            cases = [c for _s, c in self.experience.retrieve(q)]
        return articles, cases

    # ---- one frame ----
    def step(self, image, targets: Sequence[TargetState],
             ego_speed: float) -> Dict:
        """Return the decision, the command executed and the evidence used."""
        articles, cases = self.retrieve(targets, ego_speed)
        text = PR.build_prompt(targets, ego_speed, articles, cases)
        prepared = self.backbone.prepare_image(image)
        raw = self.backbone.generate([text], [prepared], PR.SYSTEM_PROMPT,
                                     PR.FEWSHOT)[0]
        decision, missing = PR.parse_decision(raw)
        if decision is None:
            raw = self.backbone.generate([text + PR.RETRY_SUFFIX], [prepared],
                                         PR.SYSTEM_PROMPT, PR.FEWSHOT)[0]
            decision, missing = PR.parse_decision(raw)

        executed, replaced = decision, False
        if decision is not None and self.shield is not None:
            executed, replaced = self.shield.repair(decision, targets,
                                                    ego_speed)

        if self.admit and executed is not None and self.experience is not None:
            self.experience.offer(executed, targets, ego_speed)

        return {"prompt": text, "raw": raw, "decision": decision,
                "executed": executed, "shield_replaced": replaced,
                "articles": list(articles), "cases": list(cases),
                "parse_missing": missing}


def load_backbone(path: Optional[str] = None, device: str = "cuda",
                  visual_tokens: int = 128, cuda_graph: bool = True):
    """Qwen2-VL through the batching wrapper, loaded once and kept resident.

    ``visual_tokens`` is the pixel budget the frame is resized to. The
    operating value is 128; the image and the retrieved text compete for the
    same attention and the statute is carried by the text, so halving the
    share taken by the image leaves the cited article unchanged and raises the
    fraction of admissible manoeuvres.
    """
    from .reasoning.backend import QwenBatchRunner
    return QwenBatchRunner(path or PR.MODEL_PATHS["2b"], device=device,
                           batch_size=1, visual_tokens=visual_tokens,
                           cuda_graph=cuda_graph, verbose=False)
