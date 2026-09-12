"""Experience space keyed by a frozen sentence encoder.

This is the retrieval described in Section 3.3 of the paper, which the older
``policies.RetrievalPolicy`` does not implement: it keys cases on a
five-dimensional geometric feature vector, whereas the paper encodes the
structured scene text with a frozen 384-dimensional sentence encoder and
scores every similarity as the normalised cosine ``(1 + cos)/2``.

Three texts are built and they are deliberately not the same string.

``scene_text``   the retrieval query and the retrieval key.  Target class,
                 bearing zone, range band, measured range, approach rate and
                 motion, plus the ego speed.  It carries no action, because a
                 query is formed before any action exists, so a key built from
                 the action could not be matched by a query.

``case_text``    the redundancy representation, the scene and the operation
                 together.  Two cases in the same geometry that command
                 different manoeuvres are not duplicates, and neither are two
                 cases that command the same manoeuvre in different
                 geometries; the gate has to be able to say both.

``rule_text``    the applicability description of an article, used only when
                 rules are retrieved by similarity rather than supplied.

The encoder is loaded once and every embedding is cached on its text, so a
sweep that re-encodes the same scene at six checkpoints pays for it once.
"""
from __future__ import annotations

import os
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config as C
from . import inland_rules as IR
from ..perception.mpaf import TargetState


DEFAULT_ENCODER = os.path.join("models", "sentence-transformers",
                               "all-MiniLM-L6-v2")


# --------------------------------------------------------------------------
# Texts
# --------------------------------------------------------------------------

def scene_text(targets: Sequence[TargetState], ego_speed: float) -> str:
    """The retrieval query, and the key a case is stored under.

    No action field appears here.  The query is formed before the model has
    produced anything, so a key that contained an action could never be
    matched by a query, and the initial-similarity statistic reported with
    this experiment would not be computable on a test frame.
    """
    if not targets:
        return f"own vessel making {ego_speed:.1f} m/s, no target in view"
    parts = []
    for t in sorted(targets, key=lambda x: x.range_m):
        parts.append(f"{t.vessel_class or 'vessel'} on the {t.bearing_zone.replace('_', ' ')} "
                     f"at {t.range_band} range, {t.range_m:.0f} metres, "
                     f"approach rate {t.approach_rate_mps:+.1f} metres per second, "
                     f"{t.motion}")
    return (f"own vessel making {ego_speed:.1f} m/s, "
            f"{len(targets)} target{'s' if len(targets) > 1 else ''}, "
            + "; ".join(parts))


def case_text(case: Dict) -> str:
    """The case-operation representation the redundancy gate compares.

    The scene is part of it, and it has to be.  A representation built from
    the encounter category, the role, the article and the commanded manoeuvre
    alone takes only a few dozen distinct values, so the first case of each
    combination exhausts the space and every later candidate is a duplicate of
    it whatever the geometry that produced it.  That is the opposite of what
    the experience space is for: a single vessel fine on the port bow and two
    vessels ahead resolve to the same category and the same manoeuvre while
    being different scenes, and it is that difference the space accumulates.
    Redundancy is therefore assessed on the scene and the operation together,
    and two cases are duplicates only when both halves agree.
    """
    return (f"{case.get('scene_text', '')} | "
            f"{case.get('encounter', 'no_encounter').replace('_', ' ')} "
            f"as {case.get('role', 'none').replace('_', ' ')} "
            f"under article {int(case.get('article', 8))}, "
            f"alter {float(case.get('turn_deg', 0.0)):+.0f} degrees, "
            f"speed {case.get('speed', 'hold')}")


def rule_text(article: int) -> str:
    a = IR.ARTICLES[article]
    return f"Article {article}, {a.short}. {a.text}"


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

class SceneEncoder:
    """Frozen sentence encoder with a text-keyed cache.

    ``stub=True`` substitutes a deterministic hash embedding so that the
    partitioning, admission and checkpoint logic can be exercised on a machine
    with no GPU.  A run made that way is labelled in the output and must not be
    reported.
    """

    def __init__(self, path: str = DEFAULT_ENCODER, dim: int = 384,
                 stub: bool = False, device: str = "cpu"):
        self.dim = dim
        self.stub = stub
        self._cache: Dict[str, np.ndarray] = {}
        self._mu: Optional[np.ndarray] = None
        self._model = None
        self._tok = None
        self._backend = "stub"
        if not stub:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(path, device=device)
                self.dim = int(self._model.get_sentence_embedding_dimension())
                self._backend = "sentence_transformers"
            except ImportError:
                # the same frozen weights through plain transformers, with the
                # mean pooling and L2 normalisation the sentence-transformers
                # wrapper applies, so the vectors are the ones the paper
                # describes whichever package is installed
                import torch
                from transformers import AutoTokenizer, AutoModel
                self._torch = torch
                self._tok = AutoTokenizer.from_pretrained(path)
                self._model = AutoModel.from_pretrained(path).to(device).eval()
                self.dim = int(self._model.config.hidden_size)
                self._device = device
                self._backend = "transformers"

    def _stub_vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.normal(size=self.dim)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        miss = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if miss:
            if self.stub:
                vecs = np.stack([self._stub_vec(t) for t in miss])
            elif self._backend == "sentence_transformers":
                vecs = self._model.encode(miss, convert_to_numpy=True,
                                          normalize_embeddings=True,
                                          show_progress_bar=False)
            else:
                torch = self._torch
                outs = []
                for i in range(0, len(miss), 64):
                    batch = miss[i:i + 64]
                    enc = self._tok(batch, padding=True, truncation=True,
                                    max_length=256, return_tensors="pt")
                    enc = {k: v.to(self._device) for k, v in enc.items()}
                    with torch.no_grad():
                        h = self._model(**enc).last_hidden_state
                    m = enc["attention_mask"].unsqueeze(-1).float()
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                    v = torch.nn.functional.normalize(v, p=2, dim=1)
                    outs.append(v.cpu().numpy())
                vecs = np.concatenate(outs, axis=0)
            for t, v in zip(miss, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._transform(self._cache[t]) for t in texts])

    def one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    # ---- centring ----
    def fit_center(self, texts: Sequence[str]) -> Dict[str, float]:
        """Fix the origin of the similarity space on a reference corpus.

        A frozen encoder places every string built from one template inside a
        narrow cone, so the raw cosine between two scenes of this benchmark
        sits above $0.96$ whatever the geometry and cannot rank one against
        another.  Subtracting the mean of a fixed reference corpus and
        renormalising removes the component every scene shares and leaves the
        component that distinguishes them.  The reference is the initial
        experience space, computed once, so the transform is frozen with the
        encoder and does not drift as cases are added.
        """
        self._mu = None
        raw = self.encode(texts)
        mu = raw.mean(axis=0)
        self._mu = mu / (np.linalg.norm(mu) + 1e-12)
        v = self.encode(texts)
        S = v @ v.T
        iu = np.triu_indices(len(texts), 1)
        return {"pairwise_cos_min": float(S[iu].min()),
                "pairwise_cos_median": float(np.median(S[iu])),
                "pairwise_cos_max": float(S[iu].max())}

    def _transform(self, v: np.ndarray) -> np.ndarray:
        if self._mu is None:
            return v
        w = v - float(v @ self._mu) * self._mu
        n = float(np.linalg.norm(w))
        return (w / n).astype(np.float32) if n > 1e-8 else v


def nsim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised cosine, ``(1 + cos)/2``, on unit-norm inputs."""
    return (1.0 + (b @ a)) / 2.0


# --------------------------------------------------------------------------
# Experience space
# --------------------------------------------------------------------------

class ExperienceSpace:
    """Cases keyed by the scene embedding, gated on admission.

    ``gate='independent'``  the deterministic rule-consistency verifier of
                            Section 3.3 decides admission and the admitted
                            case is labelled from the geometry
    ``gate='none'``         every executed case is written back as the
                            generator labelled it, which is the configuration
                            that shows what the gate is worth

    ``redundancy=True``     a candidate whose case-operation representation
                            resembles a retained one by more than ``t_red`` is
                            rejected
    """

    def __init__(self, encoder: SceneEncoder, gate: str = "independent",
                 redundancy: bool = True, t_red: float = 0.92,
                 t_sim: float = 0.75, k_cases: int = 2):
        assert gate in ("independent", "none")
        self.enc = encoder
        self.gate = gate
        self.redundancy = redundancy
        self.t_red = t_red
        self.t_sim = t_sim
        self.k = k_cases
        self.cases: List[Dict] = []
        self.skeys: List[np.ndarray] = []       # scene embeddings
        self.ckeys: List[np.ndarray] = []       # case-operation embeddings
        self.n_offered = 0
        self.n_rejected_rule = 0
        self.n_rejected_redundant = 0

    # ---- construction ----
    def seed(self, cases: Sequence[Dict]) -> None:
        """Install the initial experience space without gating it.

        Every arm starts from the identical space, so the seeds are inserted
        directly rather than offered, and the admission counters begin at zero
        with the update stream.
        """
        for c in cases:
            self._insert(c)

    def _insert(self, case: Dict) -> None:
        self.cases.append(dict(case))
        self.skeys.append(self.enc.one(case["scene_text"]))
        self.ckeys.append(self.enc.one(case_text(case)))

    def snapshot(self) -> "ExperienceSpace":
        """A frozen copy, so a checkpoint is not disturbed by later updates."""
        out = ExperienceSpace(self.enc, self.gate, self.redundancy,
                              self.t_red, self.t_sim, self.k)
        out.cases = [dict(c) for c in self.cases]
        out.skeys = list(self.skeys)
        out.ckeys = list(self.ckeys)
        out.n_offered = self.n_offered
        out.n_rejected_rule = self.n_rejected_rule
        out.n_rejected_redundant = self.n_rejected_redundant
        return out

    def __len__(self) -> int:
        return len(self.cases)

    # ---- retrieval ----
    def retrieve(self, query_vec: np.ndarray) -> List[Tuple[float, Dict]]:
        if not self.cases:
            return []
        K = np.stack(self.skeys)
        s = nsim(query_vec, K)
        order = np.argsort(-s)[: self.k]
        return [(float(s[j]), self.cases[j]) for j in order
                if s[j] >= self.t_sim]

    # ---- admission ----
    def offer(self, cmd: Dict, targets: Sequence[TargetState],
              ego_speed: float) -> bool:
        """Present one executed decision to the gate. Returns whether kept."""
        self.n_offered += 1
        cand = dict(cmd)
        if self.gate == "independent":
            gt = IR.governing_target(targets)
            enc, role, art = IR.classify_encounter(gt)
            if int(cand.get("article", 0) or 0) != int(art):
                self.n_rejected_rule += 1
                return False
            if not IR.admissible_actions(enc, gt).contains(
                    float(cand.get("turn_deg", 0.0)),
                    str(cand.get("speed", "hold"))):
                self.n_rejected_rule += 1
                return False
            cand["encounter"], cand["role"], cand["article"] = enc, role, art

        cand["scene_text"] = scene_text(targets, ego_speed)
        if self.redundancy and self.ckeys:
            v = self.enc.one(case_text(cand))
            if float(np.max(nsim(v, np.stack(self.ckeys)))) > self.t_red:
                self.n_rejected_redundant += 1
                return False
        self._insert(cand)
        return True

    # ---- statistics ----
    def nn_redundancy(self) -> Optional[float]:
        """Mean nearest-neighbour similarity among retained cases, Eq. (R_NN)."""
        n = len(self.ckeys)
        if n < 2:
            return None
        K = np.stack(self.ckeys)
        S = (1.0 + K @ K.T) / 2.0
        np.fill_diagonal(S, -np.inf)
        return float(np.mean(np.max(S, axis=1)))

    def composition(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for c in self.cases:
            k = c.get("encounter", "no_encounter")
            out[k] = out.get(k, 0) + 1
        return out

    def stats(self) -> Dict:
        return {"size": len(self.cases), "offered": self.n_offered,
                "rejected_rule": self.n_rejected_rule,
                "rejected_redundant": self.n_rejected_redundant,
                "R_NN": self.nn_redundancy(),
                "composition": self.composition()}
