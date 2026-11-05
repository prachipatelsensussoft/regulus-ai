"""Sentence-level grounding check with a small entailment model (MiniCheck, Tang et al., EMNLP 2024) - 27 Oct 2026.

The validator's support checks count shared words. They cannot tell "the hair dye exemption is defined in 70.3(v)"
from "... in 70.3(u)", nor "an amendment costs $2,600 under 70.19(c)" from what (c) says (a new petition's deposit):
the words are all in the sources. A fact-checking model reads the cited text and the sentence and judges whether the
text supports the sentence. MiniCheck-RoBERTa-Large (355M parameters, MIT licence) is trained for exactly this
document-claim judgement and matched GPT-4 on the LLM-AggreFact benchmark in its paper.

Use: EntailmentChecker.support(document, claim) -> probability (0..1) that the document supports the claim. A document
longer than CHUNK_WORDS is split into overlapping chunks and the best chunk counts (MiniCheck's own procedure).
The model loads lazily, pinned to its revision; without the model files the checker is unavailable (callers then skip
the check - it is an aid, never a requirement).
"""
from __future__ import annotations

import os
import threading

MODELS = {
    # the FULL commit hash: the offline cache (local_files_only) keys snapshots by it, and a short hash resolves only
    # online (5 Nov 2026: the short form "74c8919647e6" made every offline load fail)
    "minicheck-roberta": {"repo": "lytang/MiniCheck-RoBERTa-Large", "revision": "74c8919647e61ed0f71bc177d94f10930f090068"},
}
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50
_LOCK = threading.Lock()
_CACHE: dict[str, "EntailmentChecker | None"] = {}


class EntailmentChecker:
    def __init__(self, tokenizer, model, device: str) -> None:
        self.tokenizer, self.model, self.device = tokenizer, model, device

    @classmethod
    def load(cls, name: str = "minicheck-roberta") -> "EntailmentChecker":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = MODELS[name]
        tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"], local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], revision=spec["revision"],
                                                                   local_files_only=True)
        device = "mps" if torch.backends.mps.is_available() and os.environ.get("REGULUS_ENTAILMENT_DEVICE", "mps") == "mps" else "cpu"
        model = model.to(device).eval()
        if device == "mps":
            model = model.half()
        return cls(tok, model, device)

    @staticmethod
    def chunks(document: str) -> list[str]:
        words = (document or "").split()
        if len(words) <= CHUNK_WORDS:
            return [" ".join(words)]
        step = CHUNK_WORDS - CHUNK_OVERLAP
        return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(1, len(words) - CHUNK_OVERLAP), step)]

    def support_many(self, pairs: list[tuple[str, str]], batch: int = 8) -> list[float]:
        """(document, claim) pairs -> support probabilities."""
        import torch

        flat, owner = [], []
        for k, (doc, claim) in enumerate(pairs):
            for c in self.chunks(doc):
                flat.append((c, claim))
                owner.append(k)
        best = [0.0] * len(pairs)
        with torch.no_grad():
            for i in range(0, len(flat), batch):
                part = flat[i:i + batch]
                # longest_first (5 Nov 2026): "only_first" raised on a long claim beside a full chunk ("Sequence to truncate
                # too short"); trimming the longer of the two never fails, and a chunk is the longer one in practice
                enc = self.tokenizer([d for d, _ in part], [c for _, c in part], truncation="longest_first",
                                     max_length=512, padding=True, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, 1].tolist()
                for j, p in enumerate(probs):
                    k = owner[i + j]
                    best[k] = max(best[k], float(p))
        return best

    def support(self, document: str, claim: str) -> float:
        return self.support_many([(document, claim)])[0]


def entailment_checker(name: str = "minicheck-roberta") -> EntailmentChecker | None:
    """The shared checker, or None when its model files are not available."""
    with _LOCK:
        if name not in _CACHE:
            try:
                _CACHE[name] = EntailmentChecker.load(name)
            except Exception:  # noqa: BLE001 - an aid only
                _CACHE[name] = None
        return _CACHE[name]
