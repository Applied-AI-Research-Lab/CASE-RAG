from __future__ import annotations

from typing import Sequence

import numpy as np


class EmbeddingVerifier:
    name = "embedding"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def score_batch(self, claim: str, passages: Sequence[dict]) -> list[float]:
        self._ensure()
        texts = [((p.get("title") or "") + " " + p.get("text", "")).strip()[:512]
                 for p in passages]
        if not texts:
            return []
        cemb = self._model.encode([claim], normalize_embeddings=True)
        pemb = self._model.encode(texts, normalize_embeddings=True)
        return (pemb @ cemb.T).ravel().tolist()


class NLIBasedVerifier:
    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base",
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _ensure(self) -> None:
        if self._pipe is None:
            from transformers import pipeline
            self._pipe = pipeline("text-classification", model=self.model_name,
                                  device=self.device, top_k=None)

    def score_batch(self, claim: str, passages: Sequence[dict]) -> list[float]:
        self._ensure()
        if not passages:
            return []
        pairs = [((p.get("text") or "")[:1000], claim) for p in passages]
        outs = self._pipe(pairs, batch_size=16)

        scores: list[float] = []
        for out in outs:
            rows = out if isinstance(out, list) else [out]
            ent = 0.0
            for r in rows:
                if r["label"].lower().startswith("entail"):
                    ent = r["score"]
            scores.append(float(ent))
        return scores


def build_verifier(kind: str, device: str | None = None,
                   model_name: str | None = None):
    if kind == "nli":
        return NLIBasedVerifier(model_name or "cross-encoder/nli-deberta-v3-base",
                                device=device)
    if kind == "embedding":
        return EmbeddingVerifier(model_name or "BAAI/bge-small-en-v1.5", device=device)
    if kind == "none":
        class _None:
            name = "none"
            def score_batch(self, claim, passages):
                return [0.5] * len(passages) if passages else []
        return _None()
    raise ValueError(f"unknown verifier: {kind}")
