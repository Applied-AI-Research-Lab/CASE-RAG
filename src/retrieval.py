from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievalHit:
    passage_id: str
    score: float


class BaseRetriever:
    name: str = "base"

    def build(self, passages: list[dict]) -> None:
        raise NotImplementedError

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        raise NotImplementedError


class BM25Ranker(BaseRetriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._index = None

    def build(self, passages: list[dict]) -> None:
        from rank_bm25 import BM25Okapi
        self._ids = [p["id"] for p in passages]
        docs = [tokenize((p.get("title") or "") + " " + p["text"]) for p in passages]
        docs = [d if d else ["empty"] for d in docs]
        self._index = BM25Okapi(docs)

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        scores = self._index.get_scores(tokenize(query))
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argsort(-scores)[:k]
        return [RetrievalHit(self._ids[i], float(scores[i])) for i in top]


class DenseRanker(BaseRetriever):
    name = "dense"

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 batch_size: int = 256, device: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._ids: list[str] = []
        self._emb = None
        self._model = None
        self._passages: dict[str, dict] = {}

    def build(self, passages: list[dict]) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._ids = [p["id"] for p in passages]
        self._passages = {p["id"]: p for p in passages}
        texts = [self._pre(p) for p in passages]
        self._emb = self._model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=True,
            normalize_embeddings=True).astype(np.float32)

    @staticmethod
    def _pre(p: dict) -> str:
        title = p.get("title") or ""
        text = p.get("text") or ""
        return (title + " " + text).strip()[:1024] or "empty"

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        q = self._model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores = (self._emb @ q.T).ravel()
        k = min(k, len(scores))
        top = np.argsort(-scores)[:k]
        return [RetrievalHit(self._ids[i], float(scores[i])) for i in top]


class HybridRanker(BaseRetriever):
    name = "hybrid"

    def __init__(self, bm25: BM25Ranker, dense: DenseRanker, k: int = 50,
                 rrf_k: int = 60) -> None:
        self.bm25, self.dense = bm25, dense
        self.candidate_k, self.rrf_k = k, rrf_k

    def build(self, passages: list[dict]) -> None:
        self.bm25.build(passages)
        self.dense.build(passages)

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        b = {h.passage_id: r for r, h in enumerate(self.bm25.rank(query, self.candidate_k))}
        d = {h.passage_id: r for r, h in enumerate(self.dense.rank(query, self.candidate_k))}
        fused: dict[str, float] = {}
        for src in (b, d):
            for pid, r in src.items():
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + r + 1)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [RetrievalHit(pid, s) for pid, s in ordered]


class NoneRetriever(BaseRetriever):
    name = "none"

    def build(self, passages: list[dict]) -> None:
        return None

    def rank(self, query: str, k: int) -> list[RetrievalHit]:
        return []


def build_retriever(kind: str, passages: list[dict], dense_model: str | None = None,
                    device: str | None = None) -> BaseRetriever:
    if kind == "none":
        r = NoneRetriever()
    elif kind == "bm25":
        r = BM25Ranker()
    elif kind == "dense":
        r = DenseRanker(model_name=dense_model or "BAAI/bge-small-en-v1.5", device=device)
    elif kind == "hybrid":
        r = HybridRanker(BM25Ranker(),
                         DenseRanker(model_name=dense_model or "BAAI/bge-small-en-v1.5",
                                     device=device))
    else:
        raise ValueError(f"unknown retriever: {kind}")
    r.build(passages)
    return r
