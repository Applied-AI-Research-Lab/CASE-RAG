from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

from .router import ACTIONS, EvidenceState, RoutingDecision


class ClefUnavailable(RuntimeError):
    pass


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e) or 1.0
    return [v / s for v in e]


ROUTE_CRITERIA = {
    "ANSWER": "The evidence clearly supports the candidate answer; respond now.",
    "RETRIEVE_MORE": (
        "The evidence is insufficient or unclear, and more retrieval is likely "
        "to find better support."
    ),
    "ABSTAIN": (
        "The evidence is insufficient and further retrieval is unlikely to "
        "help; the system should not answer."
    ),
    "ESCALATE": (
        "The question is ambiguous, sensitive, contested, or needs "
        "human/domain-expert judgment."
    ),
}


@dataclass
class ClefRouter:
    backend: str = "http"
    base_url: str = ""
    model_path: str = "Cloudflare/clef"
    device: str = "cuda"
    timeout_s: int = 60
    max_retries: int = 5
    n_calls: int = 0
    n_transient_errors: int = 0
    latencies: list[float] = field(default_factory=list)
    _model: object = None
    _processor: object = None
    _jsm: object = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url or os.environ.get(
            "CLEF_BASE_URL", "http://127.0.0.1:8100")

    def available(self) -> bool:
        if self.backend == "local":
            return True
        try:
            import requests
            r = requests.get(f"{self.base_url}/health", timeout=10)
            return r.status_code < 500
        except Exception:
            return False


    def _state_text(self, state: EvidenceState) -> str:
        lines = [
            f"QUESTION: {state.query}",
            f"CANDIDATE ANSWER: {state.candidate_answer or '(none)'}",
            f"VERIFIER BEST SCORE: {state.evidence_sufficiency:.3f}",
            f"RETRIEVAL STEPS USED: {state.retrieval_steps} of {state.max_retrievals}",
            "EVIDENCE:",
        ]
        for p in state.passages[:5]:
            lines.append(f"- {p.get('title','')}: {p.get('text','')[:600]}")
        if not state.passages:
            lines.append("(no evidence retrieved)")
        return "\n".join(lines)

    def _record(self, state: EvidenceState) -> dict:
        return {
            "state": self._state_text(state),
            "questions": {
                "route": {"type": "choice", "criteria": ROUTE_CRITERIA},
                "evidence_supports_answer": {
                    "type": "noul",
                    "instructions": ("The retrieved evidence supports the "
                                     "candidate answer to the question."),
                },
                "evidence_quality": {
                    "type": "score",
                    "criteria": [
                        "Evidence is irrelevant or empty.",
                        "Evidence is topically related but does not address the question.",
                        "Evidence partially addresses the question.",
                        "Evidence largely addresses the question with minor gaps.",
                        "Evidence fully and clearly supports answering now.",
                    ],
                },
            },
        }


    def _call_http(self, record: dict) -> dict:
        import requests
        r = requests.post(f"{self.base_url}/v1/decide", json=record,
                          timeout=self.timeout_s)
        if r.status_code != 200:
            raise ClefUnavailable(f"Clef HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def _load_local(self):
        if self._model is not None:
            return
        try:
            from .clef_runtime import load_clef
            self._model, self._processor, self._jsm = load_clef(
                self.model_path, device=self.device)
        except (ImportError, OSError) as exc:
            raise ClefUnavailable(
                f"could not load {self.model_path} locally ({exc}). "
                "Download it with `hf download Cloudflare/clef`, or use "
                "--clef-backend http."
            ) from exc

    def _call_local(self, record: dict) -> dict:
        from .clef_runtime import score_record
        self._load_local()
        logits, _ = score_record(self._model, self._processor, self._jsm, record)
        return {"logits": logits}

    @staticmethod
    def _route_probs(payload: dict) -> dict:
        if "choices" in payload and "route" in payload["choices"]:
            raw = payload["choices"]["route"].get("probabilities", {})
            return {a: float(raw.get(a, 0.0)) for a in ACTIONS}
        logits = payload.get("logits", {}).get("route")
        if logits is None:
            raise ClefUnavailable(f"unrecognised Clef payload: {list(payload)[:5]}")
        if isinstance(logits, dict):
            vals = [float(logits.get(a, -1e9)) for a in ACTIONS]
        else:
            if len(logits) != len(ACTIONS):
                raise ClefUnavailable(
                    f"Clef returned {len(logits)} route logits, expected {len(ACTIONS)}")
            vals = [float(x) for x in logits]
        return dict(zip(ACTIONS, _softmax(vals)))


    def decide(self, state: EvidenceState) -> RoutingDecision:
        record = self._record(state)
        caller = self._call_local if self.backend == "local" else self._call_http
        t0 = time.perf_counter()
        payload = None
        last_exc: Exception | None = None
        try:
            for attempt in range(self.max_retries):
                try:
                    payload = caller(record)
                    break
                except Exception as exc:
                    last_exc = exc
                    self.n_transient_errors += 1
                    if attempt < self.max_retries - 1:
                        time.sleep(min(2 ** attempt, 16))
            if payload is None:
                raise ClefUnavailable(
                    f"Clef call failed after {self.max_retries} attempts: {last_exc}"
                ) from last_exc
        finally:
            self.latencies.append(time.perf_counter() - t0)
            self.n_calls += 1

        probs = self._route_probs(payload)
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        action = max(probs, key=probs.get)

        suff = state.evidence_sufficiency
        return RoutingDecision(
            action=action,
            probabilities=probs,
            evidence_sufficiency=suff,
            estimated_factual_risk=1.0 - probs.get("ANSWER", 0.0),
            selected_evidence_ids=[p["id"] for p in state.passages],
        )

    def stats(self) -> dict:
        n = len(self.latencies) or 1
        return {"n_calls": self.n_calls,
                "mean_latency_s": sum(self.latencies) / n,
                "n_transient_errors": self.n_transient_errors}
