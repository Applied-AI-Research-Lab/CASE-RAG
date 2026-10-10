from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from .router import ACTIONS, EvidenceState, RoutingDecision


class JevUnavailable(RuntimeError):
    pass


@dataclass
class JevRouter:
    api_key: str = ""
    model: str = "speed_latest"
    timeout_s: int = 30
    n_calls: int = 0
    latencies: list[float] = field(default_factory=list)
    last_usage: dict = field(default_factory=dict)
    input_tokens_total: int = 0
    output_tokens_total: int = 0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("TYPESAFE_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)


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


    def decide(self, state: EvidenceState) -> RoutingDecision:
        if not self.available():
            raise JevUnavailable("TYPESAFE_API_KEY not set")

        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

        criteria = {
            "ANSWER": "The evidence clearly supports the candidate answer; respond now.",
            "RETRIEVE_MORE": (
                "The evidence is insufficient or unclear, and more retrieval "
                "is likely to find better support."
            ),
            "ABSTAIN": (
                "The evidence is insufficient and further retrieval is unlikely "
                "to help; the system should not answer."
            ),
            "ESCALATE": (
                "The question is ambiguous, sensitive, contested, or needs "
                "human/domain-expert judgment."
            ),
        }
        questions = {
            "route": Choice(
                instructions=(
                    "Choose the next action for a question-answering system "
                    "deciding whether to answer with the current evidence."
                ),
                criteria=criteria,
            ),
            "evidence_supports_answer": Noul(
                instructions=(
                    "The retrieved evidence supports the candidate answer to "
                    "the question."
                )
            ),
            "evidence_quality": Score(
                instructions="How sufficient and relevant the retrieved evidence is.",
                criteria=[
                    "Evidence is irrelevant or empty.",
                    "Evidence is topically related but does not address the question.",
                    "Evidence partially addresses the question.",
                    "Evidence largely addresses the question with minor gaps.",
                    "Evidence fully and clearly supports answering now.",
                ],
            ),
        }

        t0 = time.perf_counter()


        max_attempts = getattr(self, "max_retries", 5)
        response = None
        last_exc: Exception | None = None
        try:
            for attempt in range(max_attempts):
                try:
                    with TypeSafeClient(api_key=self.api_key) as client:
                        response = client.system_one(
                            self._state_text(state), questions, model=self.model)
                    break
                except Exception as exc:
                    last_exc = exc
                    self.n_transient_errors = getattr(self, "n_transient_errors", 0) + 1
                    if attempt < max_attempts - 1:
                        time.sleep(min(2 ** attempt, 16))
            if response is None:
                raise JevUnavailable(
                    f"Jev call failed after {max_attempts} attempts: {last_exc}"
                ) from last_exc
        finally:
            self.latencies.append(time.perf_counter() - t0)
            self.n_calls += 1

        probs_raw = response.choices["route"].probabilities
        probs = {a: float(probs_raw.get(a, 0.0)) for a in ACTIONS}
        s = sum(probs.values()) or 1.0
        probs = {a: v / s for a, v in probs.items()}

        action = max(probs, key=probs.get)
        suff = float(response.nouls["evidence_supports_answer"].noul)
        qual = list(response.scores["evidence_quality"].probabilities.values())
        quality = float(sum((i + 1) / 5 * p for i, p in enumerate(qual))) if qual else suff

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
                "input_tokens_total": getattr(usage, "input_tokens_total", None),
                "output_tokens_total": getattr(usage, "output_tokens_total", None),
                "latency": getattr(usage, "latency", None),
            }
            self.input_tokens_total += int(getattr(usage, "input_tokens_total", 0)
                                            or getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens_total += int(getattr(usage, "output_tokens_total", 0)
                                            or getattr(usage, "output_tokens", 0) or 0)

        return RoutingDecision(
            action=action,
            probabilities=probs,
            evidence_sufficiency=max(0.0, min(1.0, (suff + quality) / 2)),
            estimated_factual_risk=1.0 - probs["ANSWER"],
            selected_evidence_ids=[p["id"] for p in state.passages],
        )

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "mean_latency_s": sum(self.latencies) / len(self.latencies)
            if self.latencies else 0.0,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "last_usage": self.last_usage,
        }
