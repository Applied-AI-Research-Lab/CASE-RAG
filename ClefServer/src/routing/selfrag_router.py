from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field

from .router import ACTIONS, EvidenceState, RoutingDecision, _parse_router_json

SELF_RAG_PROMPT = """You are evaluating a question answering system's candidate answer.

Question: {query}
Candidate answer: {candidate}
Evidence:
{evidence}

Reflect on the evidence:
1. Is the candidate answer SUPPORTED by the evidence? (yes/no)
2. Would retrieving MORE evidence likely help reach a supported answer? (yes/no)
3. How confident are you that answering now is correct? (a number from 0.0 to 1.0)

Respond ONLY with compact JSON:
{{"supported": true/false, "useful": true/false, "confident": 0.0-1.0}}"""


@dataclass
class SelfRAGRouter:
    llm: object = None
    max_tokens: int = 100
    n_calls: int = 0
    latencies: list[float] = field(default_factory=list)
    n_fallbacks: int = 0

    def decide(self, state: EvidenceState) -> RoutingDecision:
        evidence = "\n".join(
            f"- {p.get('text', '')[:600]}" for p in state.passages[:6]) or "(none)"
        prompt = SELF_RAG_PROMPT.format(
            query=state.query,
            candidate=state.candidate_answer or "(none)",
            evidence=evidence)

        t0 = time.perf_counter()
        raw = self.llm.generate(prompt, max_tokens=self.max_tokens)
        self.latencies.append(time.perf_counter() - t0)
        self.n_calls += 1

        supported, useful, confident = True, False, 0.5
        try:
            parsed = _parse_router_json(raw)
            supported = bool(parsed.get("supported", True))
            useful = bool(parsed.get("useful", False))
            confident = float(parsed.get("confident", 0.5))
            confident = max(0.0, min(1.0, confident))
        except Exception:
            self.n_fallbacks += 1

        p_answer = confident if supported else 0.0
        p_retrieve = (1.0 - confident) * useful if not supported else useful * 0.2
        p_abstain = (1.0 - confident) * (1.0 - useful) if not supported else (1.0 - confident) * 0.3
        p_escalate = 0.0


        raw_probs = {"ANSWER": p_answer, "RETRIEVE_MORE": p_retrieve,
                     "ABSTAIN": p_abstain, "ESCALATE": p_escalate}
        total = sum(raw_probs.values())
        probs = {a: raw_probs[a] / total if total > 0 else 0.0 for a in ACTIONS}
        if total == 0:
            probs["ABSTAIN"] = 1.0

        action = max(probs, key=probs.get)
        return RoutingDecision(
            action=action,
            probabilities=probs,
            evidence_sufficiency=confident if supported else 0.0,
            estimated_factual_risk=1.0 - probs["ANSWER"],
            selected_evidence_ids=[p["id"] for p in state.passages],
        )

    def stats(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "mean_latency_s": sum(self.latencies) / len(self.latencies)
            if self.latencies else 0.0,
            "n_fallbacks": self.n_fallbacks,
        }
