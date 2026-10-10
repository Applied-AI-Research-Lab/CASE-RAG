from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Sequence

ACTIONS = ("ANSWER", "RETRIEVE_MORE", "ABSTAIN", "ESCALATE")


class Action(str, Enum):
    ANSWER = "ANSWER"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


@dataclass
class EvidenceState:
    query: str
    passages: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    verifier_scores: list[float] = field(default_factory=list)
    candidate_answer: str = ""
    candidate_confidence: float = 0.0
    retrieval_steps: int = 0
    max_retrievals: int = 3

    @property
    def evidence_sufficiency(self) -> float:
        if not self.verifier_scores:
            return 0.0
        return max(0.0, min(1.0, max(self.verifier_scores)))

    @property
    def uncertainty(self) -> float:
        c = min(max(self.candidate_confidence, 1e-6), 1 - 1e-6)
        return -(c * math.log(c) + (1 - c) * math.log(1 - c)) / math.log(2)


@dataclass
class RoutingDecision:
    action: str
    probabilities: dict[str, float]
    evidence_sufficiency: float
    estimated_factual_risk: float
    selected_evidence_ids: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def softmax(xs: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(xs)
    exps = [math.exp((x - m) / temperature) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


class FeatureRouter:
    def __init__(self, answer_threshold: float = 0.62,
                 retrieve_gain: float = 2.2, escalate_threshold: float = 0.35) -> None:
        self.answer_threshold = answer_threshold
        self.retrieve_gain = retrieve_gain
        self.escalate_threshold = escalate_threshold

    def decide(self, state: EvidenceState) -> RoutingDecision:
        suff = state.evidence_sufficiency
        conf = state.candidate_confidence
        budget_left = state.max_retrievals - state.retrieval_steps


        logit_answer = 2.0 * conf + 2.5 * suff - 1.2
        logit_retrieve = self.retrieve_gain * (1.0 - suff) + 0.8 * (1.0 - conf)
        if budget_left <= 0:
            logit_retrieve = -10.0
        logit_abstain = 1.6 * (1.0 - conf) * (1.0 - suff) - 0.2
        logit_esc = 0.0
        if suff < self.escalate_threshold and budget_left <= 0:
            logit_esc = 1.0

        probs = dict(zip(ACTIONS, softmax(
            [logit_answer, logit_retrieve, logit_abstain, logit_esc])))

        action = max(probs, key=probs.get)
        risk = 1.0 - (0.5 * conf + 0.5 * suff)
        return RoutingDecision(
            action=action,
            probabilities=probs,
            evidence_sufficiency=suff,
            estimated_factual_risk=risk,
            selected_evidence_ids=[p["id"] for p in state.passages],
        )


class FixedBudgetRouter:
    def __init__(self, force_steps: int | None = None) -> None:
        self.force_steps = force_steps

    def decide(self, state: EvidenceState) -> RoutingDecision:
        suff = state.evidence_sufficiency
        conf = state.candidate_confidence
        target = (state.max_retrievals if self.force_steps is None
                  else min(self.force_steps, state.max_retrievals))
        more = state.retrieval_steps < target
        score = max(0.0, min(1.0, 0.5 * conf + 0.5 * suff))
        if more:
            probs = {"ANSWER": 0.0, "RETRIEVE_MORE": 1.0,
                     "ABSTAIN": 0.0, "ESCALATE": 0.0}
            action = "RETRIEVE_MORE"
        else:
            probs = {"ANSWER": score, "RETRIEVE_MORE": 0.0,
                     "ABSTAIN": 1.0 - score, "ESCALATE": 0.0}
            action = "ANSWER"
        return RoutingDecision(
            action=action,
            probabilities=probs,
            evidence_sufficiency=suff,
            estimated_factual_risk=1.0 - score,
            selected_evidence_ids=[p["id"] for p in state.passages],
        )


ROUTER_PROMPT = """You are a reliability router for a question answering system.
Given the question, retrieved evidence, and a candidate answer, decide the next action.

Actions:
- ANSWER: the candidate answer is supported by the evidence
- RETRIEVE_MORE: evidence is insufficient; more retrieval may help
- ABSTAIN: evidence is insufficient and further retrieval is unlikely to help
- ESCALATE: the question is ambiguous, sensitive, or requires human judgment

Question: {query}
Candidate answer: {candidate}
Evidence:
{evidence}

Respond ONLY with compact JSON:
{{"action": "ANSWER|RETRIEVE_MORE|ABSTAIN|ESCALATE",
  "probabilities": {{"ANSWER": p1, "RETRIEVE_MORE": p2, "ABSTAIN": p3, "ESCALATE": p4}},
  "evidence_sufficiency": s, "estimated_factual_risk": r}}
"""


class LLMRouter:
    def __init__(self, llm, fallback: FeatureRouter | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or FeatureRouter()

    def decide(self, state: EvidenceState) -> RoutingDecision:
        evidence = "\n".join(
            f"[{p['id']}] {p.get('title','')}: {p.get('text','')[:400]}"
            for p in state.passages) or "(none)"
        prompt = ROUTER_PROMPT.format(
            query=state.query, candidate=state.candidate_answer or "(none)",
            evidence=evidence)
        raw = self.llm.generate(prompt, max_tokens=220)
        try:
            parsed = _parse_router_json(raw)
            probs = parsed["probabilities"]
            missing = [a for a in ACTIONS if a not in probs]
            if missing:
                raise ValueError(f"missing actions: {missing}")
            s = sum(probs.values()) or 1.0
            probs = {a: max(0.0, probs[a]) / s for a in ACTIONS}
            action = parsed.get("action", max(probs, key=probs.get))
            if action not in ACTIONS:
                action = max(probs, key=probs.get)
            return RoutingDecision(
                action=action,
                probabilities=probs,
                evidence_sufficiency=float(parsed.get("evidence_sufficiency",
                                                       state.evidence_sufficiency)),
                estimated_factual_risk=float(parsed.get("estimated_factual_risk", 0.5)),
                selected_evidence_ids=[p["id"] for p in state.passages],
            )
        except Exception:
            return self.fallback.decide(state)


def _parse_router_json(text: str) -> dict:
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON")
