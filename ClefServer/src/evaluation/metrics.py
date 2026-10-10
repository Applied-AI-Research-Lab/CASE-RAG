from __future__ import annotations

from typing import Sequence

import numpy as np


def risk_coverage_curve(conf: Sequence[float], correct: Sequence[float],
                        points: int = 20) -> list[dict]:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-conf)
    sorted_correct = correct[order]
    out = []
    n = len(conf)
    for i in range(0, n, max(1, n // points)):
        k = i + 1
        answered = sorted_correct[:k]
        out.append({
            "coverage": k / n,
            "risk": float(1.0 - answered.mean()),
            "accuracy": float(answered.mean()),
            "k": k,
        })
    out.append({"coverage": 1.0, "risk": float(1.0 - correct.mean()),
                "accuracy": float(correct.mean()), "k": n})

    seen, dedup = set(), []
    for p in out:
        if p["coverage"] not in seen:
            seen.add(p["coverage"])
            dedup.append(p)
    return dedup


def selective_risk_at_coverage(conf: Sequence[float], correct: Sequence[float],
                               target_coverage: float) -> float:
    curve = risk_coverage_curve(conf, correct, points=100)
    best = min(curve, key=lambda p: abs(p["coverage"] - target_coverage))
    return best["risk"]


def area_over_risk_coverage(conf: Sequence[float], correct: Sequence[float],
                            points: int = 50) -> float:
    curve = risk_coverage_curve(conf, correct, points=points)
    xs = [p["coverage"] for p in curve]
    ys = [p["risk"] for p in curve]
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(ys, xs))


def abstention_metrics(decisions: Sequence[str], correct: Sequence[bool]) -> dict:
    decisions = list(decisions)
    correct = list(correct)
    n = len(decisions)
    answered = [d in ("ANSWER",) for d in decisions]
    abstained = [d == "ABSTAIN" for d in decisions]
    escalated = [d == "ESCALATE" for d in decisions]
    n_ans = sum(answered)
    if n_ans:
        answered_correct = sum(c for a, c in zip(answered, correct) if a)
        answered_risk = 1 - answered_correct / n_ans
    else:
        answered_risk = 0.0
    unnecessary_abstain = sum(1 for a, c in zip(abstained, correct) if c)
    n_abstain = sum(abstained)
    return {
        "coverage": n_ans / n if n else 0.0,
        "answered_risk": answered_risk,
        "abstain_rate": n_abstain / n if n else 0.0,
        "abstain_precision": unnecessary_abstain / n_abstain if n_abstain else 0.0,
        "escalate_rate": sum(escalated) / n if n else 0.0,
    }


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    stats = [float(stat(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def paired_bootstrap_delta(a: Sequence[float], b: Sequence[float],
                           n_boot: int = 1000, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, "paired arrays must align"
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), size=len(a))
        deltas.append(float(np.mean(a[idx] - b[idx])))
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta_mean": float(np.mean(a - b)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
