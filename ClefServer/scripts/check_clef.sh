#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export CLEF_BASE_URL=${CLEF_BASE_URL:-http://127.0.0.1:8100}

python3 - <<'PY'
import os, sys, time
from src.routing.clef_adapter import ClefRouter
from src.routing.router import ACTIONS, EvidenceState

r = ClefRouter(backend="http", max_retries=1)
if not r.available():
    print(f"DOWN: {r.base_url}/health does not answer -- is the clef session up?")
    sys.exit(1)

def state(passages, suff):
    return EvidenceState(query="Water boils at 100 degrees Celsius at sea level.",
                         passages=passages, verifier_scores=[suff],
                         candidate_answer="supported", candidate_confidence=0.9,
                         retrieval_steps=0, max_retrievals=2)

good = state([{"id": "p1", "title": "Boiling point",
               "text": "At standard atmospheric pressure (sea level), pure water "
                       "boils at 100 degrees Celsius (212 Fahrenheit)."}], 0.95)
empty = state([], 0.0)

ok = True
for name, s in (("supported", good), ("no evidence", empty)):
    t0 = time.perf_counter()
    d = r.decide(s)
    dt = time.perf_counter() - t0
    tot = sum(d.probabilities.values())
    print(f"  {name:12s} -> {d.action:14s} "
          + "  ".join(f"{a}={d.probabilities[a]:.3f}" for a in ACTIONS)
          + f"   ({dt*1000:.0f} ms)")
    ok &= abs(tot - 1.0) < 1e-3 and set(d.probabilities) == set(ACTIONS)
    if name == "supported":
        p_good = d.probabilities["ANSWER"]
    else:
        p_empty = d.probabilities["ANSWER"]

if p_good <= p_empty:
    print("  ANSWER is not more likely with supporting evidence than with none.")
    ok = False
print("CLEF OK" if ok else "CLEF CHECK FAILED -- see logs/clef.log")
sys.exit(0 if ok else 1)
PY
