#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
OUT=runs/matrix_glm
MODEL=glm-5.3:cloud
mkdir -p "$OUT"

while pgrep -f "run_fixed_budget.sh" > /dev/null; do
  echo "[wait] forced-budget matrix still running ($(date +%H:%M))"
  sleep 300
done
echo "[ok  ] forced-budget matrix finished; starting $MODEL backbone at $(date +%H:%M)"

if [ -z "${TYPESAFE_API_KEY:-}" ] && [ -f ../api-keys.txt ]; then
  _k=$(sed -n 's/^[[:space:]]*TYPESAFE_API_KEY[[:space:]]*[=:][[:space:]]*//p' ../api-keys.txt \
       | head -1 | tr -d '"'"'"'[:space:]')
  if [ -n "$_k" ]; then
    export TYPESAFE_API_KEY="$_k"
    echo "[key ] TYPESAFE_API_KEY loaded from api-keys.txt (${#TYPESAFE_API_KEY} chars)"
  else
    echo "[key ] WARNING: could not parse TYPESAFE_API_KEY from api-keys.txt"
  fi
  unset _k
elif [ -n "${TYPESAFE_API_KEY:-}" ]; then
  echo "[key ] TYPESAFE_API_KEY already present in environment"
else
  echo "[key ] WARNING: no TYPESAFE_API_KEY and no ../api-keys.txt — jev runs will be skipped"
fi

corpus_for () {
  case "$1" in
    scifact)  echo data/processed/corpus/scifact/corpus.jsonl ;;
    fever)    echo data/processed/corpus/fever/corpus_gold.jsonl ;;
    hotpotqa) echo data/processed/corpus/hotpotqa/corpus.jsonl ;;
  esac
}
calibsplit_for () {
  case "$1" in
    scifact) echo train ;;
    fever)   echo dev ;;
    *)       echo "" ;;
  esac
}

jev_up () {
  [ -z "${TYPESAFE_API_KEY:-}" ] && return 1
  ../.venv/bin/python - <<'PY' >/dev/null 2>&1
import sys
sys.path.insert(0, ".")
from src.routing.jev_adapter import JevRouter
from src.routing import EvidenceState
st = EvidenceState(query="probe", candidate_answer="yes", candidate_confidence=0.5,
                   passages=[{"id": "p", "title": "t", "text": "probe text"}],
                   scores=[0.5], verifier_scores=[0.5], retrieval_steps=0, max_retrievals=1)
JevRouter().decide(st)
PY
}

run_one () {
  local ds=$1 rt=$2 seed=$3
  local out="$OUT/${ds}_${rt}_s${seed}.json"
  if [ -f "$out" ]; then echo "[skip] $out"; return 0; fi
  if [ "$rt" = "jev" ]; then
    if [ -z "${TYPESAFE_API_KEY:-}" ]; then
      echo "[SKIP-NOKEY] $ds jev s$seed — TYPESAFE_API_KEY not set"
      return 0
    fi
    if ! jev_up; then
      echo "[SKIP-DOWN] $ds jev s$seed — Jev API unhealthy, deferring ($(date +%H:%M))"
      return 0
    fi
  fi
  local cs extra=""
  cs=$(calibsplit_for "$ds")
  [ -n "$cs" ] && extra="--calib-split $cs"
  echo "[run ] $ds $rt s$seed ($(date +%H:%M))"
  $PY -m src.run_pilot \
    --dataset "$ds" --split dev --n 300 --calib 300 --k 5 \
    --max-retrievals 2 --corpus-max 70000 \
    --corpus-file "$(corpus_for "$ds")" \
    --retriever bm25 --verifier embedding \
    --router "$rt" \
    --backend ollama --llm-model "$MODEL" \
    --answer-max-tokens 1024 \
    --answer-system "Output only the single requested answer. Do not explain, reason, or add any preamble." \
    --target-risk 0.15 --seed "$seed" $extra \
    --out "$out" > "$OUT/${ds}_${rt}_s${seed}.log" 2>&1
  if [ $? -eq 0 ]; then echo "[done] $ds $rt s$seed"; else echo "[FAIL] $ds $rt s$seed (see $OUT/${ds}_${rt}_s${seed}.log)"; fi
  return 0
}

retry_pending_jev () {
  local seed=$1 d
  for d in scifact fever hotpotqa; do
    [ -f "$OUT/${d}_jev_s${seed}.json" ] || run_one "$d" jev "$seed"
  done
}

for seed in 1 2 3; do
  retry_pending_jev "$seed"
  for rt in selfrag fixed feature; do
    for ds in scifact fever hotpotqa; do
      retry_pending_jev "$seed"
      run_one "$ds" "$rt" "$seed"
    done
  done
  retry_pending_jev "$seed"
done

echo "GLM BACKBONE MATRIX COMPLETE $(date +%H:%M)"
ls "$OUT"/*.json 2>/dev/null | grep -v records | wc -l | xargs echo "runs completed:"
