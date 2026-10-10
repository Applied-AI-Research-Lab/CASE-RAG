#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
OUT=runs/matrix_glm
MODEL=glm-5.3:cloud
mkdir -p "$OUT"
if [ -z "${TYPESAFE_API_KEY:-}" ] && [ -f ../api-keys.txt ]; then
  export TYPESAFE_API_KEY=$(sed -n 's/^[[:space:]]*TYPESAFE_API_KEY[[:space:]]*[=:][[:space:]]*//p' ../api-keys.txt | head -1 | tr -d '"'"'"'[:space:]')
fi

quota_ok () {
  local body
  body=$(curl -s -m 30 -X POST http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" 2>/dev/null)
  case "$body" in *"429"*|*"5-hour limit"*|*"rate"*) return 1;; esac
  case "$body" in *'"choices"'*) return 0;; esac
  return 1
}

wait_for_quota () {
  local waited=0
  while ! quota_ok; do
    echo "[quota] capped, waiting 15m (waited ${waited}m so far, $(date +%H:%M))"
    sleep 900; waited=$((waited+15))
    [ "$waited" -ge 360 ] && { echo "[quota] still capped after 6h, giving up"; return 1; }
  done
  echo "[quota] available ($(date +%H:%M))"
  return 0
}

corpus_for () { case "$1" in scifact) echo data/processed/corpus/scifact/corpus.jsonl;; fever) echo data/processed/corpus/fever/corpus_gold.jsonl;; hotpotqa) echo data/processed/corpus/hotpotqa/corpus.jsonl;; esac; }
calib_for  () { case "$1" in scifact) echo "--calib-split train";; fever) echo "--calib-split dev";; *) echo "";; esac; }

run_one () {
  local ds=$1 rt=$2 seed=$3
  local out="$OUT/${ds}_${rt}_s${seed}.json"
  [ -f "$out" ] && return 0
  wait_for_quota || return 1
  echo "[run ] $ds $rt s$seed ($(date +%H:%M))"
  $PY -m src.run_pilot --dataset "$ds" --split dev --n 300 --calib 300 --k 5 \
    --max-retrievals 2 --corpus-max 70000 --corpus-file "$(corpus_for "$ds")" \
    --retriever bm25 --verifier embedding --router "$rt" \
    --backend ollama --llm-model "$MODEL" \
    --answer-max-tokens 1024 \
    --answer-system "Output only the single requested answer. Do not explain, reason, or add any preamble." \
    --target-risk 0.15 --seed "$seed" $(calib_for "$ds") \
    --out "$out" > "$OUT/${ds}_${rt}_s${seed}.log" 2>&1 \
    && echo "[done] $ds $rt s$seed ($(date +%H:%M))" \
    || { echo "[FAIL] $ds $rt s$seed"; rm -f "$out"; }
  return 0
}

for seed in 1 2 3; do
  for ds in scifact fever hotpotqa; do run_one "$ds" jev "$seed"; done
done

n_jev=$(ls "$OUT"/*_jev_s*.json 2>/dev/null | grep -vc records)
echo "[jev ] $n_jev/9 complete"

if [ "${GLM_FULL_MATRIX:-0}" != "1" ]; then
  echo "GLM_JEV_SET_COMPLETE $(date +%H:%M) — stopping here by design."
  echo "  The cross-backbone claim now rests on 3 seeds per dataset."
  echo "  Set GLM_FULL_MATRIX=1 to continue with the remaining baselines."
  exit 0
fi

echo "=== GLM_FULL_MATRIX=1: continuing with the remaining baselines ==="
for ds in fever hotpotqa;         do run_one "$ds" selfrag 1; done
for ds in scifact fever hotpotqa; do run_one "$ds" fixed   1; done
for ds in scifact fever hotpotqa; do run_one "$ds" feature 1; done
for seed in 2 3; do
  for ds in scifact fever hotpotqa; do run_one "$ds" selfrag "$seed"; done
done
for seed in 2 3; do
  for rt in fixed feature; do
    for ds in scifact fever hotpotqa; do run_one "$ds" "$rt" "$seed"; done
  done
done

echo "GLM_QUOTA_DRIVER_FINISHED $(date +%H:%M) runs=$(ls $OUT/*.json 2>/dev/null | grep -vc records)"
