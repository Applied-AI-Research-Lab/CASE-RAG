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
corpus_for () { case "$1" in scifact) echo data/processed/corpus/scifact/corpus.jsonl;; fever) echo data/processed/corpus/fever/corpus_gold.jsonl;; hotpotqa) echo data/processed/corpus/hotpotqa/corpus.jsonl;; esac; }
calib_for  () { case "$1" in scifact) echo "--calib-split train";; fever) echo "--calib-split dev";; *) echo "";; esac; }
for pass in 1 2 3 4 5 6; do
  pending=0
  for seed in 1 2 3; do
    for ds in scifact fever hotpotqa; do
      out="$OUT/${ds}_jev_s${seed}.json"
      [ -f "$out" ] && continue
      pending=1
      echo "[jev ] $ds s$seed ($(date +%H:%M))"
      $PY -m src.run_pilot --dataset "$ds" --split dev --n 300 --calib 300 --k 5 \
        --max-retrievals 2 --corpus-max 70000 --corpus-file "$(corpus_for "$ds")" \
        --retriever bm25 --verifier embedding --router jev \
        --backend ollama --llm-model "$MODEL" \
        --answer-max-tokens 1024 \
        --answer-system "Output only the single requested answer. Do not explain, reason, or add any preamble." \
        --target-risk 0.15 --seed "$seed" $(calib_for "$ds") \
        --out "$out" > "$OUT/${ds}_jev_s${seed}.log" 2>&1 \
        && echo "[done] $ds jev s$seed" || { echo "[FAIL] $ds jev s$seed — API likely down, waiting 10m"; sleep 600; }
    done
  done
  [ "$pending" -eq 0 ] && break
done
echo "JEV_RUNS_FINISHED $(date +%H:%M)"
