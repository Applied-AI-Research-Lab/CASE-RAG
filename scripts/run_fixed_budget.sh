#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
OUT=runs/matrix
LOGDIR=runs/matrix
mkdir -p "$OUT"

run_one () {
  local ds=$1 seed=$2 corpus=$3 calibsplit=$4
  local out="$OUT/${ds}_fixed_s${seed}.json"
  if [ -f "$out" ]; then echo "[skip] $out exists"; return 0; fi
  local extra=""
  [ -n "$calibsplit" ] && extra="--calib-split $calibsplit"
  echo "[run ] $ds seed=$seed -> $out"
  $PY -m src.run_pilot \
    --dataset "$ds" --split dev --n 300 --calib 300 --k 5 \
    --max-retrievals 2 --corpus-max 70000 \
    --corpus-file "$corpus" \
    --retriever bm25 --verifier embedding \
    --router fixed \
    --backend ollama --llm-model gemma4:31b-cloud \
    --target-risk 0.15 --seed "$seed" $extra \
    --out "$out" > "$LOGDIR/${ds}_fixed_s${seed}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then echo "[done] $ds seed=$seed"; else echo "[FAIL] $ds seed=$seed rc=$rc (see $LOGDIR/${ds}_fixed_s${seed}.log)"; fi
  return 0
}

for seed in 1 2 3; do
  run_one scifact  "$seed" data/processed/corpus/scifact/corpus.jsonl      train
  run_one fever    "$seed" data/processed/corpus/fever/corpus_gold.jsonl   dev
  run_one hotpotqa "$seed" data/processed/corpus/hotpotqa/corpus.jsonl     ""
done

echo "ALL FIXED-BUDGET RUNS COMPLETE"
ls -la "$OUT"/*_fixed_s*.json 2>/dev/null
