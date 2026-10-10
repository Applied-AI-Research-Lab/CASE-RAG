#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
OUT=${OUT:-runs/matrix}
GEMMA_URL=${GEMMA_URL:-http://127.0.0.1:8000/v1}
QWEN_URL=${QWEN_URL:-http://127.0.0.1:8001/v1}
GRANITE_URL=${GRANITE_URL:-http://127.0.0.1:8002/v1}
export CLEF_BASE_URL=${CLEF_BASE_URL:-http://127.0.0.1:8100}
mkdir -p "$OUT"

corpus_for () {
  case "$1" in
    scifact)  echo data/processed/corpus/scifact/corpus.jsonl ;;
    fever)    echo data/processed/corpus/fever/corpus_gold.jsonl ;;
    hotpotqa) echo data/processed/corpus/hotpotqa/corpus.jsonl ;;
  esac
}
calib_for () {
  case "$1" in
    scifact) echo "--calib-split train" ;;
    fever)   echo "--calib-split dev" ;;
    *)       echo "" ;;
  esac
}

run_one () {
  local ds=$1 rt=$2 seed=$3 bb=$4
  local out="$OUT/${bb}_${ds}_${rt}_s${seed}.json"
  [ -f "$out" ] && { echo "[skip] $(basename "$out")"; return 0; }

  local url model extra=""
  case "$bb" in
    qwen)
      url=$QWEN_URL; model=qwen3.8-27b-fp8
      extra="--disable-thinking --answer-max-tokens 512" ;;
    granite)
      url=$GRANITE_URL; model=granite-4.2-3b
      extra="--disable-thinking --answer-max-tokens 512" ;;
    *)
      url=$GEMMA_URL; model=gemma-4-31b-it
      extra="--answer-max-tokens 64" ;;
  esac

  echo "[run ] $bb $ds $rt s$seed ($(date +%H:%M))"
  $PY -m src.run_pilot \
    --dataset "$ds" --split dev --n 300 --calib 300 --k 5 \
    --max-retrievals 2 --corpus-max 70000 \
    --corpus-file "$(corpus_for "$ds")" \
    --retriever bm25 --verifier embedding \
    --router "$rt" \
    --backend vllm --llm-model "$model" --base-url "$url" \
    --clef-backend http \
    --answer-system "Output only the single requested answer. Do not explain, reason, or add any preamble." \
    $extra \
    --target-risk 0.15 --seed "$seed" $(calib_for "$ds") \
    --out "$out" > "$OUT/${bb}_${ds}_${rt}_s${seed}.log" 2>&1 \
    && echo "[done] $bb $ds $rt s$seed ($(date +%H:%M))" \
    || { echo "[FAIL] $bb $ds $rt s$seed -- see $OUT/${bb}_${ds}_${rt}_s${seed}.log"; rm -f "$out"; }
  return 0
}

BACKBONES=${1:-"gemma qwen"}
has () { case " $BACKBONES " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "backbones: $BACKBONES"

echo "=== 1. Clef on the primary backbone, all seeds (headline) ==="
has gemma && for seed in 1 2 3; do
  for ds in scifact fever hotpotqa; do run_one "$ds" clef "$seed" gemma; done
done

echo "=== 2. Baselines on the primary backbone ==="
has gemma && for seed in 1 2 3; do
  for rt in selfrag fixed feature; do
    for ds in scifact fever hotpotqa; do run_one "$ds" "$rt" "$seed" gemma; done
  done
done

echo "=== 3. Clef on the second backbone (SHARED-LINEAGE stress test) ==="
has qwen && for seed in 1 2 3; do
  for ds in scifact fever hotpotqa; do run_one "$ds" clef "$seed" qwen; done
done

echo "=== 4. Baselines on the second backbone ==="
has qwen && for seed in 1 2 3; do
  for rt in selfrag fixed feature; do
    for ds in scifact fever hotpotqa; do run_one "$ds" "$rt" "$seed" qwen; done
  done
done

echo "=== 5. Small unrelated-lineage contrast (optional) ==="
has granite && for seed in 1 2 3; do
  for ds in scifact fever hotpotqa; do run_one "$ds" clef "$seed" granite; done
done
has granite && for seed in 1 2 3; do
  for rt in selfrag fixed feature; do
    for ds in scifact fever hotpotqa; do run_one "$ds" "$rt" "$seed" granite; done
  done
done

echo "MATRIX COMPLETE $(date +%H:%M): $(ls "$OUT"/*.json 2>/dev/null | grep -vc records) runs"
