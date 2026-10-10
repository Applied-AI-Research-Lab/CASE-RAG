#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
PORT=${CLEF_PORT:-8100}
GPU=${CLEF_GPU:-0}
MODEL=${CLEF_MODEL:-Cloudflare/clef}

echo "[clef] serving $MODEL on GPU $GPU, port $PORT"
exec env CUDA_VISIBLE_DEVICES=$GPU python3 -m src.clef_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-length 16384 \
  --served-model-name clef
