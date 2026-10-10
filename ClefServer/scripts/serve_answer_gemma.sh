#!/usr/bin/env bash
set -eu
PORT=${GEMMA_PORT:-8000}
GPU=${GEMMA_GPU:-1}
MODEL=${GEMMA_MODEL:-google/gemma-4-31B-it}

echo "[gemma] serving $MODEL on GPU $GPU, port $PORT"
exec env CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --served-model-name gemma-4-31b-it
