#!/usr/bin/env bash
set -eu
PORT=${QWEN_PORT:-8001}
GPU=${QWEN_GPU:-2}
export VLLM_USE_DEEP_GEMM=0
MODEL=${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}

echo "[qwen] serving $MODEL on GPU $GPU, port $PORT"
exec env CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 16384 \
  --served-model-name qwen3.8-27b-fp8
