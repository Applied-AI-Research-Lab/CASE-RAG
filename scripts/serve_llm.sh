#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${2:-8000}"
TP="${3:-1}"

echo "serving $MODEL on :$PORT (tensor_parallel=$TP)"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
