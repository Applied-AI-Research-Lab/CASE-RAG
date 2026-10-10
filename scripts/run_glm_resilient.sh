#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for pass in 1 2 3; do
  echo "=== PASS $pass ($(date +%H:%M)) ==="
  bash scripts/run_glm_backbone.sh
  n=$(ls runs/matrix_glm/*.json 2>/dev/null | grep -vc records)
  echo "=== after pass $pass: $n runs complete ==="
  [ "$n" -ge 36 ] && break
done
echo "GLM_MATRIX_FINISHED $(date +%H:%M)"
