#!/usr/bin/env bash
set -u
CLEF=${CLEF_BASE_URL:-http://127.0.0.1:8100}
GEMMA=${GEMMA_URL:-http://127.0.0.1:8000/v1}
QWEN=${QWEN_URL:-http://127.0.0.1:8001/v1}
ok=0; bad=0

probe () {
  local name=$1 url=$2
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 15 "$url" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then echo "  OK    $name  ($url)"; ok=$((ok+1))
  else echo "  DOWN  $name  ($url)  HTTP $code"; bad=$((bad+1)); fi
}

echo "endpoints:"
probe clef  "$CLEF/health"
probe gemma "$GEMMA/models"
probe qwen  "$QWEN/models"

echo
echo "GPU memory:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/  /' || echo "  (nvidia-smi unavailable)"

echo
if [ "$bad" -eq 0 ]; then
  echo "ALL ENDPOINTS UP — safe to run the smoke tests (section 3 of the runbook)."
else
  echo "$bad endpoint(s) down. Check: tmux attach -t clef|gemma|qwen, or tail logs/*.log"
  exit 1
fi
