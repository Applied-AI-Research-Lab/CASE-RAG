#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"

log() { printf '\033[1;34m[download]\033[0m %s\n' "$*"; }

fetch() {
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then
    log "SKIP (exists): $out"
  else
    log "GET $url"
    curl -L --fail --retry 3 --retry-delay 2 --continue-at - -o "$out" "$url"
  fi
}

mkdir -p "$RAW/fever"
FEVER_BASE="https://fever.ai/download/fever"
fetch "$FEVER_BASE/train.jsonl"            "$RAW/fever/train.jsonl"
fetch "$FEVER_BASE/shared_task_dev.jsonl" "$RAW/fever/shared_task_dev.jsonl"
fetch "$FEVER_BASE/shared_task_test.jsonl" "$RAW/fever/shared_task_test.jsonl"
fetch "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip" "$RAW/fever/fever_beir.zip"

mkdir -p "$RAW/scifact"
fetch "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz" "$RAW/scifact/scifact.tar.gz"
fetch "https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md"      "$RAW/scifact/LICENSE.md"

mkdir -p "$RAW/freshqa"
HF_BASE="https://huggingface.co/datasets/natyou/freshqa_10_06/resolve/main"
fetch "$HF_BASE/data/dev-00000-of-00001-1d5f59a9a40594d1.parquet" "$RAW/freshqa/dev.parquet"
fetch "$HF_BASE/data/test-00000-of-00001-7f8000441c84f75f.parquet" "$RAW/freshqa/test.parquet"

mkdir -p "$RAW/hotpotqa"
fetch "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet" "$RAW/hotpotqa/validation.parquet"

log "All downloads complete."
log "Now run:  python -m src.prepare  (builds normalized splits + corpora)"
