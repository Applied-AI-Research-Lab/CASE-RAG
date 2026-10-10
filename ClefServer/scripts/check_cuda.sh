#!/usr/bin/env bash
set -u
fail=0

echo "=== driver ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  drv_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  echo "driver supports CUDA up to: ${drv_cuda:-unknown}"
else
  echo "nvidia-smi not found"; fail=1; drv_cuda=""
fi

echo
echo "=== torch ==="
python3 - "$drv_cuda" <<'PY'
import sys
drv = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    import torch
except Exception as e:
    print("torch does not import:", e); raise SystemExit(2)
print("torch         :", torch.__version__)
print("built for CUDA:", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if ok:
    print("device count  :", torch.cuda.device_count())
    raise SystemExit(0)

print()
print("TORCH CANNOT SEE THE GPUs.")
try:
    built = float(".".join((torch.version.cuda or "0").split(".")[:2]))
    have  = float(".".join((drv or "0").split(".")[:2]))
except ValueError:
    built = have = 0.0
if built and have and int(built) > int(have):
    print(f"  torch is built for CUDA {built}, the driver supports {have}.")
    print("  The pins in requirements-cuda.txt were bypassed (e.g. a plain")
    print("  `pip install torch`). Reinstall the pinned stack:")
    print("    source env.csh && uv pip install -r requirements-server.txt")
    print("  Note: CUDA 12.9 builds on a 12.8 driver are fine (same major")
    print("  version); a CUDA 13 build on a 12.x driver is not.")
else:
    print("  Not a version mismatch. Check CUDA_VISIBLE_DEVICES, and whether the")
    print("  GPUs are claimed by another job:  nvidia-smi")
raise SystemExit(2)
PY
[ $? -ne 0 ] && fail=1

echo
echo "=== vllm ==="
python3 - <<'PY'
try:
    import vllm
    print("vllm          :", vllm.__version__)
except Exception as e:
    print("vllm does not import:", type(e).__name__, e)
    print()
    print("If this appeared only after changing the torch build, vLLM's own wheel")
    print("was compiled for the other CUDA version. Options, in order:")
    print("  reinstall the pinned stack:  uv pip install -r requirements-server.txt")
    raise SystemExit(2)
PY
[ $? -ne 0 ] && fail=1

echo
if [ "$fail" -eq 0 ]; then
  echo "PREFLIGHT OK -- safe to download weights and start servers."
else
  echo "PREFLIGHT FAILED -- fix the above before scripts/tmux_up.sh."
  echo "Starting servers now will fail at worker init, after loading weights."
fi
exit "$fail"
