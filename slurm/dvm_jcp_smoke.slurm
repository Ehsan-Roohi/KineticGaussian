#!/bin/bash
#SBATCH --job-name=dvm-jcp-smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${KGFR_REPO_ROOT:?KGFR_REPO_ROOT is required}"
: "${DVM_DATA_ROOT:?DVM_DATA_ROOT is required}"
: "${DVM_PYTHON:?DVM_PYTHON is required}"

cd "$KGFR_REPO_ROOT"
SMOKE_DIR="$DVM_DATA_ROOT/smoke"
mkdir -p "$SMOKE_DIR"
"$DVM_PYTHON" dvm/src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py \
  --out "$SMOKE_DIR/standing_M6_smoke.npz" \
  --fig "$SMOKE_DIR/standing_M6_smoke.png" \
  --M1 6 --xhalf-mfp 5 --nx 16 --steps 2 \
  --grid-mode composite --grid-gauss-order 2 \
  --grid-core-sigma 3 --grid-tail-sigma 6 --grid-interval-sigma 1 \
  --corr-iters 2 --center-every 2 --save-every 2 --x-chunk 4 \
  --device cuda --dtype float64

"$DVM_PYTHON" - "$SMOKE_DIR/standing_M6_smoke.npz" "$SMOKE_DIR/standing_M6_smoke_fullstate.npz" <<'PY'
import sys
import numpy as np
for path in sys.argv[1:]:
    with np.load(path, allow_pickle=True) as data:
        required = {"x", "rho", "ux", "T", "qx", "sig"}
        if not required <= set(data.files):
            raise SystemExit(f"Smoke output {path} missing {required - set(data.files)}")
        if not all(np.all(np.isfinite(data[key])) for key in required):
            raise SystemExit(f"Non-finite values in smoke output {path}")
print("[jcp-dvm] GPU smoke PASS")
PY

