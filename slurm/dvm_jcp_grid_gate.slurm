#!/bin/bash
#SBATCH --job-name=dvm-grid-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
: "${KGFR_REPO_ROOT:?KGFR_REPO_ROOT is required}"
: "${DVM_SUITE_MANIFEST:?DVM_SUITE_MANIFEST is required}"
: "${DVM_GRID_REPORT:?DVM_GRID_REPORT is required}"
: "${DVM_PYTHON:?DVM_PYTHON is required}"

cd "$KGFR_REPO_ROOT"
"$DVM_PYTHON" dvm/scripts/check_jcp_grid_convergence.py \
  --suite-manifest "$DVM_SUITE_MANIFEST" \
  --output "$DVM_GRID_REPORT"

