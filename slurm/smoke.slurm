#!/bin/bash
#SBATCH --job-name=kinetic_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/conditional/%x_%j.out
#SBATCH --error=logs/conditional/%x_%j.err

set -euo pipefail

REPO_ROOT="${KGFR_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
PYTHON_BIN="${KGFR_PYTHON:-python}"
cd "$REPO_ROOT"
mkdir -p logs/conditional

"$PYTHON_BIN" -m unittest tests/test_conditional_model.py
"$PYTHON_BIN" tests/smoke_end_to_end.py
echo "KINETIC_GAUSSIAN_PREFLIGHT_OK"
