#!/bin/bash
#SBATCH --job-name=kgfr_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/conditional/%x_%A_%a.out
#SBATCH --error=logs/conditional/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${KGFR_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
TASK_FILE="${KGFR_BASELINE_TASK_FILE:?KGFR_BASELINE_TASK_FILE is required}"
PYTHON_BIN="${KGFR_PYTHON:-python}"
cd "$REPO_ROOT"
mkdir -p logs/conditional
"$PYTHON_BIN" scripts/run_baseline_task.py --task-file "$TASK_FILE" --index "$SLURM_ARRAY_TASK_ID"
