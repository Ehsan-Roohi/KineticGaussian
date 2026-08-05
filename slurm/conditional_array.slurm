#!/bin/bash
#SBATCH --job-name=kinetic_cond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/conditional/%x_%A_%a.out
#SBATCH --error=logs/conditional/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${KGFR_REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
TASK_FILE="${KGFR_TASK_FILE:?KGFR_TASK_FILE is required}"
PYTHON_BIN="${KGFR_PYTHON:-python}"
cd "$REPO_ROOT"
mkdir -p logs/conditional

CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASK_FILE")"
if [[ -z "$CONFIG" ]]; then
  echo "No config for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

hostname
nvidia-smi
"$PYTHON_BIN" train_conditional_shocks.py --config "$CONFIG"
RUN_NAME="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_name"])' "$CONFIG")"
"$PYTHON_BIN" evaluate_conditional_shocks.py \
  --config "$CONFIG" \
  --checkpoint "runs/conditional/${RUN_NAME}/best.pt" \
  --x-stride 5 \
  --x-chunk 16 \
  --v-chunk 2048 \
  --sample-points 50000
