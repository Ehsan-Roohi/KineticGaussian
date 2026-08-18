#!/bin/bash
#SBATCH --job-name=kgfr_eval_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

readarray -t RUN_FIELDS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; c=json.load(open(sys.argv[1])); print(c["run_name"]); print(c.get("output_dir", "runs/conditional"))' \
    "$CONFIG"
)
RUN_NAME="${RUN_FIELDS[0]}"
RUN_ROOT="${RUN_FIELDS[1]}"
CHECKPOINT="$RUN_ROOT/$RUN_NAME/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 3
fi

hostname
nvidia-smi
"$PYTHON_BIN" evaluate_conditional_shocks.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases M2p5 M3 M5 \
  --output-subdir eval_all_cases \
  --x-stride 5 \
  --x-chunk 16 \
  --v-chunk 2048 \
  --sample-points 50000
