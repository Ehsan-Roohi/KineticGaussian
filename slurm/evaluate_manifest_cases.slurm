#!/bin/bash
#SBATCH --job-name=kgfr_eval_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
    'import json,sys; c=json.load(open(sys.argv[1])); print(c["run_name"]); print(c.get("output_dir", "runs/conditional")); print(c["manifest_path"])' \
    "$CONFIG"
)
RUN_NAME="${RUN_FIELDS[0]}"
RUN_ROOT="${RUN_FIELDS[1]}"
MANIFEST="${RUN_FIELDS[2]}"
CHECKPOINT="$RUN_ROOT/$RUN_NAME/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 3
fi

readarray -t CASES < <(
  "$PYTHON_BIN" -c \
    'import json,sys; print(*[str(c["name"]) for c in json.load(open(sys.argv[1]))["cases"]], sep="\n")' \
    "$MANIFEST"
)
if [[ "${#CASES[@]}" -lt 1 ]]; then
  echo "No cases found in manifest: $MANIFEST" >&2
  exit 4
fi

hostname
nvidia-smi
"$PYTHON_BIN" evaluate_conditional_shocks.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --cases "${CASES[@]}" \
  --output-subdir eval_manifest \
  --x-stride 5 \
  --x-chunk 16 \
  --v-chunk 2048 \
  --sample-points 50000
