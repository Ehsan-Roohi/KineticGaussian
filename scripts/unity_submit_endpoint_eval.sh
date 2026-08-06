#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dde-tf/bin/python
PYTHON_BIN="${KGFR_PYTHON:-$DEFAULT_PY}"
TASK_FILE="$REPO_ROOT/configs/generalization/generated/tasks.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ ! -f "$TASK_FILE" ]]; then
  echo "Missing $TASK_FILE; run the original Unity launcher first." >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p logs/conditional
TASK_COUNT="$(awk 'NF{n++} END{print n+0}' "$TASK_FILE")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No endpoint-evaluation tasks found" >&2
  exit 3
fi

JOB_ID="$(sbatch --parsable \
  --array="0-$((TASK_COUNT - 1))%4" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_TASK_FILE=$TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/evaluate_all_cases.slurm)"
echo "Submitted all-case checkpoint evaluation: $JOB_ID ($TASK_COUNT tasks)"
echo "Monitor: squeue -j $JOB_ID"
echo "Results: runs/conditional/*/eval_all_cases/metrics.json"
