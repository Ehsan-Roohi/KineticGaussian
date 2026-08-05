#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/unity_submit.sh /project/pi_roohie_umass_edu/BGK_shock" >&2
  exit 2
fi

BGK_ROOT="$(cd "$1" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PY=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dde-tf/bin/python
PYTHON_BIN="${KGFR_PYTHON:-$DEFAULT_PY}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

cd "$REPO_ROOT"
mkdir -p data configs/generalization/generated logs/conditional
"$PYTHON_BIN" scripts/make_unity_manifest.py \
  --bgk-root "$BGK_ROOT" \
  --output data/unity_manifest.json
"$PYTHON_BIN" scripts/prepare_generalization_suite.py \
  --manifest data/unity_manifest.json \
  --output-dir configs/generalization/generated \
  --holdouts M3 \
  --capacities 256,512,1024 \
  --seeds 1234,2026,3407 \
  --objectives logf,moment \
  --steps 80000

TASK_FILE="$REPO_ROOT/configs/generalization/generated/tasks.txt"
BASELINE_TASK_FILE="$REPO_ROOT/configs/generalization/generated/baseline_tasks.jsonl"
TASK_COUNT="$(awk 'NF{n++} END{print n+0}' "$TASK_FILE")"
BASELINE_COUNT="$(awk 'NF{n++} END{print n+0}' "$BASELINE_TASK_FILE")"
if [[ "$TASK_COUNT" -lt 1 ]]; then
  echo "No GPU tasks generated" >&2
  exit 3
fi

GPU_JOB="$(sbatch --parsable \
  --array="0-$((TASK_COUNT - 1))%4" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_TASK_FILE=$TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/conditional_array.slurm)"
echo "Submitted conditional suite: job $GPU_JOB ($TASK_COUNT tasks)"

if [[ "$BASELINE_COUNT" -gt 0 ]]; then
  BASELINE_JOB="$(sbatch --parsable \
    --array="0-$((BASELINE_COUNT - 1))%2" \
    --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_BASELINE_TASK_FILE=$BASELINE_TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
    slurm/baseline_array.slurm)"
  echo "Submitted matched-storage baselines: job $BASELINE_JOB ($BASELINE_COUNT tasks)"
fi

echo "Monitor: squeue -u \"$USER\""
echo "Logs:    tail -f \"$REPO_ROOT\"/logs/conditional/*.out"
