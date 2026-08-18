#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/unity_submit_v2.sh /project/pi_roohie_umass_edu/BGK_shock" >&2
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
mkdir -p data configs/generalization/v2_generated logs/conditional
"$PYTHON_BIN" scripts/make_unity_manifest.py \
  --bgk-root "$BGK_ROOT" \
  --output data/unity_manifest.json
"$PYTHON_BIN" scripts/prepare_generalization_suite.py \
  --manifest data/unity_manifest.json \
  --output-dir configs/generalization/v2_generated \
  --holdouts M3 \
  --capacities 128,256,512 \
  --seeds 1234,2026,3407 \
  --objectives moment \
  --conditionings amplitude \
  --coordinate-normalization shared_training \
  --skip-baselines \
  --steps 80000

TASK_FILE="$REPO_ROOT/configs/generalization/v2_generated/tasks.txt"
TASK_COUNT="$(awk 'NF{n++} END{print n+0}' "$TASK_FILE")"
if [[ "$TASK_COUNT" -ne 9 ]]; then
  echo "Expected exactly 9 v2 GPU tasks, found $TASK_COUNT" >&2
  exit 3
fi

SMOKE_JOB="$(sbatch --parsable \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/smoke.slurm)"
echo "Submitted v2 preflight smoke test: $SMOKE_JOB"

GPU_JOB="$(sbatch --parsable \
  --dependency="afterok:$SMOKE_JOB" \
  --array="0-$((TASK_COUNT - 1))%4" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_TASK_FILE=$TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/conditional_array.slurm)"
echo "Submitted v2 held-out-M3 suite: $GPU_JOB ($TASK_COUNT tasks)"
echo "Monitor: squeue -j $SMOKE_JOB,$GPU_JOB"
echo "Logs: logs/conditional/kinetic_cond_${GPU_JOB}_*.out"
