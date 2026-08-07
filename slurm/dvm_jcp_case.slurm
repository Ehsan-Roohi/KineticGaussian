#!/bin/bash
#SBATCH --job-name=dvm-jcp
#SBATCH --partition=gpu
#SBATCH --qos=long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=12-00:00:00
#SBATCH --mail-type=TIME_LIMIT_80,END,FAIL

set -euo pipefail

: "${KGFR_REPO_ROOT:?KGFR_REPO_ROOT is required}"
: "${DVM_TASK_FILE:?DVM_TASK_FILE is required}"
: "${DVM_PYTHON:?DVM_PYTHON is required}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"

cd "$KGFR_REPO_ROOT"
mkdir -p logs/dvm_jcp
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[start] $(date -Is)"
echo "[host] $(hostname)"
echo "[task] $SLURM_ARRAY_TASK_ID from $DVM_TASK_FILE"
"$DVM_PYTHON" -c 'import torch; print("[torch]", torch.__version__, "cuda=", torch.cuda.is_available(), "device=", torch.cuda.get_device_name(0))'
nvidia-smi || true

"$DVM_PYTHON" dvm/scripts/run_jcp_dvm_case.py \
  --task-file "$DVM_TASK_FILE" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --python "$DVM_PYTHON" \
  --repo-root "$KGFR_REPO_ROOT"

echo "[done] $(date -Is)"

