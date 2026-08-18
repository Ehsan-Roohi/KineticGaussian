#!/bin/bash
#SBATCH -J dvm_M6
#SBATCH -p gpu
#SBATCH --qos=long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=9-00:00:00
#SBATCH --mail-type=TIME_LIMIT_80,END,FAIL
#SBATCH -o /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.out
#SBATCH -e /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.err

set -euo pipefail

cd /project/pi_roohie_umass_edu/BGK_shock
mkdir -p logs "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M6"

source ~/.bashrc >/dev/null 2>&1 || true
mamba activate dsmc-gpu >/dev/null 2>&1 || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export DVM_DEVICE=cuda

echo "[START] $(date)"
echo "[HOST] $(hostname)"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true
echo "[CASE] M6, M=6.0, xhalf=95, nx=2400, v=151x29x29, vmax=26, steps=120000"
echo "[OUTDIR] /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M6"

python scripts/run_densemicro_case_adapt.py \
  --driver "src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py" \
  --tag "M6" \
  --mach "6.0" \
  --xhalf "95" \
  --nx "2400" \
  --nvx "151" \
  --nvy "29" \
  --nvz "29" \
  --vmax "26" \
  --steps "120000" \
  --outdir "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M6"

echo "[DONE] $(date)"
