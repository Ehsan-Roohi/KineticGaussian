#!/bin/bash
#SBATCH -J dvm_M2p5
#SBATCH -p gpu
#SBATCH --qos=long
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=TIME_LIMIT_80,END,FAIL
#SBATCH -o /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.out
#SBATCH -e /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.err

set -euo pipefail

cd /project/pi_roohie_umass_edu/BGK_shock
mkdir -p logs "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M2p5"

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
echo "[CASE] M2p5, M=2.5, xhalf=50, nx=1600, v=111x21x21, vmax=14, steps=70000"
echo "[OUTDIR] /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M2p5"

python scripts/run_densemicro_case_adapt.py \
  --driver "src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py" \
  --tag "M2p5" \
  --mach "2.5" \
  --xhalf "50" \
  --nx "1600" \
  --nvx "111" \
  --nvy "21" \
  --nvz "21" \
  --vmax "14" \
  --steps "70000" \
  --outdir "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M2p5"

echo "[DONE] $(date)"
