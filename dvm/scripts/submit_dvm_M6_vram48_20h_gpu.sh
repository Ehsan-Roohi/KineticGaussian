#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48
#SBATCH -t 20:00:00
#SBATCH --mem=160G
#SBATCH -J dvm_M6_48
#SBATCH --cpus-per-task=16
#SBATCH -o /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.out
#SBATCH -e /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[PYTHON] /work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python"
"/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"


set -euo pipefail

cd /project/pi_roohie_umass_edu/BGK_shock
mkdir -p logs "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M6"

source ~/.bashrc >/dev/null 2>&1 || true
mamba activate dsmc-gpu >/dev/null 2>&1 || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg

echo "[START] $(date)"
echo "[HOST] $(hostname)"
echo "[CASE] M6, M=6.0, xhalf=95, nx=2400, v=151x29x29, vmax=26, steps=120000"
echo "[OUTDIR] /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M6"

"/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python" scripts/run_densemicro_case_adapt.py \
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
