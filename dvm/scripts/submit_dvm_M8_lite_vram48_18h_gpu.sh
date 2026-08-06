#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram48
#SBATCH -t 18:00:00
#SBATCH --mem=160G
#SBATCH -J dvm_M8lite48
#SBATCH --cpus-per-task=16
#SBATCH -o /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.out
#SBATCH -e /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"

echo "[NOTE] M8 lite before maintenance: M=8, xhalf=120, nx=2400, v=151x29x29, vmax=32, steps=100000"


set -euo pipefail

cd /project/pi_roohie_umass_edu/BGK_shock
mkdir -p logs "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8lite"

source ~/.bashrc >/dev/null 2>&1 || true
mamba activate dsmc-gpu >/dev/null 2>&1 || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg

echo "[START] $(date)"
echo "[HOST] $(hostname)"
echo "[CASE] M8, M=8.0, xhalf=120, nx=2800, v=171x31x31, vmax=32, steps=160000"
echo "[OUTDIR] /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8lite"

"/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python" scripts/run_densemicro_case_adapt.py \
  --driver "src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py" \
  --tag "M8lite" \
  --mach "8.0" \
  --xhalf "120" \
  --nx "2400" \
  --nvx "151" \
  --nvy "29" \
  --nvz "29" \
  --vmax "32" \
  --steps "100000" \
  --outdir "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8lite"

echo "[DONE] $(date)"
