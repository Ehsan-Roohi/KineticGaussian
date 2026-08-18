#!/bin/bash
#SBATCH -J dvm_M8
#SBATCH -p gpu
#SBATCH --qos=long
#SBATCH --gpus=1
#SBATCH --constraint=fp64&vram48
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12-00:00:00
#SBATCH --mail-type=TIME_LIMIT_80,END,FAIL
#SBATCH -o /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.out
#SBATCH -e /project/pi_roohie_umass_edu/BGK_shock/logs/%x_%j.err

set -euo pipefail

cd /project/pi_roohie_umass_edu/BGK_shock
mkdir -p logs "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8" "/project/pi_roohie_umass_edu/BGK_shock/figures/mach_sweep_extra/M8"

PY=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python

if [ ! -x "$PY" ]; then
  echo "[ERROR] Python env not found: $PY"
  echo "Trying mamba activation..."
  source ~/.bashrc
  mamba activate dsmc-gpu
  PY=$(which python)
fi

echo "[PY] $PY"
"$PY" - <<'PYCHECK'
import sys
print("[python]", sys.executable)
import torch
print("[torch]", torch.__version__, "cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[cuda device]", torch.cuda.get_device_name(0))
PYCHECK

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[START] $(date)"
echo "[HOST] $(hostname)"
echo "[CUDA_VISIBLE_DEVICES] ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi || true
scontrol show node $SLURMD_NODENAME | egrep 'NodeName|AvailableFeatures|ActiveFeatures|Gres|CfgTRES' || true

echo "[CASE] tag=M8, M1=8.0, xhalf=120, nx=2800, v=171x31x31, vmax=32, steps=160000"
echo "[OUT] /project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8/standing_M8_hmom_x120_nx2800_v171_31_31_vmax32_fullstate.npz"
echo "[FIG] /project/pi_roohie_umass_edu/BGK_shock/figures/mach_sweep_extra/M8/standing_M8_profiles.png"

"$PY" "src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py" \
  --out "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8/standing_M8_hmom_x120_nx2800_v171_31_31_vmax32_fullstate.npz" \
  --fig "/project/pi_roohie_umass_edu/BGK_shock/figures/mach_sweep_extra/M8/standing_M8_profiles.png" \
  --M1 "8.0" \
  --gamma 1.6666666666666667 \
  --rho1 1.0 \
  --T1 1.0 \
  --xhalf-mfp "120" \
  --nx "2800" \
  --nvx "171" \
  --nvy "31" \
  --nvz "31" \
  --vmax "32" \
  --steps "160000" \
  --cfl 0.75 \
  --center-every 50 \
  --save-every 1500 \
  --corr-iters 4 \
  --device cuda \
  --dtype float64

echo "[DONE] $(date)"
find "/project/pi_roohie_umass_edu/BGK_shock/ref/mach_sweep_extra/M8" -maxdepth 1 -type f -ls
