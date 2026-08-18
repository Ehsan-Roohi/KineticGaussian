#!/bin/bash
#SBATCH --job-name=M3_phase_xvx
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Adjust these lines for your cluster environment.
# module load cuda/12.1
# source .venv/bin/activate

python train_phase_gaussian.py --config configs/M3_phase_xvx.json
python evaluate_phase_gaussian.py \
  --config configs/M3_phase_xvx.json \
  --checkpoint runs/M3_phase_xvx_N256_moment/best.pt \
  --x-stride 5
