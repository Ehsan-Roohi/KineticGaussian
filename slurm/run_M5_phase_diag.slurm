#!/bin/bash
#SBATCH --job-name=M5_phase_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=72G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# Adjust these lines for your cluster environment.
# module load cuda/12.1
# source .venv/bin/activate

python train_phase_gaussian.py --config configs/M5_phase_diag.json
python evaluate_phase_gaussian.py \
  --config configs/M5_phase_diag.json \
  --checkpoint runs/M5_phase_diag_N320_moment/best.pt \
  --x-stride 5
