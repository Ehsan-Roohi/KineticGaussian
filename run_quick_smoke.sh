#!/bin/bash
set -euo pipefail

python -m kgfr.inspect_npz --path data/pack_M3_M5/M3_DVM_hmom.npz
python train_moment_gaussian_1d.py --config configs/M3_moment1d.json
