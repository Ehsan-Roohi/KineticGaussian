#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/unity_submit_ablation.sh /project/pi_roohie_umass_edu/BGK_shock" >&2
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
SUITE_ROOT="$REPO_ROOT/configs/generalization/ablation_generated"
ALL_SHARED_DIR="$SUITE_ROOT/all_shared"
AMP_PERCASE_DIR="$SUITE_ROOT/amp_percase"
TASK_FILE="$SUITE_ROOT/tasks.txt"
mkdir -p data "$ALL_SHARED_DIR" "$AMP_PERCASE_DIR" logs/conditional

"$PYTHON_BIN" scripts/make_unity_manifest.py \
  --bgk-root "$BGK_ROOT" \
  --output data/unity_manifest.json

# Missing cell 1: all Gaussian parameters vary with Mach, with one training-only
# coordinate normalization. N=256 has 5,120 learned parameters.
"$PYTHON_BIN" scripts/prepare_generalization_suite.py \
  --manifest data/unity_manifest.json \
  --output-dir "$ALL_SHARED_DIR" \
  --holdouts M3 \
  --capacities 256 \
  --seeds 1234,2026,3407 \
  --objectives moment \
  --conditionings all \
  --coordinate-normalization shared_training \
  --run-prefix ablation_all_shared \
  --skip-baselines \
  --steps 80000

# Missing cell 2: only amplitudes vary with Mach, with per-case coordinate
# normalization. N=512 has 5,632 learned parameters, close to the 5,120 above.
"$PYTHON_BIN" scripts/prepare_generalization_suite.py \
  --manifest data/unity_manifest.json \
  --output-dir "$AMP_PERCASE_DIR" \
  --holdouts M3 \
  --capacities 512 \
  --seeds 1234,2026,3407 \
  --objectives moment \
  --conditionings amplitude \
  --coordinate-normalization per_case \
  --run-prefix ablation_amp_percase \
  --skip-baselines \
  --steps 80000

awk 'NF' "$ALL_SHARED_DIR/tasks.txt" "$AMP_PERCASE_DIR/tasks.txt" > "$TASK_FILE"

"$PYTHON_BIN" - "$TASK_FILE" <<'PY'
import json
import sys
from pathlib import Path

task_file = Path(sys.argv[1])
paths = [Path(line.strip()) for line in task_file.read_text().splitlines() if line.strip()]
if len(paths) != 6:
    raise SystemExit(f"Expected exactly 6 ablation tasks, found {len(paths)}")

configs = [json.loads(path.read_text()) for path in paths]
run_names = [cfg["run_name"] for cfg in configs]
if len(set(run_names)) != 6:
    raise SystemExit(f"Ablation run names are not unique: {run_names}")

expected = {
    "ablation_all_shared": ("all", "shared_training", 256),
    "ablation_amp_percase": ("amplitude", "per_case", 512),
}
counts = {name: 0 for name in expected}
for cfg in configs:
    prefix = next((name for name in expected if cfg["run_name"].startswith(name + "_")), None)
    if prefix is None:
        raise SystemExit(f"Unexpected run name: {cfg['run_name']}")
    conditioning, normalization, kernels = expected[prefix]
    actual = (
        cfg["model"]["conditioning"],
        cfg["data"]["coordinate_normalization"],
        int(cfg["model"]["num_kernels"]),
    )
    if actual != (conditioning, normalization, kernels):
        raise SystemExit(f"Invalid ablation config {cfg['run_name']}: {actual}")
    counts[prefix] += 1
    run_dir = Path(cfg.get("output_dir", "runs/conditional")) / cfg["run_name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite existing run data in {run_dir}. "
            "Inspect or move that run directory before resubmitting."
        )

if counts != {"ablation_all_shared": 3, "ablation_amp_percase": 3}:
    raise SystemExit(f"Unexpected ablation matrix: {counts}")
print("[kinetic-gaussian] validated collision-safe 6-run ablation matrix")
PY

TASK_COUNT="$(awk 'NF{n++} END{print n+0}' "$TASK_FILE")"
SMOKE_JOB="$(sbatch --parsable \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/smoke.slurm)"

TRAIN_JOB="$(sbatch --parsable \
  --dependency="afterok:$SMOKE_JOB" \
  --array="0-$((TASK_COUNT - 1))%4" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_TASK_FILE=$TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/conditional_array.slurm)"

EVAL_JOB="$(sbatch --parsable \
  --dependency="afterok:$TRAIN_JOB" \
  --array="0-$((TASK_COUNT - 1))%4" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,KGFR_TASK_FILE=$TASK_FILE,KGFR_PYTHON=$PYTHON_BIN" \
  slurm/evaluate_all_cases.slurm)"

echo "Submitted ablation smoke test:  $SMOKE_JOB"
echo "Submitted ablation training:    $TRAIN_JOB ($TASK_COUNT tasks)"
echo "Submitted all-case evaluation: $EVAL_JOB ($TASK_COUNT tasks)"
echo "Monitor: squeue -j $SMOKE_JOB,$TRAIN_JOB,$EVAL_JOB"
echo "After completion: $PYTHON_BIN scripts/summarize_ablation.py"
