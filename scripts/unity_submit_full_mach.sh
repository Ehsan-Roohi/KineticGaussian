#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/unity_submit_full_mach.sh /project/pi_roohie_umass_edu/BGK_shock" >&2
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
SUITE_DIR="$REPO_ROOT/configs/generalization/full_mach_generated"
TASK_FILE="$SUITE_DIR/tasks.txt"
MANIFEST="$REPO_ROOT/data/unity_full_mach_manifest.json"
mkdir -p data "$SUITE_DIR" logs/conditional

"$PYTHON_BIN" scripts/make_full_mach_manifest.py \
  --bgk-root "$BGK_ROOT" \
  --output "$MANIFEST" \
  --tags M1p5,M2,M2p5,M3,M4,M5,M6,M8,M12

"$PYTHON_BIN" scripts/prepare_generalization_suite.py \
  --manifest "$MANIFEST" \
  --output-dir "$SUITE_DIR" \
  --holdouts M3,M6,M12 \
  --capacities 512 \
  --mach-degrees 2,3 \
  --seeds 1234,2026,3407 \
  --objectives moment \
  --conditionings all \
  --mach-bounds-source training \
  --coordinate-normalization shared_training \
  --run-prefix fullmach \
  --skip-baselines \
  --steps 160000

"$PYTHON_BIN" - "$MANIFEST" "$TASK_FILE" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
task_file = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
cases = manifest["cases"]
names = [case["name"] for case in cases]
expected_names = ["M1p5", "M2", "M2p5", "M3", "M4", "M5", "M6", "M8", "M12"]
if names != expected_names:
    raise SystemExit(f"Expected authoritative cases {expected_names}, found {names}")
if any("lite" in name.lower() for name in names):
    raise SystemExit(f"Lite case leaked into the primary manifest: {names}")
for case in cases:
    if case.get("moment_path") is None:
        raise SystemExit(f"Missing high-moment companion for {case['name']}")

paths = [Path(line.strip()) for line in task_file.read_text().splitlines() if line.strip()]
if len(paths) != 18:
    raise SystemExit(f"Expected 18 full-Mach tasks, found {len(paths)}")

configs = [json.loads(path.read_text()) for path in paths]
run_names = [cfg["run_name"] for cfg in configs]
if len(run_names) != len(set(run_names)):
    raise SystemExit("Generated run names are not unique")

counts = {}
for cfg in configs:
    holdouts = cfg["holdout_cases"]
    if len(holdouts) != 1:
        raise SystemExit(f"Expected one holdout in {cfg['run_name']}: {holdouts}")
    holdout = holdouts[0]
    degree = int(cfg["model"]["mach_degree"])
    key = (holdout, degree)
    counts[key] = counts.get(key, 0) + 1
    if holdout in cfg["train_cases"]:
        raise SystemExit(f"Holdout leakage in {cfg['run_name']}")
    if cfg.get("mach_bounds_source") != "training":
        raise SystemExit(f"Non-blind Mach normalization in {cfg['run_name']}")
    if cfg.get("data", {}).get("coordinate_normalization") != "shared_training":
        raise SystemExit(f"Non-shared coordinate normalization in {cfg['run_name']}")
    if holdout == "M12":
        train_machs = [float(next(c["mach"] for c in cases if c["name"] == name)) for name in cfg["train_cases"]]
        if max(train_machs) > 8.0 or cfg["mach_bounds"] != [1.5, 8.0]:
            raise SystemExit(f"M12 extrapolation leakage in {cfg['run_name']}")
    run_dir = Path(cfg.get("output_dir", "runs/conditional")) / cfg["run_name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite existing run data in {run_dir}; move it before resubmitting"
        )

expected_counts = {(holdout, degree): 3 for holdout in ("M3", "M6", "M12") for degree in (2, 3)}
if counts != expected_counts:
    raise SystemExit(f"Unexpected full-Mach matrix: {counts}")
print(
    "[kinetic-gaussian] validated 18-run full-Mach matrix with blind M12 "
    "extrapolation and training-only shared coordinate normalization"
)
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
  slurm/evaluate_manifest_cases.slurm)"

echo "Submitted full-Mach smoke test: $SMOKE_JOB"
echo "Submitted full-Mach training:   $TRAIN_JOB ($TASK_COUNT tasks)"
echo "Submitted nine-case evaluation: $EVAL_JOB ($TASK_COUNT tasks)"
echo "Monitor: squeue -j $SMOKE_JOB,$TRAIN_JOB,$EVAL_JOB"
echo "After completion: $PYTHON_BIN scripts/summarize_full_mach.py"
