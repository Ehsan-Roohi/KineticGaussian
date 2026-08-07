#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash dvm/scripts/unity_submit_jcp_dvm.sh /project/pi_roohie_umass_edu/BGK_shock" >&2
  exit 2
fi

BGK_ROOT="$(cd "$1" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PY=/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python
PYTHON_BIN="${DVM_PYTHON:-$DEFAULT_PY}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "DVM Python environment not found: $PYTHON_BIN" >&2
  exit 2
fi

cd "$REPO_ROOT"
GEN_DIR="$REPO_ROOT/configs/dvm_jcp_generated"
DATA_ROOT="$BGK_ROOT/ref/jcp_velocity_certified"
AUDIT_DIR="$DATA_ROOT/audit"
REPORT="$DATA_ROOT/grid_convergence_report.json"
mkdir -p "$GEN_DIR" "$AUDIT_DIR" "$REPO_ROOT/logs/dvm_jcp"

# Cheap deterministic gate: no long GPU job is submitted unless the selected
# quadrature resolves equilibrium moments at every requested Mach number.
"$PYTHON_BIN" dvm/scripts/audit_velocity_grid.py \
  --machs 6,7,8,10,12 \
  --require-level medium \
  --output-dir "$AUDIT_DIR"

"$PYTHON_BIN" dvm/scripts/prepare_jcp_dvm_suite.py \
  --config dvm/configs/jcp_high_mach_cases.json \
  --output-dir "$GEN_DIR" \
  --data-root "$DATA_ROOT"

submit_array() {
  local task_file="$1"
  local concurrency="$2"
  local constraint="$3"
  local dependency="${4:-}"
  local job_name="${5:-dvm-jcp}"
  local memory="${6:-220G}"
  local count
  count="$(awk 'NF{n++} END{print n+0}' "$task_file")"
  if [[ "$count" -eq 0 ]]; then
    echo ""
    return
  fi
  local dep_args=()
  if [[ -n "$dependency" ]]; then
    dep_args+=(--dependency="$dependency")
  fi
  sbatch --parsable \
    "${dep_args[@]}" \
    --array="0-$((count - 1))%$concurrency" \
    --job-name="$job_name" \
    --constraint="$constraint" \
    --mem="$memory" \
    --output="$REPO_ROOT/logs/dvm_jcp/%x_%A_%a.out" \
    --error="$REPO_ROOT/logs/dvm_jcp/%x_%A_%a.err" \
    --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,DVM_TASK_FILE=$task_file,DVM_PYTHON=$PYTHON_BIN" \
    slurm/dvm_jcp_case.slurm
}

SMOKE_JOB="$(sbatch --parsable \
  --constraint='fp64&vram48' \
  --output="$REPO_ROOT/logs/dvm_jcp/%x_%j.out" \
  --error="$REPO_ROOT/logs/dvm_jcp/%x_%j.err" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,DVM_DATA_ROOT=$DATA_ROOT,DVM_PYTHON=$PYTHON_BIN" \
  slurm/dvm_jcp_smoke.slurm)"

CONV_MID="$(submit_array "$GEN_DIR/convergence_mid.jsonl" 1 'fp64&vram48' "afterok:$SMOKE_JOB" 'dvm-m6-grid' '160G')"
CONV_HIGH="$(submit_array "$GEN_DIR/convergence_high.jsonl" 1 'fp64&vram80' "afterok:$SMOKE_JOB" 'dvm-m12-grid' '240G')"
GATE_DEP="afterok:$CONV_MID:$CONV_HIGH"
GATE_JOB="$(sbatch --parsable \
  --dependency="$GATE_DEP" \
  --output="$REPO_ROOT/logs/dvm_jcp/%x_%j.out" \
  --error="$REPO_ROOT/logs/dvm_jcp/%x_%j.err" \
  --export="ALL,KGFR_REPO_ROOT=$REPO_ROOT,DVM_SUITE_MANIFEST=$GEN_DIR/suite_manifest.json,DVM_GRID_REPORT=$REPORT,DVM_PYTHON=$PYTHON_BIN" \
  slurm/dvm_jcp_grid_gate.slurm)"

PROD_MID="$(submit_array "$GEN_DIR/production_mid.jsonl" 1 'fp64&vram48' "afterok:$GATE_JOB" 'dvm-m7m8' '160G')"
PROD_HIGH="$(submit_array "$GEN_DIR/production_high.jsonl" 1 'fp64&vram80' "afterok:$GATE_JOB" 'dvm-m10' '240G')"

cat > "$GEN_DIR/LAST_JCP_DVM_JOBS.env" <<EOF
JCP_DVM_SMOKE=$SMOKE_JOB
JCP_DVM_CONVERGENCE_MID=$CONV_MID
JCP_DVM_CONVERGENCE_HIGH=$CONV_HIGH
JCP_DVM_GRID_GATE=$GATE_JOB
JCP_DVM_PRODUCTION_MID=$PROD_MID
JCP_DVM_PRODUCTION_HIGH=$PROD_HIGH
JCP_DVM_DATA_ROOT=$DATA_ROOT
EOF

echo "Submitted GPU smoke test:                       $SMOKE_JOB"
echo "Submitted M6 convergence (48-GB FP64 GPU):  $CONV_MID"
echo "Submitted M12 convergence (80-GB FP64 GPU): $CONV_HIGH"
echo "Submitted convergence gate:                 $GATE_JOB"
echo "Submitted M7/M8 production (48-GB GPU):     $PROD_MID"
echo "Submitted M10 production (80-GB GPU):       $PROD_HIGH"
echo "Jobs file: $GEN_DIR/LAST_JCP_DVM_JOBS.env"
echo "Monitor: squeue -j $SMOKE_JOB,$CONV_MID,$CONV_HIGH,$GATE_JOB,$PROD_MID,$PROD_HIGH"
