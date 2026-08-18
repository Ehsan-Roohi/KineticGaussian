#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT/configs/dvm_jcp_generated/LAST_JCP_DVM_JOBS.env"
MANIFEST="$REPO_ROOT/configs/dvm_jcp_generated/suite_manifest.json"
if [[ ! -f "$ENV_FILE" || ! -f "$MANIFEST" ]]; then
  echo "No submitted JCP DVM campaign found. Run unity_submit_jcp_dvm.sh first." >&2
  exit 2
fi
source "$ENV_FILE"
JOB_IDS="$JCP_DVM_SMOKE,$JCP_DVM_CONVERGENCE_MID,$JCP_DVM_CONVERGENCE_HIGH,$JCP_DVM_GRID_GATE,$JCP_DVM_PRODUCTION_MID,$JCP_DVM_PRODUCTION_HIGH"

echo "===== ACTIVE / PENDING ====="
squeue -j "$JOB_IDS" -o "%.22i %.12P %.20j %.2t %.12M %.40R" || true
echo "===== ACCOUNTING ====="
sacct -X -j "$JOB_IDS" --format=JobID%24,JobName%20,State,ExitCode,Elapsed,NodeList || true
echo "===== PRODUCTS ====="
python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
for group in ("convergence", "production"):
    rows = manifest[group]
    moments = sum(Path(row["moments_path"]).is_file() for row in rows)
    fullstates = sum(Path(row["fullstate_path"]).is_file() for row in rows)
    print(f"{group:11s}: moments={moments}/{len(rows)}, fullstates={fullstates}/{len(rows)}")
PY
echo "Grid report: $JCP_DVM_DATA_ROOT/grid_convergence_report.json"
