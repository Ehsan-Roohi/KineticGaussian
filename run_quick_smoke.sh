#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${KGFR_PYTHON:-python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/kinetic-gaussian-matplotlib}"

export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m unittest discover -v
"$PYTHON_BIN" tests/smoke_end_to_end.py
