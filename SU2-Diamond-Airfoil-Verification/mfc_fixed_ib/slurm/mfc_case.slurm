#!/usr/bin/env bash
set -euo pipefail

: "${CASE_DIR:?CASE_DIR is required}"
: "${MFC_ROOT:?MFC_ROOT is required}"
: "${GRID:?GRID is required}"
: "${STEPS:?STEPS is required}"
: "${SAVE_EVERY:?SAVE_EVERY is required}"
: "${ALPHA:=40}"

# MFC was built against Unity's OpenMPI 5.0.3.  Force the same module and
# launcher used by the previously successful run; MFC's default launcher
# preference selects srun inside a Slurm allocation, which fails during
# MPI_Init on Unity for this build.
module purge
module load openmpi/5.0.3
MPI_LAUNCHER=$(command -v mpirun)
echo "MPI_LAUNCHER=$MPI_LAUNCHER"
"$MPI_LAUNCHER" --version

export OMP_NUM_THREADS=1
python3 "$CASE_DIR/validate_mfc_2d_stl.py" \
    "$CASE_DIR/Diamond_Airfoil_2D_MFC.stl" --expected-edges 4

LOG="$CASE_DIR/mfc-${GRID}.log"
cd "$MFC_ROOT"
set +e
"$MFC_ROOT/mfc.sh" run "$CASE_DIR/case.py" \
    -n "$SLURM_NTASKS" -b mpirun -- \
    --alpha "$ALPHA" \
    --mode euler \
    --grid "$GRID" \
    --steps "$STEPS" \
    --save-every "$SAVE_EVERY" 2>&1 | tee "$LOG"
run_status=${PIPESTATUS[0]}
set -e

if (( run_status != 0 )); then
    echo "MFC failed with exit code $run_status" >&2
    exit "$run_status"
fi

if ! grep -Eq 'Number of 2D model boundary edges:[[:space:]]+4[[:space:]]*$' "$LOG"; then
    echo "Immersed-boundary gate failed: MFC did not report exactly four boundary edges." >&2
    exit 40
fi

FINAL_SILO="$CASE_DIR/silo_hdf5/root/collection_${STEPS}.silo"
if [[ ! -s "$FINAL_SILO" ]]; then
    echo "Missing or empty final Silo collection: $FINAL_SILO" >&2
    exit 41
fi

printf 'status=PASS\ngrid=%s\nsteps=%s\nfinal_silo=%s\n' \
    "$GRID" "$STEPS" "$FINAL_SILO" | tee "$CASE_DIR/RUN_OK.txt"
