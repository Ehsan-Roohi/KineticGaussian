# Reproducibility guide

This repository separates lightweight software validation from the large Unity
campaign used for scientific results. Synthetic tests verify code paths; they
are not evidence for the physical accuracy reported in a paper.

## Local validation

Create an isolated Python 3.10 or newer environment, then run:

```bash
python -m pip install -e .
bash run_quick_smoke.sh
```

The command runs the unit-test suite and a three-step end-to-end training and
evaluation job on synthetic DVM data. It does not need private or large data.

## Scientific data

Large DVM arrays and trained checkpoints are intentionally excluded from Git.
Every scientific run must instead be traceable through a generated JSON
manifest containing the Mach number, absolute source path, array keys, source
size, train/holdout split, normalization, seed, hyperparameters, and output
directory. `data/manifest.example.json` documents the portable schema.

The current Unity data root is supplied explicitly to a launcher; it is not
embedded in the Python package:

```bash
bash scripts/unity_submit.sh /project/pi_roohie_umass_edu/BGK_shock
```

## Evidence levels

Use these labels when describing repository outputs:

1. **synthetic-smoke**: code-path validation only;
2. **Unity-complete**: Slurm job completed and required artifacts exist;
3. **numerically-certified**: spatial/velocity and temporal convergence gates passed;
4. **paper-release**: exact code, manifests, metrics, figures, environment, and
   archived data/model DOI are linked to a tagged release.

A completed Slurm job is not automatically a numerically certified result.

## Current release limitations

- The minimum dependency versions in `pyproject.toml` are install metadata, not
  an exact lock of the Unity environment.
- The high-Mach JCP workflow checks velocity/spatial medium-to-fine agreement,
  but a strict temporal residual gate is still required before paper release.
- The exploratory full-Mach launcher currently uses per-case coordinate
  normalization and skips matched-storage baselines. A publication-grade
  rerun should use training-only shared normalization and include a like-for-like
  baseline.
- Public archival copies of the scientific manifests, final checkpoints,
  metrics, and permitted data are still pending.

See `docs/RELEASE_CHECKLIST.md` before tagging a release.
