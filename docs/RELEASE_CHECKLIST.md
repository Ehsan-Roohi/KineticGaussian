# Scientific release checklist

## Software

- [x] Python sources compile.
- [x] Lightweight DVM/manifest tests pass.
- [x] Synthetic end-to-end smoke test is defined.
- [x] Pull-request CI is defined.
- [x] CI passes in a clean CPU environment (GitHub Actions run 32128850259).
- [ ] Exact Unity environment lock is exported and verified.

## Numerical evidence

- [ ] High-Mach velocity and spatial convergence gates pass on final data.
- [ ] A temporal convergence/residual criterion passes for every certified DVM case.
- [ ] Full-Mach KGFR results are rerun with training-only shared coordinate
  normalization (the launcher now enforces it; archived results are pending).
- [ ] Standard-MLP and matched-storage DVM baselines are complete and like-for-like.
- [ ] Three-seed summaries include uncertainty and failure accounting.

## Archival release

- [ ] A small public example dataset is included or linked.
- [ ] Final manifests, metrics, plotting inputs, and allowed checkpoints are archived.
- [ ] The archive has a persistent DOI and checksums.
- [ ] `CITATION.cff` contains the final article title, authors, and DOI.
- [ ] Code ownership, coauthor approval, and license are confirmed.
- [ ] The manuscript cites the exact Git tag and data DOI.

Do not label a result publication-grade until the applicable unchecked items
are complete.
