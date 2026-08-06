# KineticGaussian

Positive, compact Gaussian representations of rarefied shock distributions, with a strict leave-one-Mach-out experiment for flow-level generalization.

This repository contains the original single-shock KGFR implementation and a flow-level generalization experiment. The new model conditions every Gaussian kernel on Mach number through a low-order Legendre expansion. It is trained on complete DVM distributions at selected Mach numbers and evaluated on a shock whose distribution was never used during training.

## Main generalization question

Can one compact positive phase-space representation interpolate an unseen nonequilibrium flow while preserving both the full distribution and its high-order moments?

The primary split is:

- training: M2.5 and M5 full-state DVM shocks;
- held out: M3 full-state DVM shock;
- capacities: 256, 512, and 1024 kernels;
- uncertainty: seeds 1234, 2026, and 3407;
- objectives: log-distribution only and log-distribution plus sampled moment regularization;
- baseline: coarse DVM grids constrained to the same stored-value budget.

M3 is not loaded by the training process. Its Mach number is used only as a query value at evaluation time. Coordinate-domain bounds are fixed in the manifest and do not use held-out distribution values.

## Model

For normalized phase coordinates `z=(x,vx,vy,vz)` and normalized Mach number `m`,

```text
log f_hat(m,z) = logsumexp_n [ a_n(m) - 0.5 Q_n(m,z) ].
```

Kernel amplitudes, centers, widths, and `(x,vx)` correlations are low-order Legendre functions of `m`. The primary two-training-case experiment uses degree one; a quadratic Mach law would be underdetermined by only two Mach values. The log-sum-exp construction makes `f_hat` positive everywhere. The `xvx` covariance block represents the position/streamwise-velocity coupling that is characteristic of a shock.

## One-line Unity submission

After cloning this repository on Unity, run:

```bash
bash scripts/unity_submit.sh /project/pi_roohie_umass_edu/BGK_shock
```

The script discovers the existing M2.5, M3, and M5 full-state files, validates their required arrays, writes a manifest, and submits a small preflight smoke job. Only after that job succeeds does Slurm release the 18-run GPU array and three matched-storage CPU baselines. At most four GPU jobs run simultaneously. The launcher uses the existing Unity Python environment at:

```text
/work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dde-tf/bin/python
```

Override it when needed with `KGFR_PYTHON=/path/to/python`.

## Diagnostic evaluation of existing checkpoints

Evaluate every completed v1 checkpoint on the two training cases and the held-out M3 case without overwriting the original held-out outputs:

```bash
bash scripts/unity_submit_endpoint_eval.sh
```

Results are written under `eval_all_cases/`. Comparing training-case and held-out errors distinguishes representation underfitting from failure to interpolate in Mach number.

## Version 2 held-out-Mach suite

Version 2 fixes one Gaussian geometry across Mach number and conditions only the positive kernel amplitudes. It also uses one coordinate normalization computed exclusively from the M2.5 and M5 training domains. This prevents kernel identities from crossing or permuting with Mach and avoids case-specific coordinate maps. Submit the nine-run `N=128,256,512`, three-seed, moment-aware suite with:

```bash
bash scripts/unity_submit_v2.sh /project/pi_roohie_umass_edu/BGK_shock
```

Evaluation now reports the nonequilibrium fourth-order diagnostic `M400neq = M400 - 3 rho T^2` when a raw fourth-moment reference is available. The original v1 launcher and checkpoint format remain supported.

## Conditioning/normalization ablation

The v1-to-v2 change altered both Mach conditioning and coordinate normalization. The
six-run diagnostic suite fills the two missing cells without overwriting either suite:

- all kernel parameters conditioned on Mach with shared training-only normalization,
  using `N=256` (5,120 parameters);
- amplitude-only conditioning with per-case normalization, using `N=512` (5,632
  parameters).

Each cell uses seeds `1234`, `2026`, and `3407` with the moment-aware objective. A
smoke test gates training, and all-case evaluation on M2.5/M3/M5 is submitted
automatically after every training task succeeds:

```bash
bash scripts/unity_submit_ablation.sh /project/pi_roohie_umass_edu/BGK_shock
```

After the dependent evaluation array completes, compare all four cells with:

```bash
python scripts/summarize_ablation.py
```

## Local validation

```bash
python -m unittest tests/test_conditional_model.py
python tests/smoke_end_to_end.py
```

The end-to-end smoke test creates three small synthetic DVM files, trains for three steps on two Mach cases, evaluates the third case, and verifies the metrics artifact.

## Outputs

Each conditional run writes:

```text
runs/conditional/<run_name>/
  config.json
  history.csv
  best.pt
  last.pt
  training_history.png
  eval/
    metrics.json
    moments_<case>.csv
    moments_<case>.png
    ladder_<case>.png
```

`metrics.json` reports phase-sample errors, relative moment errors, parameter bytes, raw DVM bytes, and nominal compression. Baseline outputs are under `runs/matched_baselines/`.

## Existing single-case reproduction

The original commands remain available:

```bash
python train_phase_gaussian.py --config configs/M3_phase_xvx.json
python evaluate_phase_gaussian.py --config configs/M3_phase_xvx.json --checkpoint runs/M3_phase_xvx_N256_moment/best.pt --x-stride 5
```

Expected NPZ keys are `x`, `f`, `v`, `w`, `rho`, `ux`, `T`, `qx`, and `sig` or `sigma_xx`. Optional high-moment files can add `M300_neq` and `M400_raw`.

## Reproducibility note

The repository intentionally excludes DVM arrays, checkpoints, and generated configs. The Unity launcher records absolute source paths and every hyperparameter in each generated config. Do not interpret the synthetic smoke-test errors as scientific results; only completed Unity DVM runs belong in the paper.
