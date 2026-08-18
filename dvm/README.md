# DVM reference solvers

This directory contains both the current JCP certification workflow and older
Unity launch scripts retained for provenance.

## Current workflow

- `configs/jcp_high_mach_cases.json`: campaign matrix;
- `src/dvm_velocity_grid.py`: composite velocity quadrature;
- `src/dvm_bgk_normal_shock_conservative_hmom_densemicro.py`: production solver;
- `scripts/audit_velocity_grid.py`: equilibrium quadrature audit;
- `scripts/prepare_jcp_dvm_suite.py`: generated task manifests;
- `scripts/check_jcp_grid_convergence.py`: spatial/velocity convergence gate;
- `scripts/unity_submit_jcp_dvm.sh`: supported Unity entry point;
- `scripts/unity_status_jcp_dvm.sh`: dependency-chain status.

Submit the current campaign from the repository root:

```bash
bash dvm/scripts/unity_submit_jcp_dvm.sh /project/pi_roohie_umass_edu/BGK_shock
```

## Legacy launchers

Files named `scripts/submit_dvm_M*_*.sh` preserve earlier uniform-grid and
pre-maintenance runs. They are useful for provenance and diagnosis, but their
outputs must not be treated as JCP-certified references. They remain in place
for now so existing Unity job records and commands are not broken.

The release gate and known limitations are documented in
`../docs/REPRODUCIBILITY.md` and `../docs/RELEASE_CHECKLIST.md`.
