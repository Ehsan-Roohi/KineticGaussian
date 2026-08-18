#!/usr/bin/env python
"""Run one generated JCP DVM task using only files in this repository."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def require_temporal_convergence(path: Path) -> None:
    with np.load(path, allow_pickle=True) as data:
        required = {
            "temporal_gate_enabled",
            "temporal_converged",
            "temporal_history_json",
        }
        missing = required - set(data.files)
        if missing:
            raise SystemExit(f"Temporal metadata missing from {path}: {sorted(missing)}")
        if not bool(np.asarray(data["temporal_gate_enabled"]).item()):
            raise SystemExit(f"Temporal convergence gate was not enabled for {path}")
        if not bool(np.asarray(data["temporal_converged"]).item()):
            macro = float(np.asarray(data["temporal_latest_macro_relative_l2_max"]).item())
            noneq = float(np.asarray(data["temporal_latest_noneq_relative_l2_max"]).item())
            raise SystemExit(
                f"Temporal convergence gate failed for {path}: "
                f"latest macro={macro:.6e}, noneq={noneq:.6e}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rows = [line for line in Path(args.task_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.task_index < 0 or args.task_index >= len(rows):
        raise SystemExit(f"Task index {args.task_index} outside 0..{len(rows)-1}")
    task = json.loads(rows[args.task_index])
    outdir = Path(task["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)
    final_path = Path(task["fullstate_path"])
    if final_path.exists() and final_path.stat().st_size > 0:
        raise SystemExit(f"Refusing to overwrite completed full state: {final_path}")

    solver = repo_root / "dvm" / "src" / "dvm_bgk_normal_shock_conservative_hmom_densemicro.py"
    command = [
        args.python,
        str(solver),
        "--out", task["moments_path"],
        "--fig", task["figure_path"],
        "--M1", str(task["mach"]),
        "--gamma", "1.6666666666666667",
        "--rho1", "1.0",
        "--T1", "1.0",
        "--xhalf-mfp", str(task["xhalf"]),
        "--nx", str(task["nx"]),
        "--steps", str(task["steps"]),
        "--grid-mode", "composite",
        "--grid-gauss-order", str(task["grid_gauss_order"]),
        "--grid-core-sigma", str(task["grid_core_sigma"]),
        "--grid-tail-sigma", str(task["grid_tail_sigma"]),
        "--grid-interval-sigma", str(task["grid_interval_sigma"]),
        "--cfl", "0.75",
        "--center-every", "50",
        "--save-every", "2000",
        "--corr-iters", "5",
        "--x-chunk", os.environ.get("DVM_X_CHUNK", "256"),
        "--device", "cuda",
        "--dtype", os.environ.get("DVM_DTYPE", "float64"),
    ]
    temporal = task.get("temporal_convergence")
    if temporal is not None:
        command.extend(
            [
                "--temporal-macro-tol", str(temporal["macro_relative_l2"]),
                "--temporal-noneq-tol", str(temporal["noneq_relative_l2"]),
                "--temporal-required-checks", str(temporal["required_consecutive_checks"]),
                "--temporal-min-step-fraction", str(temporal["min_step_fraction"]),
            ]
        )
    print("[jcp-dvm task]", json.dumps(task, indent=2), flush=True)
    print("[jcp-dvm command]", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise SystemExit(f"Solver returned without final full state: {final_path}")
    if temporal is not None:
        require_temporal_convergence(Path(task["moments_path"]))
        require_temporal_convergence(final_path)
    print(f"[jcp-dvm complete] {final_path}", flush=True)


if __name__ == "__main__":
    main()
