#!/usr/bin/env python
"""Run one generated JCP DVM task using only files in this repository."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


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
    print("[jcp-dvm task]", json.dumps(task, indent=2), flush=True)
    print("[jcp-dvm command]", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise SystemExit(f"Solver returned without final full state: {final_path}")
    print(f"[jcp-dvm complete] {final_path}", flush=True)


if __name__ == "__main__":
    main()
