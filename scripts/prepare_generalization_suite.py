#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

def parse_csv(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def conditional_parameter_count(kernels: int, degree: int, variant: str, conditioning: str) -> int:
    terms = degree + 1
    correlation = 1 if variant == "xvx" else 0
    if conditioning == "all":
        return int(kernels * terms * (4 + 4 + 1 + correlation))
    if conditioning == "amplitude":
        return int(kernels * (4 + 4 + correlation + terms))
    raise ValueError(f"Unknown conditioning: {conditioning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the reproducible held-out-Mach Slurm task matrix")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="configs/generalization/generated")
    parser.add_argument("--holdouts", default="M3")
    parser.add_argument("--capacities", default="256,512,1024")
    parser.add_argument("--seeds", default="1234,2026,3407")
    parser.add_argument("--objectives", default="logf,moment")
    parser.add_argument("--conditionings", default="all")
    parser.add_argument("--coordinate-normalization", choices=("per_case", "shared_training"), default="per_case")
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Optional collision-safe run-name prefix for diagnostic suites",
    )
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--steps", type=int, default=80000)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = list(manifest.get("cases", []))
    if not raw_cases:
        raise ValueError(f"No cases in {manifest_path}")
    names = [str(case["name"]) for case in raw_cases]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate case names: {names}")
    by_name = {str(case["name"]): case for case in raw_cases}
    holdouts = parse_csv(args.holdouts)
    missing_holdouts = [name for name in holdouts if name not in by_name]
    if missing_holdouts:
        raise KeyError(f"Unknown holdout cases {missing_holdouts}; available={names}")
    capacities = parse_csv(args.capacities, int)
    seeds = parse_csv(args.seeds, int)
    objectives = parse_csv(args.objectives)
    conditionings = parse_csv(args.conditionings)
    unknown_objectives = sorted(set(objectives) - {"logf", "moment"})
    if unknown_objectives:
        raise ValueError(f"Unknown objectives: {unknown_objectives}")
    unknown_conditionings = sorted(set(conditionings) - {"all", "amplitude"})
    if unknown_conditionings:
        raise ValueError(f"Unknown conditionings: {unknown_conditionings}")
    if args.run_prefix is not None and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_prefix) is None:
        raise ValueError(
            "--run-prefix must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task_paths: list[str] = []
    baseline_tasks: list[dict] = []
    # Two training Mach values identify a linear Mach law. A quadratic law
    # would be underdetermined in the primary M2.5/M5 -> M3 experiment.
    variant, degree = "xvx", 1
    for holdout in holdouts:
        train_cases = [name for name in names if name != holdout]
        if len(train_cases) < 2:
            raise ValueError("A held-out experiment requires at least two training Mach cases")
        holdout_spec = by_name[holdout]
        holdout_data_path = Path(str(holdout_spec["data_path"])).expanduser()
        holdout_moment_value = holdout_spec.get("moment_path")
        holdout_moment_path = None if holdout_moment_value is None else Path(str(holdout_moment_value)).expanduser()
        if not holdout_data_path.is_absolute():
            holdout_data_path = (manifest_path.parent / holdout_data_path).resolve()
        if holdout_moment_path is not None and not holdout_moment_path.is_absolute():
            holdout_moment_path = (manifest_path.parent / holdout_moment_path).resolve()
        if holdout_moment_path is None:
            print(f"[kinetic-gaussian] baseline skipped for {holdout}: no separate high-moment file")
        for kernels in capacities:
            parameter_count = max(
                conditional_parameter_count(kernels, degree, variant, conditioning)
                for conditioning in conditionings
            )
            if holdout_moment_path is not None and not args.skip_baselines:
                baseline_tasks.append(
                    {
                        "name": f"baseline_holdout-{holdout}_N{kernels}",
                        "fullstate": str(holdout_data_path),
                        "moments": str(holdout_moment_path),
                        "budget": parameter_count,
                        "out": str((Path("runs/matched_baselines") / f"holdout-{holdout}_N{kernels}").resolve()),
                    }
                )
            for conditioning in conditionings:
                for seed in seeds:
                    for objective in objectives:
                        prefix = args.run_prefix or (
                            "conditional" if conditioning == "all" else "conditional_v2_amp"
                        )
                        run_name = f"{prefix}_holdout-{holdout}_N{kernels}_s{seed}_{objective}"
                        cfg = {
                            "run_name": run_name,
                            "manifest_path": str(manifest_path),
                            "output_dir": "runs/conditional",
                            "device": "cuda",
                            "seed": seed,
                            "mach_bounds": manifest.get(
                                "mach_bounds",
                                [min(float(c["mach"]) for c in raw_cases), max(float(c["mach"]) for c in raw_cases)],
                            ),
                            "train_cases": train_cases,
                            "holdout_cases": [holdout],
                            "eval_cases": [holdout],
                            "data": {"coordinate_normalization": args.coordinate_normalization},
                            "model": {
                                "variant": variant,
                                "mach_degree": degree,
                                "conditioning": conditioning,
                                "num_kernels": kernels,
                                "log_scale_min": -6.0,
                                "log_scale_max": -0.7,
                                "init_log_scale": -2.4,
                                "init_log_amp": -7.0,
                                "init_samples": max(100000, 200 * kernels),
                            },
                            "sampling": {
                                "x_batch": 24,
                                "vel_per_x": 384,
                                "uniform_vel_frac": 0.15,
                                "mass_alpha": 0.55,
                                "f_floor": 1.0e-35,
                            },
                            "train": {
                                "steps": args.steps,
                                "lr": 5.0e-4,
                                "warmup_steps": 1000,
                                "min_lr_ratio": 0.05,
                                "center_lr_mult": 0.5,
                                "scale_lr_mult": 0.25,
                                "amp_lr_mult": 1.0,
                                "corr_lr_mult": 0.2,
                                "weight_decay": 0.0,
                                "grad_clip": 1.0,
                                "log_every": 50,
                                "save_every": 2000,
                                "eval_every": 1000,
                                "logf_loss": "huber",
                                "logf_error_clip": 20.0,
                                "lambda_moment": 0.0 if objective == "logf" else 0.01,
                                "moment_every": 10,
                                "moment_x_count": 12,
                                "moment_vel_count": 1536,
                                "moment_uniform_vel_frac": 0.10,
                                "moment_mass_alpha": 0.55,
                                "moment_keys": ["rho", "ux", "T", "qx", "sig", "M300"],
                            },
                            "evaluation": {
                                "moment_keys": ["rho", "ux", "T", "qx", "sig", "M300", "M400", "M400neq"]
                            },
                        }
                        config_path = output_dir / f"{run_name}.json"
                        config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
                        task_paths.append(str(config_path))

    task_file = output_dir / "tasks.txt"
    task_file.write_text("\n".join(task_paths) + "\n", encoding="utf-8")
    baseline_file = output_dir / "baseline_tasks.jsonl"
    baseline_file.write_text("\n".join(json.dumps(item) for item in baseline_tasks) + ("\n" if baseline_tasks else ""), encoding="utf-8")
    print(f"[kinetic-gaussian] wrote {len(task_paths)} GPU tasks to {task_file}")
    print(f"[kinetic-gaussian] wrote {len(baseline_tasks)} matched-storage tasks to {baseline_file}")


if __name__ == "__main__":
    main()
