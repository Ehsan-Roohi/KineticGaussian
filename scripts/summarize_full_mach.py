#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MOMENT_KEYS = ["rho", "ux", "T", "qx", "sig", "M300", "M400", "M400neq"]


def finite_mean(values: list[float]) -> float:
    return statistics.mean(values)


def finite_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M3/M6 holdouts and blind M12 extrapolation")
    parser.add_argument(
        "--task-file",
        default="configs/generalization/full_mach_generated/tasks.txt",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args()

    task_file = Path(args.task_file)
    config_paths = [Path(line.strip()) for line in task_file.read_text().splitlines() if line.strip()]
    if not config_paths:
        raise SystemExit(f"No tasks in {task_file}")

    rows = []
    missing = []
    for config_path in config_paths:
        cfg = json.loads(config_path.read_text())
        run_name = cfg["run_name"]
        run_root = Path(cfg.get("output_dir", "runs/conditional")) / run_name
        metrics_path = run_root / "eval_manifest" / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        metrics = json.loads(metrics_path.read_text())
        holdout = cfg["holdout_cases"][0]
        case = metrics["cases"][holdout]
        moments = case.get("moment_relative_l2", {})
        row = {
            "run": run_name,
            "holdout": holdout,
            "mach": float(case["mach"]),
            "degree": int(cfg["model"]["mach_degree"]),
            "kernels": int(cfg["model"]["num_kernels"]),
            "seed": int(cfg["seed"]),
            "parameter_count": int(case["parameter_count"]),
            "mach_bounds": json.dumps(cfg["mach_bounds"]),
            "f_rel_l2": float(case["sampled_f_relative_l2"]),
            "logf_rmse": float(case["weighted_logf_rmse"]),
        }
        for key in MOMENT_KEYS:
            row[key] = float(moments.get(key, float("nan")))
        rows.append(row)

    if missing and not args.allow_incomplete:
        preview = "\n".join(missing[:10])
        raise SystemExit(f"Missing {len(missing)} evaluations; rerun after completion:\n{preview}")
    if not rows:
        raise SystemExit("No completed full-Mach evaluations found")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "full_mach_runs.csv"
    run_fields = list(rows[0])
    with run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["holdout"], row["degree"], row["seed"])))

    summaries = []
    for holdout in ("M3", "M6", "M12"):
        for degree in (2, 3):
            group = [row for row in rows if row["holdout"] == holdout and row["degree"] == degree]
            if not group:
                continue
            summary = {
                "holdout": holdout,
                "mach": group[0]["mach"],
                "degree": degree,
                "kernels": group[0]["kernels"],
                "parameter_count": group[0]["parameter_count"],
                "seeds_completed": len(group),
                "mach_bounds": group[0]["mach_bounds"],
            }
            for key in ("f_rel_l2", "logf_rmse", *MOMENT_KEYS):
                values = [float(row[key]) for row in group]
                summary[f"{key}_mean"] = finite_mean(values)
                summary[f"{key}_std"] = finite_std(values)
            summaries.append(summary)

    summary_csv = output_dir / "full_mach_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nFULL-MACH GENERALIZATION: MEAN +/- STD OVER SEEDS (PERCENT)")
    print("holdout degree params       f-L2       rho        ux         T         qx        sig      M300   M400neq")
    for row in summaries:
        def fmt(key: str) -> str:
            return f"{100*row[f'{key}_mean']:6.2f}+/-{100*row[f'{key}_std']:5.2f}"

        print(
            f"{row['holdout']:7s} D{row['degree']} {row['parameter_count']:6d} "
            + " ".join(fmt(key) for key in ("f_rel_l2", "rho", "ux", "T", "qx", "sig", "M300", "M400neq"))
        )
    print(f"\nWROTE: {run_csv}")
    print(f"WROTE: {summary_csv}")
    if missing:
        print(f"INCOMPLETE: {len(missing)} evaluations were missing")


if __name__ == "__main__":
    main()
