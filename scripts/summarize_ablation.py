#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import statistics
from pathlib import Path


MOMENT_KEYS = ["rho", "ux", "T", "qx", "sig", "M300", "M400neq"]
EXPERIMENTS = [
    ("all + per_case (V1)", "conditional_holdout-M3_N256_s*_moment"),
    ("all + shared", "ablation_all_shared_holdout-M3_N256_s*_moment"),
    ("amplitude + per_case", "ablation_amp_percase_holdout-M3_N512_s*_moment"),
    ("amplitude + shared (V2)", "conditional_v2_amp_holdout-M3_N512_s*_moment"),
]


def mean_std(values: list[float]) -> str:
    return f"{statistics.mean(values):6.2f}+/-{statistics.stdev(values):5.2f}"


def main() -> None:
    rows: list[dict] = []
    missing: list[str] = []
    for label, pattern in EXPERIMENTS:
        paths = sorted(
            glob.glob(f"runs/conditional/{pattern}/eval_all_cases/metrics.json")
        )
        if len(paths) != 3:
            missing.append(f"{label}: expected 3 metrics files, found {len(paths)}")
            continue
        for path in paths:
            payload = json.loads(Path(path).read_text())
            for case_name, case in payload["cases"].items():
                moments = case["moment_relative_l2"]
                row = {
                    "experiment": label,
                    "run": payload["run_name"],
                    "case": case_name,
                    "split": case["split"],
                    "parameter_count": case["parameter_count"],
                    "f_rel_l2": case["sampled_f_relative_l2"],
                    "logf_rmse": case["weighted_logf_rmse"],
                }
                row.update({key: moments.get(key, float("nan")) for key in MOMENT_KEYS})
                rows.append(row)

    if missing:
        raise SystemExit("Ablation is incomplete:\n  " + "\n  ".join(missing))

    output = Path("runs/ablation_summary.csv")
    fields = [
        "experiment", "run", "case", "split", "parameter_count",
        "f_rel_l2", "logf_rmse", *MOMENT_KEYS,
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nFOUR-CELL ABLATION: MEAN +/- STD OVER THREE SEEDS (PERCENT)")
    print("experiment                  case   params     f-L2      rho       ux        T       qx      sig     M300  M400neq")
    for label, _ in EXPERIMENTS:
        for case_name in ("M2p5", "M3", "M5"):
            group = [
                row for row in rows
                if row["experiment"] == label and row["case"] == case_name
            ]

            def values(key: str) -> list[float]:
                return [100.0 * float(row[key]) for row in group]

            metrics = " ".join(
                mean_std(values(key))
                for key in ("f_rel_l2", *MOMENT_KEYS)
            )
            print(
                f"{label:27s} {case_name:4s} {int(group[0]['parameter_count']):7d} {metrics}"
            )
    print(f"\nWROTE: {output}")


if __name__ == "__main__":
    main()
