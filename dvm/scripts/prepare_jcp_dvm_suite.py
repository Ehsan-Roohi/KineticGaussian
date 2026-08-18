#!/usr/bin/env python
"""Generate collision-safe task files for the JCP DVM campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def task(
    case_name: str,
    case: dict,
    level_name: str,
    level: dict,
    output_root: Path,
    temporal_convergence: dict | None = None,
) -> dict:
    nx = int(round(case["nx"] * level["spatial_factor"]))
    run_name = f"{case_name}_{level_name}_gl{level['grid_gauss_order']}_nx{nx}"
    outdir = output_root / case_name / level_name
    row = {
        "case": case_name,
        "mach": case["mach"],
        "level": level_name,
        "run_name": run_name,
        "xhalf": case["xhalf"],
        "nx": nx,
        "steps": case["steps"],
        "gpu_class": case["gpu_class"],
        "grid_gauss_order": level["grid_gauss_order"],
        "grid_core_sigma": level["grid_core_sigma"],
        "grid_tail_sigma": level["grid_tail_sigma"],
        "grid_interval_sigma": level["grid_interval_sigma"],
        "outdir": str(outdir),
        "moments_path": str(outdir / f"standing_{run_name}.npz"),
        "fullstate_path": str(outdir / f"standing_{run_name}_fullstate.npz"),
        "figure_path": str(outdir / f"standing_{run_name}.png"),
    }
    if temporal_convergence is not None:
        row["temporal_convergence"] = dict(temporal_convergence)
    return row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    print(f"[jcp-dvm] wrote {len(rows)} tasks: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dvm/configs/jcp_high_mach_cases.json")
    parser.add_argument("--output-dir", default="configs/dvm_jcp_generated")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    levels = config["levels"]
    cases = config["cases"]
    temporal_convergence = config.get("temporal_convergence")
    if temporal_convergence is not None:
        required = {
            "macro_relative_l2",
            "noneq_relative_l2",
            "required_consecutive_checks",
            "min_step_fraction",
        }
        missing = required - set(temporal_convergence)
        if missing:
            raise ValueError(f"Temporal convergence config missing {sorted(missing)}")
        if float(temporal_convergence["macro_relative_l2"]) <= 0.0:
            raise ValueError("macro_relative_l2 must be positive")
        if float(temporal_convergence["noneq_relative_l2"]) <= 0.0:
            raise ValueError("noneq_relative_l2 must be positive")
        if int(temporal_convergence["required_consecutive_checks"]) < 1:
            raise ValueError("required_consecutive_checks must be at least one")
        min_fraction = float(temporal_convergence["min_step_fraction"])
        if not 0.0 <= min_fraction <= 1.0:
            raise ValueError("min_step_fraction must be between zero and one")

    convergence = []
    for name in config["convergence_cases"]:
        for level_name in ("coarse", "medium", "fine"):
            convergence.append(
                task(
                    name, cases[name], level_name, levels[level_name], data_root,
                    temporal_convergence,
                )
            )
    production = [
        task(
            name, cases[name], "medium", levels["medium"], data_root,
            temporal_convergence,
        )
        for name in config["production_cases"]
    ]

    groups = {
        "convergence_mid.jsonl": [row for row in convergence if row["gpu_class"] == "mid"],
        "convergence_high.jsonl": [row for row in convergence if row["gpu_class"] == "high"],
        "production_mid.jsonl": [row for row in production if row["gpu_class"] == "mid"],
        "production_high.jsonl": [row for row in production if row["gpu_class"] == "high"],
    }
    for filename, rows in groups.items():
        write_jsonl(output_dir / filename, rows)

    manifest = {
        "config": str(Path(args.config).resolve()),
        "data_root": str(data_root),
        "temporal_convergence": temporal_convergence,
        "convergence": convergence,
        "production": production,
    }
    (output_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    all_paths = [row["fullstate_path"] for row in convergence + production]
    if len(all_paths) != len(set(all_paths)):
        raise SystemExit("Generated output paths are not collision-safe")
    print(f"[jcp-dvm] validated {len(convergence)} convergence and {len(production)} production tasks")


if __name__ == "__main__":
    main()
