#!/usr/bin/env python
"""Gate production DVM cases on medium-to-fine profile convergence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MACRO_KEYS = ("rho", "ux", "T")
NONEQ_ALIASES = {
    "qx": ("qx_neq_discrete", "qx"),
    "sig": ("sig_neq_discrete", "sig", "sigma_xx"),
    "M300": ("M300_neq",),
    "M400neq": ("M400_neq",),
}


def get_array(data: np.lib.npyio.NpzFile, aliases: tuple[str, ...]) -> np.ndarray:
    for key in aliases:
        if key in data.files:
            return np.asarray(data[key], dtype=np.float64)
    raise KeyError(f"None of {aliases} found in {data.files}")


def coordinate(data: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("x_mfp", "x"):
        if key in data.files:
            return np.asarray(data[key], dtype=np.float64)
    raise KeyError("No x coordinate in DVM result")


def relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), 1.0e-30))


def compare(candidate_path: Path, reference_path: Path) -> dict:
    candidate = np.load(candidate_path, allow_pickle=True)
    reference = np.load(reference_path, allow_pickle=True)
    xc = coordinate(candidate)
    xr = coordinate(reference)
    lo, hi = max(xc.min(), xr.min()), min(xc.max(), xr.max())
    mask = (xr >= lo) & (xr <= hi)
    x_eval = xr[mask]
    errors = {}
    for key in MACRO_KEYS:
        candidate_values = get_array(candidate, (key,))
        reference_values = get_array(reference, (key,))[mask]
        errors[key] = relative_l2(np.interp(x_eval, xc, candidate_values), reference_values)
    for key, aliases in NONEQ_ALIASES.items():
        candidate_values = get_array(candidate, aliases)
        reference_values = get_array(reference, aliases)[mask]
        errors[key] = relative_l2(np.interp(x_eval, xc, candidate_values), reference_values)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--macro-tolerance", type=float, default=0.01)
    parser.add_argument("--noneq-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    suite = json.loads(Path(args.suite_manifest).read_text(encoding="utf-8"))
    grouped = {}
    for row in suite["convergence"]:
        grouped.setdefault(row["case"], {})[row["level"]] = row
    report = {"thresholds": vars(args), "cases": {}, "pass": True}
    for case_name, levels in grouped.items():
        for required in ("coarse", "medium", "fine"):
            path = Path(levels[required]["moments_path"])
            if not path.exists():
                raise SystemExit(f"Missing {case_name} {required} result: {path}")
        case_report = {
            "coarse_vs_medium": compare(Path(levels["coarse"]["moments_path"]), Path(levels["medium"]["moments_path"])),
            "medium_vs_fine": compare(Path(levels["medium"]["moments_path"]), Path(levels["fine"]["moments_path"])),
        }
        gate = case_report["medium_vs_fine"]
        case_report["pass"] = bool(
            all(gate[key] <= args.macro_tolerance for key in MACRO_KEYS)
            and all(gate[key] <= args.noneq_tolerance for key in NONEQ_ALIASES)
        )
        report["cases"][case_name] = case_report
        report["pass"] = bool(report["pass"] and case_report["pass"])
        print(f"{case_name}: pass={case_report['pass']} medium_vs_fine={gate}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE: {output}")
    if not report["pass"]:
        raise SystemExit("Grid-convergence gate failed; production M7/M8/M10 jobs remain blocked")


if __name__ == "__main__":
    main()

