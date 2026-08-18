#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def choose(candidates: list[Path], label: str, allow_running: bool = False) -> Path:
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    preferred = [path for path in files if "running" not in path.name and "microanchor" not in path.name]
    if preferred:
        return preferred[-1]
    if allow_running and files:
        return files[-1]
    lines = "\n".join(str(path) for path in files[:20])
    raise FileNotFoundError(f"Could not resolve {label}. Candidates were:\n{lines or '(none)'}")


def validate_fullstate(path: Path) -> None:
    with np.load(path, allow_pickle=True) as archive:
        required = {"x", "f", "v", "w", "rho", "ux", "T", "qx"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise KeyError(f"{path} is not a full-state shock file; missing {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover existing Unity DVM shocks and write the case manifest")
    parser.add_argument("--bgk-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.bgk_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    m25 = choose(
        list((root / "ref" / "mach_sweep_extra" / "M2p5").glob("standing_M2p5*fullstate*.npz")),
        "M2.5 full-state file",
        allow_running=True,
    )
    search_roots = [root / "shock_phase_gaussian" / "data", root / "data", root / "ref"]
    m3_candidates: list[Path] = []
    m5_candidates: list[Path] = []
    m3_moments: list[Path] = []
    m5_moments: list[Path] = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        m3_candidates.extend(search_root.rglob("standing_M3*fullstate.npz"))
        m5_candidates.extend(search_root.rglob("standing_M5*fullstate.npz"))
        m3_moments.extend(search_root.rglob("M3_DVM_hmom.npz"))
        m5_moments.extend(search_root.rglob("M5_DVM_hmom.npz"))
    m3 = choose(m3_candidates, "M3 full-state file")
    m5 = choose(m5_candidates, "M5 full-state file")
    m3_moment = choose(m3_moments, "M3 high-moment file")
    m5_moment = choose(m5_moments, "M5 high-moment file")
    for path in (m25, m3, m5):
        validate_fullstate(path)

    payload = {
        "description": "Existing Unity DVM shock cases for leave-one-Mach-out experiments",
        "mach_bounds": [2.5, 5.0],
        "cases": [
            {"name": "M2p5", "mach": 2.5, "data_path": str(m25), "moment_path": None},
            {"name": "M3", "mach": 3.0, "data_path": str(m3), "moment_path": str(m3_moment)},
            {"name": "M5", "mach": 5.0, "data_path": str(m5), "moment_path": str(m5_moment)},
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[kinetic-gaussian] wrote {output.resolve()}")


if __name__ == "__main__":
    main()
