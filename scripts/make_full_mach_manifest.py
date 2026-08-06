#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_CASES = {
    "M1p5": 1.5,
    "M2": 2.0,
    "M2p5": 2.5,
    "M3": 3.0,
    "M4": 4.0,
    "M5": 5.0,
    "M6": 6.0,
    "M8": 8.0,
    "M12": 12.0,
}

REQUIRED_FULLSTATE_KEYS = {"x", "f", "v", "w", "rho", "ux", "T", "qx"}
HIGH_MOMENT_KEYS = {
    "M300_neq",
    "M400_raw",
    "mxxx",
    "Rxx_neq",
    "Rxx_closure",
}
REJECTED_PATH_PARTS = {
    "bad_wrong_default_m2_runs_20260613",
    "smoke_200_steps_20260613",
}


def parse_tags(text: str) -> list[str]:
    tags = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [tag for tag in tags if tag not in DEFAULT_CASES]
    if unknown:
        raise ValueError(f"Unknown case tags {unknown}; supported={list(DEFAULT_CASES)}")
    if len(tags) != len(set(tags)):
        raise ValueError(f"Duplicate case tags: {tags}")
    return tags


def archive_keys(path: Path) -> set[str]:
    try:
        with np.load(path, allow_pickle=True) as archive:
            return set(archive.files)
    except Exception as exc:
        raise RuntimeError(f"Cannot inspect NPZ archive {path}: {exc}") from exc


def is_rejected(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    lowered_name = path.name.lower()
    return (
        bool(lowered_parts & REJECTED_PATH_PARTS)
        or "incomplete" in lowered_name
        or "running" in lowered_name
        or "microanchor" in lowered_name
        or "lite" in lowered_name
    )


def tag_matches(path: Path, tag: str) -> bool:
    name = path.name
    return name.startswith(f"standing_{tag}_") or name.startswith(f"{tag}_DVM_")


def fullstate_score(path: Path, tag: str) -> tuple[int, int, str]:
    score = 0
    if path.name.endswith("_fullstate_fullstate.npz"):
        score += 100
    elif path.name.endswith("_fullstate.npz"):
        score += 60
    if path.name.startswith(f"standing_{tag}_"):
        score += 20
    if tag in path.parts:
        score += 10
    if "final" in path.parts:
        score -= 5
    return score, path.stat().st_size, str(path)


def discover_fullstate(root: Path, tag: str) -> tuple[Path, set[str]]:
    candidates: list[tuple[Path, set[str]]] = []
    for search_root in (root / "ref" / "mach_sweep", root / "ref" / "mach_sweep_extra"):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.npz"):
            if is_rejected(path) or not tag_matches(path, tag):
                continue
            keys = archive_keys(path)
            if REQUIRED_FULLSTATE_KEYS <= keys:
                candidates.append((path.resolve(), keys))
    if not candidates:
        raise FileNotFoundError(f"No authoritative full-state archive found for {tag}")
    return max(candidates, key=lambda item: fullstate_score(item[0], tag))


def companion_candidates(data_path: Path) -> list[Path]:
    candidates: list[Path] = []
    name = data_path.name
    if name.endswith("_fullstate_fullstate.npz"):
        candidates.append(data_path.with_name(name.replace("_fullstate_fullstate.npz", "_fullstate.npz")))
    if name.endswith("_fullstate.npz"):
        candidates.append(data_path.with_name(name.replace("_fullstate.npz", ".npz")))
    return candidates


def discover_moment_file(root: Path, tag: str, data_path: Path, data_keys: set[str]) -> tuple[Path | None, set[str]]:
    if HIGH_MOMENT_KEYS & data_keys:
        return data_path, data_keys

    candidates: list[Path] = companion_candidates(data_path)
    for search_root in (root / "ref" / "mach_sweep", root / "ref" / "mach_sweep_extra"):
        if search_root.exists():
            candidates.extend(search_root.rglob(f"*{tag}*hmom*.npz"))

    valid: list[tuple[int, int, Path, set[str]]] = []
    seen: set[Path] = set()
    preferred = set(companion_candidates(data_path))
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file() or is_rejected(path) or not tag_matches(path, tag):
            continue
        seen.add(path)
        keys = archive_keys(path)
        if not (HIGH_MOMENT_KEYS & keys):
            continue
        score = 100 if path in preferred else 0
        if path.parent == data_path.parent:
            score += 20
        if "ext" in path.stem.lower():
            score += 5
        valid.append((score, len(HIGH_MOMENT_KEYS & keys), path, keys))
    if not valid:
        return None, set()
    _, _, path, keys = max(valid, key=lambda item: (item[0], item[1], str(item[2])))
    return path, keys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover authoritative Unity DVM shocks from Mach 1.5 through Mach 12"
    )
    parser.add_argument("--bgk-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tags", default=",".join(DEFAULT_CASES))
    args = parser.parse_args()

    root = Path(args.bgk_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    tags = parse_tags(args.tags)

    cases = []
    for tag in tags:
        data_path, data_keys = discover_fullstate(root, tag)
        moment_path, moment_keys = discover_moment_file(root, tag, data_path, data_keys)
        cases.append(
            {
                "name": tag,
                "mach": DEFAULT_CASES[tag],
                "data_path": str(data_path),
                "moment_path": None if moment_path is None else str(moment_path),
                "data_size_bytes": data_path.stat().st_size,
                "fullstate_keys": sorted(data_keys),
                "high_moment_keys": sorted(HIGH_MOMENT_KEYS & moment_keys),
            }
        )

    machs = [float(case["mach"]) for case in cases]
    payload = {
        "description": "Authoritative Unity DVM normal-shock sweep; lite, smoke, running, and known-bad files excluded",
        "mach_bounds": [min(machs), max(machs)],
        "cases": cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[kinetic-gaussian] wrote {output.resolve()}")


if __name__ == "__main__":
    main()
