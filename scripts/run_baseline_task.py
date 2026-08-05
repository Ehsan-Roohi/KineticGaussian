#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    lines = [line for line in Path(args.task_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.index < 0 or args.index >= len(lines):
        raise IndexError(f"Task index {args.index} outside 0..{len(lines) - 1}")
    task = json.loads(lines[args.index])
    command = [
        sys.executable,
        "baselines/shock_matched_storage_baseline.py",
        "--fullstate", task["fullstate"],
        "--moments", task["moments"],
        "--out", task["out"],
        "--budget", str(task["budget"]),
        "--x-chunk", "4",
    ]
    print("[kinetic-gaussian]", task["name"], flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
