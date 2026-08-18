from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--max-keys", type=int, default=100)
    args = ap.parse_args()
    path = Path(args.path)
    d = np.load(path, allow_pickle=True)
    print(f"file: {path}")
    print(f"keys: {d.files}")
    for k in d.files[: args.max_keys]:
        a = d[k]
        shape = getattr(a, "shape", None)
        dtype = getattr(a, "dtype", None)
        msg = f"{k:20s} shape={shape!s:18s} dtype={dtype}"
        try:
            if np.size(a) > 0 and np.issubdtype(a.dtype, np.number):
                msg += f" min={np.nanmin(a): .6e} max={np.nanmax(a): .6e}"
        except Exception:
            pass
        print(msg)


if __name__ == "__main__":
    main()
