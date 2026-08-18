from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import ShockFullState
from .moments import moments_from_fullstate_np
from .utils import rel_l2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--moment-path", default=None)
    ap.add_argument("--x-chunk", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = ShockFullState(args.path, moment_path=args.moment_path)
    print(data.summary())
    pred = moments_from_fullstate_np(data, x_chunk=args.x_chunk)
    errors = {}
    mapping = {"rho": "rho", "ux": "ux", "T": "T", "qx": "qx", "sig": "sig"}
    for pk, rk in mapping.items():
        ref = data.moment_ref[rk]
        err = rel_l2(pred[pk], ref)
        max_abs = float(np.max(np.abs(pred[pk] - ref)))
        errors[pk] = {"rel_l2": err, "max_abs": max_abs}
        print(f"{pk:8s} rel_l2={err:.6e} max_abs={max_abs:.6e}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
