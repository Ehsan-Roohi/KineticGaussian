from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a smaller full-state NPZ for code debugging.")
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=300)
    ap.add_argument("--nv", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = np.load(args.path, allow_pickle=True)
    x = d["x"]
    f = d["f"]
    v = d["v"]
    w = d["w"]
    Nx, Nv = f.shape
    ix = np.linspace(0, Nx - 1, min(args.nx, Nx)).round().astype(int)
    if args.nv < Nv:
        # Keep a mix of low-speed and random velocities.
        speed = np.linalg.norm(v, axis=1)
        core = np.argsort(speed)[: max(1, args.nv // 2)]
        rest = rng.choice(Nv, size=args.nv - len(core), replace=False)
        iv = np.unique(np.concatenate([core, rest]))
        if len(iv) > args.nv:
            iv = iv[: args.nv]
    else:
        iv = np.arange(Nv)
    out = {k: d[k][ix] if k in ["x", "x_scaled", "x_mfp", "rho", "ux", "T", "qx", "sig"] else d[k] for k in d.files if k != "f"}
    out["v"] = v[iv]
    out["w"] = w[iv]
    out["f"] = f[np.ix_(ix, iv)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"wrote {out_path}")
    print(f"subset f shape: {out['f'].shape}")


if __name__ == "__main__":
    main()
