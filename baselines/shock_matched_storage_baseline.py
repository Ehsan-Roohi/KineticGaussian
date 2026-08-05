#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MOMENT_NAMES = ("rho", "ux", "T", "qx", "sig", "M300", "M400neq")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa - bb) / (np.linalg.norm(bb) + 1.0e-300))


def get_key(npz: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name], dtype=np.float64)
    raise KeyError(f"None of {names} found. Available keys: {npz.files}")


def uniform_indices(n: int, m: int) -> np.ndarray:
    if m < 2 or m > n:
        raise ValueError(f"Need 2 <= m <= n, got m={m}, n={n}")
    idx = np.rint(np.linspace(0, n - 1, m)).astype(np.int64)
    idx = np.unique(idx)
    if len(idx) != m:
        # Fill rare duplicate gaps deterministically.
        missing = [i for i in range(n) if i not in set(idx.tolist())]
        idx = np.sort(np.concatenate([idx, np.asarray(missing[: m - len(idx)])]))
    return idx


def weighted_quantile_indices(weight: np.ndarray, m: int) -> np.ndarray:
    """Adaptive 1-D node indices, always including both endpoints."""
    w = np.asarray(weight, dtype=np.float64).copy()
    if w.ndim != 1 or len(w) < m:
        raise ValueError("Invalid adaptive-grid weight array")
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    if not np.any(w > 0):
        return uniform_indices(len(w), m)
    w += 1.0e-6 * np.max(w)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    q = np.linspace(0.0, 1.0, m)
    idx = np.searchsorted(cdf, q, side="left").astype(np.int64)
    idx[0] = 0
    idx[-1] = len(w) - 1
    idx = np.unique(idx)
    # Fill missing indices by farthest-point insertion on index distance.
    while len(idx) < m:
        candidates = np.setdiff1d(np.arange(len(w)), idx, assume_unique=False)
        dist = np.min(np.abs(candidates[:, None] - idx[None, :]), axis=1)
        idx = np.sort(np.append(idx, candidates[np.argmax(dist)]))
    if len(idx) > m:
        idx = idx[uniform_indices(len(idx), m)]
    return np.asarray(idx, dtype=np.int64)


def infer_tensor_velocity_grid(v: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return three axes and map [ix,iy,iz] -> original velocity-column index."""
    vv = np.asarray(v, dtype=np.float64)
    if vv.ndim != 2 or vv.shape[1] != 3:
        raise ValueError(f"Expected v with shape (Nv,3), got {vv.shape}")
    axes = [np.unique(vv[:, j]) for j in range(3)]
    shape = tuple(len(a) for a in axes)
    if int(np.prod(shape)) != len(vv):
        raise ValueError(
            f"Velocity nodes are not a complete tensor product: axes={shape}, Nv={len(vv)}"
        )
    ijk = []
    for j, axis in enumerate(axes):
        ind = np.searchsorted(axis, vv[:, j])
        if not np.allclose(axis[ind], vv[:, j], rtol=0.0, atol=1.0e-7):
            raise ValueError(f"Could not map velocity component {j} to a unique axis")
        ijk.append(ind)
    grid_to_col = np.full(shape, -1, dtype=np.int64)
    grid_to_col[ijk[0], ijk[1], ijk[2]] = np.arange(len(vv), dtype=np.int64)
    if np.any(grid_to_col < 0):
        raise ValueError("Velocity tensor-product map contains missing entries")
    return axes, grid_to_col


def interpolation_brackets(src: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    hi = np.searchsorted(src, query, side="right")
    hi = np.clip(hi, 1, len(src) - 1)
    lo = hi - 1
    den = src[hi] - src[lo]
    t = (query - src[lo]) / (den + 1.0e-300)
    t = np.clip(t, 0.0, 1.0)
    return lo, hi, t


def interpolate_axis(arr: np.ndarray, src: np.ndarray, query: np.ndarray, axis: int) -> np.ndarray:
    lo, hi, t = interpolation_brackets(src, query)
    a0 = np.take(arr, lo, axis=axis)
    a1 = np.take(arr, hi, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = len(query)
    tt = t.reshape(shape)
    return (1.0 - tt) * a0 + tt * a1


@dataclass(frozen=True)
class Layout:
    name: str
    nx: int
    nvx: int
    nvy: int
    nvz: int
    x_scheme: str

    @property
    def stored(self) -> int:
        return int(self.nx * self.nvx * self.nvy * self.nvz)


def default_layouts(budget: int) -> List[Layout]:
    if budget == 4608:
        return [
            Layout("balanced_uniform", 9, 8, 8, 8, "uniform"),
            Layout("physics_uniform", 16, 11, 5, 5, "uniform"),
            Layout("physics_adaptive_x", 16, 11, 5, 5, "adaptive"),
        ]
    if budget == 5120:
        return [
            Layout("balanced_uniform", 10, 8, 8, 8, "uniform"),
            Layout("physics_uniform", 20, 10, 5, 5, "uniform"),
            Layout("physics_adaptive_x", 20, 10, 5, 5, "adaptive"),
        ]
    # Generic layouts use as much of the budget as possible. The physics
    # layout gives x and vx more nodes than the transverse velocities.
    n = max(2, int(budget ** 0.25))
    while (n + 1) ** 4 <= budget:
        n += 1
    balanced_nx = max(2, budget // (n**3))

    best = None
    for transverse in range(2, max(3, 2 * n + 1)):
        nvx = 2 * transverse
        nx = min(4 * nvx, budget // (nvx * transverse * transverse))
        if nx < max(2, nvx):
            continue
        stored = nx * nvx * transverse * transverse
        candidate = (stored, nx, nvx, transverse)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    _, physics_nx, physics_nvx, transverse = best
    return [
        Layout("balanced_uniform", balanced_nx, n, n, n, "uniform"),
        Layout("physics_uniform", physics_nx, physics_nvx, transverse, transverse, "uniform"),
        Layout("physics_adaptive_x", physics_nx, physics_nvx, transverse, transverse, "adaptive"),
    ]


def parse_layout(text: str) -> Layout:
    # name:nx,nvx,nvy,nvz[:uniform|adaptive]
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid layout specification: {text}")
    dims = [int(x) for x in parts[1].split(",")]
    if len(dims) != 4:
        raise ValueError(f"Layout requires four dimensions: {text}")
    scheme = parts[2] if len(parts) == 3 else "uniform"
    return Layout(parts[0], *dims, scheme)


def reference_moments(moment_path: Path, x_query: np.ndarray) -> Dict[str, np.ndarray]:
    tr = np.load(moment_path, allow_pickle=True)
    xt = get_key(tr, "x_mfp", "x")

    def interp(*names: str) -> np.ndarray:
        return np.interp(x_query, xt, get_key(tr, *names))

    rho = interp("rho")
    T = interp("T")
    m4raw = interp("M400_raw", "M400")
    return {
        "rho": rho,
        "ux": interp("ux"),
        "T": T,
        "qx": interp("qx"),
        "sig": interp("sigma_xx", "sig"),
        "M300": interp("M300_neq", "M300"),
        "M400neq": m4raw - 3.0 * rho * T * T,
    }


def compute_moments_for_layout(
    x: np.ndarray,
    f: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    axes: Sequence[np.ndarray],
    grid_to_col: np.ndarray,
    ref: Dict[str, np.ndarray],
    layout: Layout,
    f_floor: float,
    x_chunk: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, object]]:
    nx_full = len(x)
    nv_shape = tuple(len(a) for a in axes)
    if layout.nx > nx_full or any(m > n for m, n in zip((layout.nvx, layout.nvy, layout.nvz), nv_shape)):
        raise ValueError(f"Layout {layout} exceeds the stored grid")

    if layout.x_scheme == "uniform":
        ix = uniform_indices(nx_full, layout.nx)
    elif layout.x_scheme == "adaptive":
        gr = np.abs(np.gradient(ref["rho"], x))
        score = gr / (np.max(gr) + 1e-300)
        for key in ("qx", "sig"):
            a = np.abs(ref[key])
            score += a / (np.max(a) + 1e-300)
        ix = weighted_quantile_indices(1.0 + 5.0 * score, layout.nx)
    else:
        raise ValueError(f"Unknown x scheme: {layout.x_scheme}")

    iv = [
        uniform_indices(len(axes[0]), layout.nvx),
        uniform_indices(len(axes[1]), layout.nvy),
        uniform_indices(len(axes[2]), layout.nvz),
    ]
    coarse_cols = grid_to_col[np.ix_(iv[0], iv[1], iv[2])].ravel()
    coarse = np.asarray(f[np.ix_(ix, coarse_cols)], dtype=np.float64)
    coarse = np.log(np.maximum(coarse, f_floor)).reshape(
        layout.nx, layout.nvx, layout.nvy, layout.nvz
    )

    # Separable trilinear velocity interpolation onto the original DVM tensor grid.
    lv = interpolate_axis(coarse, axes[0][iv[0]], axes[0], axis=1)
    lv = interpolate_axis(lv, axes[1][iv[1]], axes[1], axis=2)
    lv = interpolate_axis(lv, axes[2][iv[2]], axes[2], axis=3)
    lv = lv.reshape(layout.nx, -1)

    # Put velocity nodes and weights into the same tensor-grid order as lv.
    order = grid_to_col.ravel()
    vg = np.asarray(v[order], dtype=np.float64)
    wg = np.asarray(w[order], dtype=np.float64)
    v2g = np.sum(vg * vg, axis=1)

    pred = {k: np.zeros(nx_full, dtype=np.float64) for k in MOMENT_NAMES}
    lo_x, hi_x, tx = interpolation_brackets(x[ix], x)
    log_floor = math.log(f_floor)

    for i0 in range(0, nx_full, x_chunk):
        i1 = min(nx_full, i0 + x_chunk)
        t = tx[i0:i1, None]
        logf = (1.0 - t) * lv[lo_x[i0:i1]] + t * lv[hi_x[i0:i1]]
        ff = np.exp(np.clip(logf, log_floor, 30.0))
        fw = ff * wg[None, :]
        rho = np.sum(fw, axis=1)
        m1 = fw @ vg
        u = m1 / np.maximum(rho[:, None], 1.0e-300)
        e2 = fw @ v2g
        T = (e2 / np.maximum(rho, 1.0e-300) - np.sum(u * u, axis=1)) / 3.0
        c = vg[None, :, :] - u[:, None, :]
        c2 = np.sum(c * c, axis=2)
        cx = c[:, :, 0]
        sxx_raw = np.sum(fw * cx * cx, axis=1)
        qx = 0.5 * np.sum(fw * c2 * cx, axis=1)
        m3 = np.sum(fw * cx**3, axis=1)
        m4 = np.sum(fw * cx**4, axis=1)

        pred["rho"][i0:i1] = rho
        pred["ux"][i0:i1] = u[:, 0]
        pred["T"][i0:i1] = T
        pred["qx"][i0:i1] = qx
        pred["sig"][i0:i1] = sxx_raw - rho * T
        pred["M300"][i0:i1] = m3
        pred["M400neq"][i0:i1] = m4 - 3.0 * rho * T * T

    errors = {key: rel_l2(pred[key], ref[key]) for key in MOMENT_NAMES}
    meta = {
        "layout": layout.name,
        "x_scheme": layout.x_scheme,
        "dimensions": [layout.nx, layout.nvx, layout.nvy, layout.nvz],
        "stored_values": layout.stored,
        "x_indices": ix.tolist(),
        "vx_indices": iv[0].tolist(),
        "vy_indices": iv[1].tolist(),
        "vz_indices": iv[2].tolist(),
    }
    return pred, errors, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Matched-storage coarse-grid baseline for M5 shock log-f data")
    ap.add_argument("--fullstate", required=True)
    ap.add_argument("--moments", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--layout", action="append", default=[], help="name:nx,nvx,nvy,nvz[:scheme]")
    ap.add_argument("--f-floor", type=float, default=1.0e-30)
    ap.add_argument("--x-chunk", type=int, default=4)
    args = ap.parse_args()

    fullstate = Path(args.fullstate)
    moment_path = Path(args.moments)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not fullstate.exists():
        raise FileNotFoundError(fullstate)
    if not moment_path.exists():
        raise FileNotFoundError(moment_path)

    print(f"Loading full-state shock file: {fullstate}", flush=True)
    d = np.load(fullstate, allow_pickle=True)
    required = ("x", "f", "v", "w")
    missing = [k for k in required if k not in d.files]
    if missing:
        raise KeyError(f"Missing full-state arrays: {missing}")
    x = np.asarray(d["x"], dtype=np.float64)
    f = np.asarray(d["f"], dtype=np.float32)
    v = np.asarray(d["v"], dtype=np.float64)
    w = np.asarray(d["w"], dtype=np.float64)
    print(f"Nx={len(x)}, Nv={len(v)}, f shape={f.shape}", flush=True)

    axes, grid_to_col = infer_tensor_velocity_grid(v)
    print(f"Velocity tensor grid: {[len(a) for a in axes]}", flush=True)
    ref = reference_moments(moment_path, x)

    layouts = [parse_layout(s) for s in args.layout] if args.layout else default_layouts(args.budget)
    for layout in layouts:
        if layout.stored > args.budget:
            raise ValueError(f"Layout {layout.name} stores {layout.stored}, above budget {args.budget}")

    rows: List[Dict[str, object]] = []
    details: Dict[str, object] = {
        "fullstate": str(fullstate),
        "moments": str(moment_path),
        "raw_phase_values": int(f.shape[0] * f.shape[1]),
        "budget": int(args.budget),
        "velocity_grid": [len(a) for a in axes],
        "results": {},
    }

    for layout in layouts:
        print(f"\nRunning layout {layout.name}: {layout}", flush=True)
        pred, errors, meta = compute_moments_for_layout(
            x=x,
            f=f,
            v=v,
            w=w,
            axes=axes,
            grid_to_col=grid_to_col,
            ref=ref,
            layout=layout,
            f_floor=args.f_floor,
            x_chunk=args.x_chunk,
        )
        max_field = max(errors, key=errors.get)
        tag = f"{layout.name}_{layout.x_scheme}"
        table = {"x": x, **pred}
        with open(out / f"moments_{tag}.csv", "w", newline="") as fp:
            wr = csv.writer(fp)
            wr.writerow(table.keys())
            wr.writerows(zip(*table.values()))
        result = {
            **meta,
            "parameter_budget": int(args.budget),
            "nominal_compression": float((f.shape[0] * f.shape[1]) / layout.stored),
            "errors": errors,
            "max_all": float(errors[max_field]),
            "max_field": max_field,
        }
        details["results"][tag] = result
        rows.append({
            "tag": tag,
            "layout": layout.name,
            "x_scheme": layout.x_scheme,
            "budget": args.budget,
            "stored_values": layout.stored,
            "nominal_compression": result["nominal_compression"],
            "max_all": result["max_all"],
            "max_field": max_field,
            **errors,
        })
        print(json.dumps(result, indent=2), flush=True)

    with open(out / "summary.csv", "w", newline="") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    (out / "summary.json").write_text(json.dumps(details, indent=2))

    # Lightweight comparison plot; calculations do not depend on matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [r["tag"] for r in rows]
        vals = [100.0 * float(r["max_all"]) for r in rows]
        plt.figure(figsize=(7.0, 4.0))
        plt.bar(np.arange(len(labels)), vals)
        plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        plt.ylabel("Maximum relative moment error [%]")
        plt.tight_layout()
        plt.savefig(out / "shock_baseline_max_error.pdf")
        plt.close()
    except Exception as exc:
        print(f"Plotting skipped: {exc}")

    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
