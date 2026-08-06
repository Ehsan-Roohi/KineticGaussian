#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {
    "x": ["x", "xs", "xstar", "x_over_lambda", "x_over_lambda1", "x_lambda1", "x_lambda"],
    "rho": ["rho", "density", "n"],
    "ux": ["ux", "u_x", "u", "velocity", "Vx"],
    "T": ["T", "temp", "temperature"],
    "qx": ["qx", "q_x", "heat_flux", "q"],
    "sigma_xx": ["sigma_xx", "sigmaxx", "sig_xx", "stress", "tau_xx", "tau"],
    "mxxx": ["mxxx", "m_xxx", "mxxx_cl", "m_cl_xxx"],
    "Rxx_closure": ["Rxx_closure", "Rxx_cl", "R_cl_xx", "Rxx", "R_xx"],
}

def find_key(z, name, required=False):
    for k in ALIASES[name]:
        if k in z.files:
            return k
    if required:
        raise KeyError(f"Missing {name}; available keys are:\n{z.files}")
    return None

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    out = {}
    kx = find_key(z, "x", required=True)
    x = np.asarray(z[kx], dtype=float).reshape(-1)
    order = np.argsort(x)
    out["x"] = x[order]

    for name in ["rho", "ux", "T", "qx", "sigma_xx", "mxxx", "Rxx_closure"]:
        k = find_key(z, name, required=False)
        if k is not None:
            arr = np.asarray(z[k], dtype=float).reshape(-1)
            if arr.size == x.size:
                out[name] = arr[order]
            else:
                print(f"[WARN] {path}: key {k} for {name} has size {arr.size}, expected {x.size}; skipped.")
    return out

def rel_l2_on_ref_grid(case, ref, field, mask=None):
    x_ref = ref["x"]
    y_ref = ref[field]
    x = case["x"]
    y = case[field]
    y_interp = np.interp(x_ref, x, y)

    if mask is None:
        mask = np.isfinite(y_ref) & np.isfinite(y_interp)
    else:
        mask = mask & np.isfinite(y_ref) & np.isfinite(y_interp)

    return np.linalg.norm(y_interp[mask] - y_ref[mask]) / (np.linalg.norm(y_ref[mask]) + 1e-300)

def midpoint_x0(x, rho):
    return x[np.argmin(np.abs(rho - 0.5*(rho[0] + rho[-1])))]

def make_masks(ref, shock_window):
    x = ref["x"]
    masks = {"full": np.ones_like(x, dtype=bool)}
    if "rho" in ref:
        x0 = midpoint_x0(x, ref["rho"])
    else:
        x0 = 0.5*(x.min() + x.max())
    masks["shock_window"] = np.abs(x - x0) <= shock_window

    # Active support masks for m and R, useful because these vanish in plateaus.
    for field in ["mxxx", "Rxx_closure"]:
        if field in ref:
            a = np.abs(ref[field])
            masks[f"{field}_active"] = a >= 0.05 * max(np.max(a), 1e-300)
    return masks

def fmt(x):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.2e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", nargs="+", required=True, help="DVM npz files from coarse to finest; last is reference")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--outdir", default="appendix_validation_dvm/mR_convergence")
    ap.add_argument("--shock-window", type=float, default=20.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = args.labels
    if labels is None:
        labels = [Path(p).stem for p in args.refs]
    if len(labels) != len(args.refs):
        raise ValueError("--labels length must match --refs length")

    data = [load_npz(p) for p in args.refs]
    ref = data[-1]
    ref_label = labels[-1]

    fields = [f for f in ["rho", "ux", "T", "qx", "sigma_xx", "mxxx", "Rxx_closure"] if f in ref]
    missing = [f for f in ["mxxx", "Rxx_closure"] if f not in ref]
    if missing:
        print("[WARN] Finest file does not contain:", missing)
        print("       Use hmom/densemicro DVM npz files if you want m and R convergence.")

    masks = make_masks(ref, args.shock_window)

    rows = []
    for label, case, path in zip(labels[:-1], data[:-1], args.refs[:-1]):
        for field in fields:
            if field not in case:
                rows.append({
                    "case": label, "file": path, "reference": ref_label,
                    "quantity": field, "relL2_full": np.nan,
                    "relL2_shock_window": np.nan, "relL2_active_support": np.nan,
                    "note": "missing in this file"
                })
                continue

            rel_full = rel_l2_on_ref_grid(case, ref, field, masks["full"])
            rel_shock = rel_l2_on_ref_grid(case, ref, field, masks["shock_window"])

            active_key = f"{field}_active"
            if active_key in masks:
                rel_active = rel_l2_on_ref_grid(case, ref, field, masks[active_key])
            else:
                rel_active = np.nan

            rows.append({
                "case": label, "file": path, "reference": ref_label,
                "quantity": field,
                "relL2_full": rel_full,
                "relL2_shock_window": rel_shock,
                "relL2_active_support": rel_active,
                "note": ""
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "dvm_mR_convergence_errors.csv", index=False)

    print("\n=== convergence errors ===")
    print(df.to_string(index=False))

    # Wide table for paper: use full-profile errors by default.
    wide = df.pivot(index="case", columns="quantity", values="relL2_full").reset_index()
    wide.to_csv(outdir / "dvm_mR_convergence_wide.csv", index=False)

    wanted = ["case", "rho", "ux", "T", "qx", "sigma_xx", "mxxx", "Rxx_closure"]
    for w in wanted:
        if w not in wide.columns:
            wide[w] = np.nan
    wide = wide[wanted]

    tex = []
    tex.append(r"\begin{table}")
    tex.append(r"\centering")
    tex.append(r"\caption{DVM grid and velocity-domain refinement. Relative \(L^2\) differences are computed against the finest member of the sequence.}")
    tex.append(r"\label{tab:dvm_mR_convergence}")
    tex.append(r"\begin{tabular}{lccccccc}")
    tex.append(r"\toprule")
    tex.append(r"Grid & \(\rho\) & \(u_x\) & \(T\) & \(q_x\) & \(\sigma_{xx}\) & \(m_{xxx}^{cl}\) & \(R_{xx}^{cl}\)\\")
    tex.append(r"\midrule")
    for _, r in wide.iterrows():
        tex.append(
            f"{r['case']} & {fmt(r['rho'])} & {fmt(r['ux'])} & {fmt(r['T'])} & "
            f"{fmt(r['qx'])} & {fmt(r['sigma_xx'])} & {fmt(r['mxxx'])} & {fmt(r['Rxx_closure'])}\\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    (outdir / "dvm_mR_convergence_table.tex").write_text("\n".join(tex))

    print("\n[saved]")
    print(outdir / "dvm_mR_convergence_errors.csv")
    print(outdir / "dvm_mR_convergence_wide.csv")
    print(outdir / "dvm_mR_convergence_table.tex")

if __name__ == "__main__":
    main()
