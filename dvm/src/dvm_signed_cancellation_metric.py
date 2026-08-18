#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ALIASES = {
    "x": ["x", "xs", "xstar", "x_over_lambda", "x_over_lambda1", "x_lambda1"],
    "rho": ["rho", "density", "n"],
    "ux": ["ux", "u_x", "u", "velocity", "Vx"],
    "T": ["T", "temp", "temperature"],
    "sigma_xx": ["sigma_xx", "sigmaxx", "sig_xx", "stress", "tau_xx", "tau"],
    "G": ["G", "g", "f_minus_Md", "f_minus_M", "f_neq", "Fneq", "neq"],
    "f": ["f", "F", "dist", "distribution"],
    "vx": ["vx", "v_x", "cx_grid", "Vx_grid", "cvx"],
    "vy": ["vy", "v_y", "cy_grid", "Vy_grid", "cvy"],
    "vz": ["vz", "v_z", "cz_grid", "Vz_grid", "cvz"],
    "w": ["w", "weights", "quad_w", "wv", "weight"],
    "v": ["v", "vel", "velocity_grid", "cgrid"],
}

def find_key(z, names, required=False):
    if isinstance(names, str):
        names = ALIASES[names]
    for k in names:
        if k in z.files:
            return k
    if required:
        raise KeyError(f"Cannot find any of {names}. Available keys:\n{z.files}")
    return None

def get1(z, name, required=True):
    k = find_key(z, name, required=required)
    if k is None:
        return None
    return np.asarray(z[k])

def flatten_velocity_grid(z):
    # Case 1: separate flattened vx,vy,vz,w
    kvx = find_key(z, "vx", required=False)
    kvy = find_key(z, "vy", required=False)
    kvz = find_key(z, "vz", required=False)
    kw = find_key(z, "w", required=False)

    if kvx and kvy and kvz and kw:
        vx = np.asarray(z[kvx], float).reshape(-1)
        vy = np.asarray(z[kvy], float).reshape(-1)
        vz = np.asarray(z[kvz], float).reshape(-1)
        w = np.asarray(z[kw], float).reshape(-1)
        return vx, vy, vz, w

    # Case 2: v is Nv x 3
    kv = find_key(z, "v", required=False)
    if kv and kw:
        v = np.asarray(z[kv], float)
        w = np.asarray(z[kw], float).reshape(-1)
        if v.ndim == 2 and v.shape[1] >= 3:
            return v[:,0], v[:,1], v[:,2], w

    raise KeyError(
        "Could not identify velocity grid. Need either vx,vy,vz,w or v(Nv,3),w. "
        f"Available keys:\n{z.files}"
    )

def continuous_maxwellian(rho, ux, T, vx, vy, vz):
    Cx = vx[None,:] - rho[:,None]*0.0 - ux[:,None]
    Cy = vy[None,:]
    Cz = vz[None,:]
    C2 = Cx*Cx + Cy*Cy + Cz*Cz
    coef = rho[:,None] / ((2*np.pi*T[:,None])**1.5)
    return coef * np.exp(-C2/(2*T[:,None]))

def chunk_slices(n, chunk):
    for i in range(0, n, chunk):
        yield slice(i, min(i+chunk, n))

def safe_ratio(num, den):
    eps = 1e-14 * max(1.0, np.nanmax(np.abs(den)))
    return num / (np.abs(den) + eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="ref/standing_M2_x40_hmom_dvm_densemicro.npz")
    ap.add_argument("--outdir", default="appendix_validation_dvm/cancellation_metric")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--shock-window", type=float, default=20.0)
    ap.add_argument("--use-continuous-maxwellian-if-needed", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    z = np.load(args.ref, allow_pickle=True)
    print("[file]", args.ref)
    print("[keys]", z.files)

    x = np.asarray(z[find_key(z, "x", required=True)], float).reshape(-1)
    rho = np.asarray(z[find_key(z, "rho", required=True)], float).reshape(-1)
    ux = np.asarray(z[find_key(z, "ux", required=True)], float).reshape(-1)
    T = np.asarray(z[find_key(z, "T", required=True)], float).reshape(-1)

    ksig = find_key(z, "sigma_xx", required=False)
    sigma = np.asarray(z[ksig], float).reshape(-1) if ksig else None

    order = np.argsort(x)
    x, rho, ux, T = x[order], rho[order], ux[order], T[order]
    if sigma is not None:
        sigma = sigma[order]

    vx, vy, vz, w = flatten_velocity_grid(z)
    Nv = len(w)
    print(f"[velocity grid] Nv={Nv}")

    kG = find_key(z, "G", required=False)
    kf = find_key(z, "f", required=False)

    mode = None
    if kG is not None:
        Graw = np.asarray(z[kG])
        mode = f"G from key {kG}"
    elif kf is not None and args.use_continuous_maxwellian_if_needed:
        Fraw = np.asarray(z[kf])
        mode = f"f from key {kf}; continuous Maxwellian fallback"
    else:
        print("\n[ERROR] No nonequilibrium distribution was found.")
        print("Need one of these keys for G=f-Md:", ALIASES["G"])
        print("or f plus --use-continuous-maxwellian-if-needed.")
        print("Your file may contain only moment profiles, not phase-space data.")
        raise SystemExit(2)

    print("[mode]", mode)

    # Shape handling: expected Nx x Nv
    Nx = len(x)
    if kG is not None:
        Graw = np.asarray(Graw)
        if Graw.shape[0] != Nx and Graw.shape[-1] == Nx:
            Graw = np.swapaxes(Graw, 0, -1)
        if Graw.shape[0] != Nx or Graw.reshape(Nx, -1).shape[1] != Nv:
            raise ValueError(f"G shape {Graw.shape} incompatible with Nx={Nx}, Nv={Nv}")
        Graw = Graw.reshape(Nx, Nv)
    else:
        Fraw = np.asarray(Fraw)
        if Fraw.shape[0] != Nx and Fraw.shape[-1] == Nx:
            Fraw = np.swapaxes(Fraw, 0, -1)
        if Fraw.shape[0] != Nx or Fraw.reshape(Nx, -1).shape[1] != Nv:
            raise ValueError(f"f shape {Fraw.shape} incompatible with Nx={Nx}, Nv={Nv}")
        Fraw = Fraw.reshape(Nx, Nv)

    rows_profile = []
    summary_rows = []

    x0 = x[np.argmin(np.abs(rho - 0.5*(rho[0]+rho[-1])))]
    shock_mask = np.abs(x - x0) <= args.shock_window

    # Output arrays
    ratios = {name: np.full(Nx, np.nan) for name in ["sigma_xx", "qx", "mxxx", "Rxx_closure"]}
    moments = {name: np.full(Nx, np.nan) for name in ["sigma_xx", "qx", "mxxx", "Rxx_closure"]}

    for sl in chunk_slices(Nx, args.chunk):
        xs = x[sl]
        rhos = rho[sl]
        uxs = ux[sl]
        Ts = T[sl]
        ns = len(xs)

        Cx = vx[None,:] - uxs[:,None]
        Cy = vy[None,:]
        Cz = vz[None,:]
        C2 = Cx*Cx + Cy*Cy + Cz*Cz
        ww = w[None,:]

        if kG is not None:
            G = Graw[sl, :]
        else:
            M = continuous_maxwellian(rhos, uxs, Ts, vx, vy, vz)
            G = Fraw[sl, :] - M

        kernels = {
            "sigma_xx": Cx*Cx - C2/3.0,
            "qx": 0.5*C2*Cx,
            "mxxx": Cx**3 - (3.0/5.0)*Cx*C2,
            "Rxx_closure": (C2 - 7.0*Ts[:,None])*(Cx*Cx - C2/3.0),
        }

        for name, K in kernels.items():
            integrand = K * G * ww
            signed = np.sum(integrand, axis=1)
            unsigned = np.sum(np.abs(integrand), axis=1)
            moments[name][sl] = signed
            ratios[name][sl] = safe_ratio(unsigned, signed)

    # Save profile
    dfp = pd.DataFrame({"x": x})
    for name in ratios:
        dfp[f"{name}_moment"] = moments[name]
        dfp[f"{name}_cancellation_ratio"] = ratios[name]
        dfp[f"{name}_inverse_ratio"] = 1.0 / ratios[name]
    dfp.to_csv(outdir / "signed_cancellation_profiles.csv", index=False)

    # Summary: avoid plateau blowup by active support mask based on moment magnitude.
    for name in ratios:
        m = moments[name]
        c = ratios[name]
        active = shock_mask & np.isfinite(c) & (np.abs(m) >= 0.05*np.nanmax(np.abs(m)))
        if not np.any(active):
            active = shock_mask & np.isfinite(c)

        ipeak = np.nanargmax(np.abs(m))
        summary_rows.append({
            "observable": name,
            "max_abs_moment": np.nanmax(np.abs(m)),
            "x_at_max_abs_moment": x[ipeak],
            "cancellation_at_peak": c[ipeak],
            "median_cancellation_active": np.nanmedian(c[active]),
            "p90_cancellation_active": np.nanpercentile(c[active], 90),
            "p95_cancellation_active": np.nanpercentile(c[active], 95),
            "min_inverse_ratio_active": np.nanmin(1.0/c[active]),
            "active_points": int(np.sum(active)),
        })

    dfs = pd.DataFrame(summary_rows)
    dfs.to_csv(outdir / "signed_cancellation_summary.csv", index=False)

    print("\n=== signed cancellation summary ===")
    print(dfs.to_string(index=False))

    # LaTeX table
    def fmt(x):
        if not np.isfinite(x):
            return "--"
        return f"{x:.2e}"

    label_map = {
        "sigma_xx": r"\(\sigma_{xx}\)",
        "qx": r"\(q_x\)",
        "mxxx": r"\(m_{xxx}^{cl}\)",
        "Rxx_closure": r"\(R_{xx}^{cl}\)",
    }

    tex = []
    tex.append(r"\begin{table}")
    tex.append(r"\centering")
    tex.append(r"\caption{Signed-cancellation diagnostic for moment observability in the refined DVM shock. Larger values indicate that the signed moment is obtained as a smaller residual of larger positive and negative velocity-space contributions.}")
    tex.append(r"\label{tab:signed_cancellation_metric}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"Observable & cancellation at peak & median active cancellation & 95th percentile active cancellation\\")
    tex.append(r"\midrule")
    for _, r in dfs.iterrows():
        tex.append(
            f"{label_map.get(r['observable'], r['observable'])} & "
            f"{fmt(r['cancellation_at_peak'])} & "
            f"{fmt(r['median_cancellation_active'])} & "
            f"{fmt(r['p95_cancellation_active'])}\\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    (outdir / "signed_cancellation_table.tex").write_text("\n".join(tex))

    # Plot ratios
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for name in ["sigma_xx", "qx", "mxxx", "Rxx_closure"]:
        ax.semilogy(x, ratios[name], lw=2, label=name)
    ax.axvline(x0, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel(r"$x/\lambda_1$")
    ax.set_ylabel(r"signed-cancellation ratio")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "signed_cancellation_profiles.png", dpi=350)
    fig.savefig(outdir / "signed_cancellation_profiles.pdf")
    plt.close(fig)

    print("\n[saved]")
    print(outdir / "signed_cancellation_profiles.csv")
    print(outdir / "signed_cancellation_summary.csv")
    print(outdir / "signed_cancellation_table.tex")
    print(outdir / "signed_cancellation_profiles.png")

if __name__ == "__main__":
    main()
