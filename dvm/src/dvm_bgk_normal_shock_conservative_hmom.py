#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dvm_bgk_normal_shock_conservative import (
    normal_shock_states,
    make_vgrid,
    state_maxwellian,
    moments,
    moments_chunked,
    conservative_discrete_maxwellian,
    conservative_discrete_maxwellian_chunked,
    advance_bgk_chunked,
    find_crossing,
    recenter_f,
)
from dvm_velocity_grid import (
    CompositeGridSpec,
    composite_velocity_quadrature,
    grid_metadata,
)


def higher_moments(f, v, v2, w, rho, ux, T, qx, sig, chunk=64):
    """
    Higher-order diagnostics.

    Important:
    mxxx_neq and Rxx_neq are computed from f - Md, where Md is the local
    conservative discrete Maxwellian. This removes equilibrium/quadrature
    offsets and gives quantities that vanish in the upstream/downstream
    Maxwellian plateaus.

    Rxx_closure additionally subtracts the leading Grad-13 stress contribution.
    We keep both Rxx_neq and Rxx_closure for diagnostics.
    """
    nx = f.shape[0]
    device = f.device
    dtype = f.dtype

    keys = [
        "M300_neq", "M120_neq", "M102_neq",
        "M400_raw", "M400_neq", "M400_eq_discrete", "M400_continuum_excess",
        "M220_raw", "M202_raw", "M040_raw",
        "qx_neq_discrete", "sig_neq_discrete",
        "mxxx", "Rxx_neq", "Rxx_closure",
        "skewx_neq", "kurtx_raw",
        "Qxx_bgk", "Qx_bgk",
        "mxxx_norm", "Rxx_neq_norm", "Rxx_closure_norm",
        "Qxx_norm", "Qx_norm",
    ]

    out = {k: torch.zeros(nx, device=device, dtype=dtype) for k in keys}
    eps = torch.tensor(1.0e-30, device=device, dtype=dtype)

    for i0 in range(0, nx, chunk):
        i1 = min(nx, i0 + chunk)

        Fi = f[i0:i1, :]
        rhoi = rho[i0:i1]
        uxi = ux[i0:i1]
        Ti = T[i0:i1]

        # Local conservative discrete Maxwellian for exact equilibrium subtraction.
        Md = conservative_discrete_maxwellian(v, v2, w, rhoi, uxi, Ti, niter=4)
        Gi = Fi - Md

        Cx = v[None, :, 0] - uxi[:, None]
        Cy = v[None, :, 1]
        Cz = v[None, :, 2]
        C2 = Cx**2 + Cy**2 + Cz**2
        ww = w[None, :]

        # Nonequilibrium/excess odd and mixed moments
        M300_neq = torch.sum(Gi * ww * Cx**3, dim=1)
        M120_neq = torch.sum(Gi * ww * Cx * Cy**2, dim=1)
        M102_neq = torch.sum(Gi * ww * Cx * Cz**2, dim=1)
        qx_neq_discrete = 0.5 * torch.sum(Gi * ww * C2 * Cx, dim=1)
        sig_neq_discrete = torch.sum(Gi * ww * (Cx**2 - C2 / 3.0), dim=1)

        # Raw even moments kept only for diagnostics/kurtosis.
        M400_raw = torch.sum(Fi * ww * Cx**4, dim=1)
        M400_eq_discrete = torch.sum(Md * ww * Cx**4, dim=1)
        M400_neq = torch.sum(Gi * ww * Cx**4, dim=1)
        M400_continuum_excess = M400_raw - 3.0 * rhoi * Ti**2
        M220_raw = torch.sum(Fi * ww * Cx**2 * Cy**2, dim=1)
        M202_raw = torch.sum(Fi * ww * Cx**2 * Cz**2, dim=1)
        M040_raw = torch.sum(Fi * ww * Cy**4, dim=1)

        # R13-style higher-order moment diagnostics.
        mxxx = torch.sum(Gi * ww * (Cx**3 - (3.0/5.0)*Cx*C2), dim=1)
        Rxx_neq = torch.sum(Gi * ww * C2 * (Cx**2 - C2/3.0), dim=1)

        # Optional closure residual after removing leading Grad-13 stress contribution.
        # This is diagnostic only; for paper use both Rxx_neq and Rxx_closure.
        Rxx_closure = Rxx_neq - 7.0 * Ti * sig_neq_discrete

        # BGK production terms for stress and heat flux modes.
        Qxx_bgk = -sig_neq_discrete
        Qx_bgk = -2.0 * qx_neq_discrete

        skewx_neq = M300_neq / torch.clamp(rhoi * Ti**1.5, min=eps)
        kurtx_raw = M400_raw / torch.clamp(rhoi * Ti**2, min=eps)

        mxxx_norm = mxxx / torch.clamp(rhoi * Ti**1.5, min=eps)
        Rxx_neq_norm = Rxx_neq / torch.clamp(rhoi * Ti**2, min=eps)
        Rxx_closure_norm = Rxx_closure / torch.clamp(rhoi * Ti**2, min=eps)
        Qxx_norm = Qxx_bgk / torch.clamp(rhoi * Ti, min=eps)
        Qx_norm = Qx_bgk / torch.clamp(rhoi * Ti**1.5, min=eps)

        sl = slice(i0, i1)
        out["M300_neq"][sl] = M300_neq
        out["M120_neq"][sl] = M120_neq
        out["M102_neq"][sl] = M102_neq
        out["qx_neq_discrete"][sl] = qx_neq_discrete
        out["sig_neq_discrete"][sl] = sig_neq_discrete
        out["M400_raw"][sl] = M400_raw
        out["M400_eq_discrete"][sl] = M400_eq_discrete
        out["M400_neq"][sl] = M400_neq
        out["M400_continuum_excess"][sl] = M400_continuum_excess
        out["M220_raw"][sl] = M220_raw
        out["M202_raw"][sl] = M202_raw
        out["M040_raw"][sl] = M040_raw
        out["mxxx"][sl] = mxxx
        out["Rxx_neq"][sl] = Rxx_neq
        out["Rxx_closure"][sl] = Rxx_closure
        out["skewx_neq"][sl] = skewx_neq
        out["kurtx_raw"][sl] = kurtx_raw
        out["Qxx_bgk"][sl] = Qxx_bgk
        out["Qx_bgk"][sl] = Qx_bgk
        out["mxxx_norm"][sl] = mxxx_norm
        out["Rxx_neq_norm"][sl] = Rxx_neq_norm
        out["Rxx_closure_norm"][sl] = Rxx_closure_norm
        out["Qxx_norm"][sl] = Qxx_norm
        out["Qx_norm"][sl] = Qx_norm

    return out


def save_npz(path, x_scaled, x_mfp, rho, ux, T, qx, sig, states, xhalf_mfp, extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    qx_corrected = extra.get("qx_neq_discrete", qx)
    sig_corrected = extra.get("sig_neq_discrete", sig)

    data = {
        "x": x_scaled,
        "x_mfp": x_mfp,
        "rho": rho.detach().cpu().numpy(),
        "ux": ux.detach().cpu().numpy(),
        "T": T.detach().cpu().numpy(),
        "qx": qx_corrected.detach().cpu().numpy(),
        "qx_raw": qx.detach().cpu().numpy(),
        "sigma_xx": sig_corrected.detach().cpu().numpy(),
        "sig": sig_corrected.detach().cpu().numpy(),
        "sig_raw": sig.detach().cpu().numpy(),
        "states": np.array([
            states["rho1"], states["u1"], states["T1"],
            states["rho2"], states["u2"], states["T2"]
        ]),
        "xhalf_mfp": np.array(xhalf_mfp),
        "kn_eff": np.array(1.0/(2.0*xhalf_mfp)),
    }

    for k, val in extra.items():
        data[k] = val.detach().cpu().numpy()

    np.savez(path, **data)
    print(f"[saved] {path}", flush=True)


def plot_basic(path, out):
    z = np.load(path)
    x = z["x"]
    fig, axes = plt.subplots(5, 1, figsize=(9, 10.5), sharex=True)
    for ax, key, lab in zip(
        axes,
        ["rho", "ux", "T", "qx", "sigma_xx"],
        [r"$\rho$", r"$u_x$", r"$T$", r"$q_x$", r"$\sigma_{xx}$"]
    ):
        ax.plot(x, z[key], lw=2)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"scaled coordinate $x/(2L_\lambda)$")
    fig.suptitle("Conservative DVM BGK normal shock with higher-moment diagnostics")
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"[plot] {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/standing_M2_conservative_hmom_dvm.npz")
    ap.add_argument("--fig", default="figures/standing_M2_conservative_hmom_dvm_profiles.png")
    ap.add_argument("--M1", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=5.0/3.0)
    ap.add_argument("--rho1", type=float, default=1.0)
    ap.add_argument("--T1", type=float, default=1.0)
    ap.add_argument("--xhalf-mfp", type=float, default=30.0)
    ap.add_argument("--nx", type=int, default=1200)
    ap.add_argument("--nvx", type=int, default=48)
    ap.add_argument("--nvy", type=int, default=18)
    ap.add_argument("--nvz", type=int, default=18)
    ap.add_argument("--vmax", type=float, default=12.0)
    ap.add_argument("--grid-mode", choices=("uniform", "composite"), default="uniform")
    ap.add_argument("--grid-gauss-order", type=int, default=3)
    ap.add_argument("--grid-core-sigma", type=float, default=4.0)
    ap.add_argument("--grid-tail-sigma", type=float, default=6.0)
    ap.add_argument("--grid-interval-sigma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--cfl", type=float, default=0.75)
    ap.add_argument("--center-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1500)
    ap.add_argument("--corr-iters", type=int, default=4)
    ap.add_argument("--x-chunk", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"[device] {device}, dtype={dtype}", flush=True)

    states = normal_shock_states(args.M1, args.gamma, args.rho1, args.T1)
    print("[states]", states, flush=True)

    if args.grid_mode == "composite":
        grid_spec = CompositeGridSpec(
            gauss_order=args.grid_gauss_order,
            core_sigma=args.grid_core_sigma,
            tail_sigma=args.grid_tail_sigma,
            interval_sigma=args.grid_interval_sigma,
        )
        axis_nodes, axis_weights = composite_velocity_quadrature(states, grid_spec)
        grid_info = grid_metadata(axis_nodes, grid_spec)
        v, v2, w = make_vgrid(
            args.nvx, args.nvy, args.nvz, args.vmax, device, dtype,
            axis_nodes=axis_nodes, axis_weights=axis_weights,
        )
    else:
        grid_info = {
            "mode": "uniform", "shape": [args.nvx, args.nvy, args.nvz],
            "velocity_count": args.nvx * args.nvy * args.nvz, "vmax": args.vmax,
        }
        v, v2, w = make_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)
    print(f"[vgrid] {grid_info}", flush=True)

    ML = state_maxwellian(v, v2, w, states["rho1"], states["u1"], states["T1"])
    MR = state_maxwellian(v, v2, w, states["rho2"], states["u2"], states["T2"])

    for name, M in [("left", ML), ("right", MR)]:
        rho, ux, T, qx, sig = moments(M[None, :], v, v2, w)
        print(
            f"[quad check] {name}: rho={float(rho):.8f}, ux={float(ux):.8f}, "
            f"T={float(T):.8f}, q={float(qx):.3e}, sig={float(sig):.3e}",
            flush=True
        )

    x = torch.linspace(-args.xhalf_mfp, args.xhalf_mfp, args.nx, device=device, dtype=dtype)
    dx = float((x[1]-x[0]).detach().cpu())
    vmax_streamwise = float(torch.max(torch.abs(v[:, 0])).detach().cpu())
    dt = args.cfl * dx / vmax_streamwise
    print(f"[x] [-{args.xhalf_mfp},{args.xhalf_mfp}], nx={args.nx}, dx={dx:.4e}, vmax_x={vmax_streamwise:.4e}, dt={dt:.4e}", flush=True)

    H = 0.5*(1.0 + torch.tanh(x/3.0))
    rho0 = (1-H)*states["rho1"] + H*states["rho2"]
    ux0 = (1-H)*states["u1"] + H*states["u2"]
    T0 = (1-H)*states["T1"] + H*states["T2"]

    f = conservative_discrete_maxwellian_chunked(
        v, v2, w, rho0, ux0, T0, niter=args.corr_iters, x_chunk=args.x_chunk
    )

    rho_mid = 0.5*(states["rho1"] + states["rho2"])
    last_rho = None

    running_out = str(Path(args.out).with_name(Path(args.out).stem + "_running.npz"))

    for step in range(1, args.steps+1):
        # Coordinate is scaled by upstream mean free path, so collision time is 1.
        f = advance_bgk_chunked(
            f, ML, MR, v, v2, w, dx, dt,
            corr_iters=args.corr_iters, x_chunk=args.x_chunk,
        )

        if step % args.center_every == 0:
            rho, ux, T, qx, sig = moments_chunked(f, v, v2, w, args.x_chunk)
            xs = find_crossing(x, rho, rho_mid)
            if abs(xs) > 1.0e-10:
                f = recenter_f(f, x, xs, ML, MR)

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            rho, ux, T, qx, sig = moments_chunked(f, v, v2, w, args.x_chunk)
            xs = find_crossing(x, rho, rho_mid)

            maxdrho = 0.0
            if last_rho is not None:
                maxdrho = float(torch.max(torch.abs(rho-last_rho)).detach().cpu())
            last_rho = rho.detach().clone()

            print(
                f"[step] {step:7d}/{args.steps} center={xs:+.4e} "
                f"rhoL={float(rho[0]):.6f} rhoR={float(rho[-1]):.6f} "
                f"uL={float(ux[0]):.6f} uR={float(ux[-1]):.6f} "
                f"TL={float(T[0]):.6f} TR={float(T[-1]):.6f} "
                f"maxdrho={maxdrho:.3e}",
                flush=True
            )

            x_mfp = x.detach().cpu().numpy()
            x_scaled = x_mfp/(2.0*args.xhalf_mfp)
            extra = higher_moments(f, v, v2, w, rho, ux, T, qx, sig)
            save_npz(running_out, x_scaled, x_mfp, rho, ux, T, qx, sig, states, args.xhalf_mfp, extra)

    rho, ux, T, qx, sig = moments_chunked(f, v, v2, w, args.x_chunk)
    xs = find_crossing(x, rho, rho_mid)
    f = recenter_f(f, x, xs, ML, MR)
    rho, ux, T, qx, sig = moments_chunked(f, v, v2, w, args.x_chunk)

    x_mfp = x.detach().cpu().numpy()
    x_scaled = x_mfp/(2.0*args.xhalf_mfp)
    extra = higher_moments(f, v, v2, w, rho, ux, T, qx, sig)


    # -------------------------------------------------------------------------
    # Sparse DVM velocity-distribution anchors for PINN micro-anchoring.
    # Saved at the final DVM state, from the main scope where F and velocity
    # grid variables exist.
    # -------------------------------------------------------------------------
    try:
        from dvm_micro_anchor_utils import save_micro_anchors_from_locals
        _micro_out = str(args.out).replace(".npz", "_microanchors.npz")
        _lv = locals()
        print("[micro-anchor scope keys]", sorted([k for k in _lv.keys() if k in ["f","F","Fn","v","vx","vy","vz","cx","cy","cz","w","ww","weights","x","x_mfp","x_scaled","rho","ux","T"]]), flush=True)
        save_micro_anchors_from_locals(
            _lv,
            out=_micro_out,
            x_targets=(-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0),
            n_micro_per_x=256,
        )
    except Exception as e:
        print("[micro-anchor skipped]", repr(e), flush=True)

    save_npz(args.out, x_scaled, x_mfp, rho, ux, T, qx, sig, states, args.xhalf_mfp, extra)
    plot_basic(args.out, args.fig)


if __name__ == "__main__":
    main()
