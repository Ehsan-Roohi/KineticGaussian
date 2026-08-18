#!/usr/bin/env python
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def trap_weights(n, h, device, dtype):
    w = torch.ones(n, device=device, dtype=dtype) * h
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def make_vgrid(nvx, nvy, nvz, vmax, device, dtype):
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)

    wx = trap_weights(nvx, 2*vmax/(nvx-1), device, dtype)
    wy = trap_weights(nvy, 2*vmax/(nvy-1), device, dtype)
    wz = trap_weights(nvz, 2*vmax/(nvz-1), device, dtype)

    VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
    W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]

    v = torch.stack([VX.reshape(-1), VY.reshape(-1), VZ.reshape(-1)], dim=1)
    w = W.reshape(-1)
    v2 = v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2

    print(f"[vgrid] Nv={v.shape[0]} = {nvx}x{nvy}x{nvz}, vmax={vmax}", flush=True)
    return v, v2, w


def moments_2d(f, v, v2, w):
    # f: [Ncell, Nv]
    rho = torch.sum(f * w[None, :], dim=1)

    mx = torch.sum(f * v[None, :, 0] * w[None, :], dim=1)
    my = torch.sum(f * v[None, :, 1] * w[None, :], dim=1)

    ux = mx / torch.clamp(rho, min=1e-30)
    uy = my / torch.clamp(rho, min=1e-30)

    e2 = torch.sum(f * v2[None, :] * w[None, :], dim=1)
    T = (e2 / torch.clamp(rho, min=1e-30) - ux**2 - uy**2) / 3.0
    T = torch.clamp(T, min=1e-8)

    Cx = v[None, :, 0] - ux[:, None]
    Cy = v[None, :, 1] - uy[:, None]
    Cz = v[None, :, 2]
    C2 = Cx**2 + Cy**2 + Cz**2

    qx = 0.5 * torch.sum(C2 * Cx * f * w[None, :], dim=1)
    qy = 0.5 * torch.sum(C2 * Cy * f * w[None, :], dim=1)

    sig_xx = torch.sum((Cx**2 - C2/3.0) * f * w[None, :], dim=1)
    sig_yy = torch.sum((Cy**2 - C2/3.0) * f * w[None, :], dim=1)
    sig_xy = torch.sum((Cx * Cy) * f * w[None, :], dim=1)

    return rho, ux, uy, T, qx, qy, sig_xx, sig_xy, sig_yy


def conservative_discrete_maxwellian(v, v2, w, rho_t, ux_t, uy_t, T_t, niter=5, chunk=2048):
    """
    Discrete conservative Maxwellian on the given velocity grid.
    Matches rho, ux, uy, T under the same quadrature used by the DVM.
    """
    device = v.device
    dtype = v.dtype
    N = rho_t.numel()

    out = torch.empty((N, v.shape[0]), device=device, dtype=dtype)

    vx = v[:, 0]
    vy = v[:, 1]
    vz = v[:, 2]
    two_pi = torch.tensor(2.0 * math.pi, device=device, dtype=dtype)
    I3 = torch.eye(3, device=device, dtype=dtype)[None, :, :]

    for a in range(0, N, chunk):
        b = min(N, a + chunk)

        rho = rho_t[a:b].to(device=device, dtype=dtype)
        ux_target = ux_t[a:b].to(device=device, dtype=dtype)
        uy_target = uy_t[a:b].to(device=device, dtype=dtype)
        T_target = torch.clamp(T_t[a:b].to(device=device, dtype=dtype), min=1e-8)

        ux = ux_target.clone()
        uy = uy_target.clone()
        L = torch.log(T_target)

        for _ in range(niter):
            T = torch.exp(L)

            Cx = vx[None, :] - ux[:, None]
            Cy = vy[None, :] - uy[:, None]
            C2 = Cx**2 + Cy**2 + vz[None, :]**2

            logM0 = -1.5 * torch.log(two_pi * T[:, None]) - C2 / (2.0 * T[:, None])
            M0 = torch.exp(logM0)

            Z0 = torch.sum(M0 * w[None, :], dim=1)
            Zx = torch.sum(M0 * vx[None, :] * w[None, :], dim=1)
            Zy = torch.sum(M0 * vy[None, :] * w[None, :], dim=1)
            Z2 = torch.sum(M0 * v2[None, :] * w[None, :], dim=1)

            meanx = Zx / torch.clamp(Z0, min=1e-30)
            meany = Zy / torch.clamp(Z0, min=1e-30)
            E = Z2 / torch.clamp(Z0, min=1e-30)
            temp = (E - meanx**2 - meany**2) / 3.0

            r1 = meanx - ux_target
            r2 = meany - uy_target
            r3 = temp - T_target
            r = torch.stack([r1, r2, r3], dim=1)

            dlog_ux = Cx / T[:, None]
            dlog_uy = Cy / T[:, None]
            dlog_L = -1.5 + C2 / (2.0 * T[:, None])

            derivs = [dlog_ux, dlog_uy, dlog_L]
            Jcols = []

            invZ0 = 1.0 / torch.clamp(Z0, min=1e-30)

            for dlog in derivs:
                dZ0 = torch.sum(M0 * dlog * w[None, :], dim=1)
                dZx = torch.sum(M0 * vx[None, :] * dlog * w[None, :], dim=1)
                dZy = torch.sum(M0 * vy[None, :] * dlog * w[None, :], dim=1)
                dZ2 = torch.sum(M0 * v2[None, :] * dlog * w[None, :], dim=1)

                dmeanx = dZx * invZ0 - meanx * dZ0 * invZ0
                dmeany = dZy * invZ0 - meany * dZ0 * invZ0
                dE = dZ2 * invZ0 - E * dZ0 * invZ0
                dtemp = (dE - 2.0 * meanx * dmeanx - 2.0 * meany * dmeany) / 3.0

                Jcols.append(torch.stack([dmeanx, dmeany, dtemp], dim=1))

            J = torch.stack(Jcols, dim=2)
            J = J + 1.0e-7 * I3

            delta = torch.linalg.solve(J, -r)
            delta = torch.clamp(delta, -0.35, 0.35)

            ux = ux + delta[:, 0]
            uy = uy + delta[:, 1]
            L = L + delta[:, 2]

        T = torch.exp(L)
        Cx = vx[None, :] - ux[:, None]
        Cy = vy[None, :] - uy[:, None]
        C2 = Cx**2 + Cy**2 + vz[None, :]**2

        logM0 = -1.5 * torch.log(two_pi * T[:, None]) - C2 / (2.0 * T[:, None])
        M0 = torch.exp(logM0)
        Z0 = torch.sum(M0 * w[None, :], dim=1)
        scale = rho / torch.clamp(Z0, min=1e-30)
        out[a:b] = torch.clamp(scale[:, None] * M0, min=1e-30)

    return out


def initial_four_stream(x, y, u0, rho0, T0, device, dtype):
    X, Y = torch.meshgrid(x, y, indexing="ij")

    rho = torch.full_like(X, rho0)
    T = torch.full_like(X, T0)

    # Four quadrants move toward the center.
    ux = torch.where(X < 0, torch.tensor(u0, device=device, dtype=dtype),
                    torch.tensor(-u0, device=device, dtype=dtype))
    uy = torch.where(Y < 0, torch.tensor(u0, device=device, dtype=dtype),
                    torch.tensor(-u0, device=device, dtype=dtype))

    return rho.reshape(-1), ux.reshape(-1), uy.reshape(-1), T.reshape(-1)


def save_npz(path, x, y, fields, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y, **fields, **meta)
    print(f"[saved] {path}", flush=True)


def plot_npz(path, out):
    z = np.load(path)
    x = z["x"]
    y = z["y"]
    extent = [x.min(), x.max(), y.min(), y.max()]

    rho = z["rho"]
    ux = z["ux"]
    uy = z["uy"]
    T = z["T"]
    qx = z["qx"]
    qy = z["qy"]
    sig_xx = z["sigma_xx"]
    sig_xy = z["sigma_xy"]
    sig_yy = z["sigma_yy"]

    gamma = float(z["gamma"])
    Mach = np.sqrt(ux**2 + uy**2) / np.sqrt(gamma * T)
    qmag = np.sqrt(qx**2 + qy**2)
    sigmag = np.sqrt(sig_xx**2 + sig_yy**2 + 2.0*sig_xy**2)

    items = [
        (rho, r"$\rho$"),
        (ux, r"$u_x$"),
        (uy, r"$u_y$"),
        (T, r"$T$"),
        (Mach, r"$M$"),
        (qmag, r"$|q|$"),
        (sig_xx, r"$\sigma_{xx}$"),
        (sig_xy, r"$\sigma_{xy}$"),
        (sigmag, r"$||\sigma||$"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.0, 11.5), constrained_layout=True)

    for ax, (A, title) in zip(axes.ravel(), items):
        im = ax.imshow(A.T, origin="lower", extent=extent, aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)

    fig.suptitle("2D BGK DVM Riemann problem")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=250)
    print(f"[plot] {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/riemann2d_fourstream_kn008_t006_dvm.npz")
    ap.add_argument("--fig", default="figures/riemann2d_fourstream_kn008_t006.png")

    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--xmin", type=float, default=-0.5)
    ap.add_argument("--xmax", type=float, default=0.5)
    ap.add_argument("--ymin", type=float, default=-0.5)
    ap.add_argument("--ymax", type=float, default=0.5)

    ap.add_argument("--nvx", type=int, default=18)
    ap.add_argument("--nvy", type=int, default=18)
    ap.add_argument("--nvz", type=int, default=10)
    ap.add_argument("--vmax", type=float, default=8.0)

    ap.add_argument("--Kn", type=float, default=0.08)
    ap.add_argument("--t-final", type=float, default=0.06)
    ap.add_argument("--cfl", type=float, default=0.35)

    ap.add_argument("--u0", type=float, default=1.6)
    ap.add_argument("--rho0", type=float, default=1.0)
    ap.add_argument("--T0", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=5.0/3.0)

    ap.add_argument("--corr-iters", type=int, default=4)
    ap.add_argument("--maxwell-chunk", type=int, default=2048)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--input", default="")

    args = ap.parse_args()

    if args.plot_only:
        inp = args.input or args.out
        plot_npz(inp, args.fig)
        return

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"[device] {device}, dtype={dtype}", flush=True)

    x = torch.linspace(args.xmin, args.xmax, args.nx, device=device, dtype=dtype)
    y = torch.linspace(args.ymin, args.ymax, args.ny, device=device, dtype=dtype)
    dx = float((x[1]-x[0]).detach().cpu())
    dy = float((y[1]-y[0]).detach().cpu())

    v, v2, w = make_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)

    rho0, ux0, uy0, T0 = initial_four_stream(x, y, args.u0, args.rho0, args.T0, device, dtype)
    f = conservative_discrete_maxwellian(
        v, v2, w, rho0, ux0, uy0, T0,
        niter=args.corr_iters,
        chunk=args.maxwell_chunk
    )
    f = f.reshape(args.nx, args.ny, -1).contiguous()

    # dt for 2D explicit upwind advection.
    dt_adv = args.cfl / (args.vmax/dx + args.vmax/dy)
    nsteps = int(math.ceil(args.t_final / dt_adv))
    dt = args.t_final / nsteps

    relax = math.exp(-dt / args.Kn)

    print(f"[grid] nx={args.nx}, ny={args.ny}, dx={dx:.4e}, dy={dy:.4e}", flush=True)
    print(f"[time] t_final={args.t_final}, nsteps={nsteps}, dt={dt:.4e}, Kn={args.Kn}, relax={relax:.6f}", flush=True)
    print(f"[case] four-stream Riemann, u0={args.u0}, rho0={args.rho0}, T0={args.T0}", flush=True)

    vxpos = (v[:, 0] >= 0)[None, None, :]
    vypos = (v[:, 1] >= 0)[None, None, :]

    running_out = str(Path(args.out).with_name(Path(args.out).stem + "_running.npz"))

    for step in range(1, nsteps+1):
        # Transmissive neighbors in x.
        fL = torch.empty_like(f)
        fR = torch.empty_like(f)
        fL[1:, :, :] = f[:-1, :, :]
        fL[0, :, :] = f[0, :, :]
        fR[:-1, :, :] = f[1:, :, :]
        fR[-1, :, :] = f[-1, :, :]

        # Transmissive neighbors in y.
        fD = torch.empty_like(f)
        fU = torch.empty_like(f)
        fD[:, 1:, :] = f[:, :-1, :]
        fD[:, 0, :] = f[:, 0, :]
        fU[:, :-1, :] = f[:, 1:, :]
        fU[:, -1, :] = f[:, -1, :]

        dfdx = torch.where(vxpos, (f - fL)/dx, (fR - f)/dx)
        dfdy = torch.where(vypos, (f - fD)/dy, (fU - f)/dy)

        fstar = f - dt * (v[None, None, :, 0] * dfdx + v[None, None, :, 1] * dfdy)
        fstar = torch.clamp(fstar, min=1e-30)

        ff = fstar.reshape(-1, fstar.shape[-1])
        rho, ux, uy, T, qx, qy, sig_xx, sig_xy, sig_yy = moments_2d(ff, v, v2, w)

        M = conservative_discrete_maxwellian(
            v, v2, w, rho, ux, uy, T,
            niter=args.corr_iters,
            chunk=args.maxwell_chunk
        )
        f = M.reshape(args.nx, args.ny, -1) + (fstar - M.reshape(args.nx, args.ny, -1)) * relax
        f = torch.clamp(f, min=1e-30)

        if step == 1 or step % args.save_every == 0 or step == nsteps:
            ff = f.reshape(-1, f.shape[-1])
            rho, ux, uy, T, qx, qy, sig_xx, sig_xy, sig_yy = moments_2d(ff, v, v2, w)

            def field(a):
                return a.detach().cpu().numpy().reshape(args.nx, args.ny)

            fields = {
                "rho": field(rho),
                "ux": field(ux),
                "uy": field(uy),
                "T": field(T),
                "qx": field(qx),
                "qy": field(qy),
                "sigma_xx": field(sig_xx),
                "sigma_xy": field(sig_xy),
                "sigma_yy": field(sig_yy),
            }

            meta = {
                "time": np.array(step * dt),
                "Kn": np.array(args.Kn),
                "gamma": np.array(args.gamma),
                "u0": np.array(args.u0),
                "rho0": np.array(args.rho0),
                "T0": np.array(args.T0),
                "nsteps": np.array(nsteps),
                "step": np.array(step),
            }

            save_npz(
                running_out,
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                fields,
                meta
            )

            print(
                f"[step] {step:05d}/{nsteps} t={step*dt:.5e} "
                f"rho[min,max]=({float(rho.min()):.4f},{float(rho.max()):.4f}) "
                f"T[min,max]=({float(T.min()):.4f},{float(T.max()):.4f}) "
                f"|q|max={float(torch.sqrt(qx*qx+qy*qy).max()):.4e} "
                f"|sig|max={float(torch.sqrt(sig_xx*sig_xx+sig_yy*sig_yy+2*sig_xy*sig_xy).max()):.4e}",
                flush=True
            )

    # Final save and plot.
    ff = f.reshape(-1, f.shape[-1])
    rho, ux, uy, T, qx, qy, sig_xx, sig_xy, sig_yy = moments_2d(ff, v, v2, w)

    def field(a):
        return a.detach().cpu().numpy().reshape(args.nx, args.ny)

    fields = {
        "rho": field(rho),
        "ux": field(ux),
        "uy": field(uy),
        "T": field(T),
        "qx": field(qx),
        "qy": field(qy),
        "sigma_xx": field(sig_xx),
        "sigma_xy": field(sig_xy),
        "sigma_yy": field(sig_yy),
    }

    meta = {
        "time": np.array(args.t_final),
        "Kn": np.array(args.Kn),
        "gamma": np.array(args.gamma),
        "u0": np.array(args.u0),
        "rho0": np.array(args.rho0),
        "T0": np.array(args.T0),
        "nsteps": np.array(nsteps),
        "step": np.array(nsteps),
    }

    save_npz(
        args.out,
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        fields,
        meta
    )

    plot_npz(args.out, args.fig)


if __name__ == "__main__":
    main()
