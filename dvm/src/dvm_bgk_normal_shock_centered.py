#!/usr/bin/env python
import argparse
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normal_shock_states(M1=2.0, gamma=5/3, rho1=1.0, T1=1.0):
    a1 = math.sqrt(gamma*T1)
    u1 = M1*a1
    r = ((gamma+1)*M1*M1)/((gamma-1)*M1*M1 + 2.0)
    p2_p1 = 1.0 + 2.0*gamma/(gamma+1.0)*(M1*M1 - 1.0)
    rho2 = rho1*r
    T2 = T1*p2_p1/r
    u2 = u1/r
    return dict(rho1=rho1, u1=u1, T1=T1, rho2=rho2, u2=u2, T2=T2, gamma=gamma, M1=M1)


def make_vgrid(nvx, nvy, nvz, vmax, device, dtype):
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)

    def trap_w(n, h, device, dtype):
        w = torch.ones(n, device=device, dtype=dtype)*h
        w[0] *= 0.5
        w[-1] *= 0.5
        return w

    wx = trap_w(nvx, 2*vmax/(nvx-1), device, dtype)
    wy = trap_w(nvy, 2*vmax/(nvy-1), device, dtype)
    wz = trap_w(nvz, 2*vmax/(nvz-1), device, dtype)

    VX, VY, VZ = torch.meshgrid(vx, vy, vz, indexing="ij")
    W = wx[:,None,None]*wy[None,:,None]*wz[None,None,:]
    v = torch.stack([VX.reshape(-1), VY.reshape(-1), VZ.reshape(-1)], dim=1)
    w = W.reshape(-1)
    return v, w


def maxwellian(v, rho, u, T):
    c2 = (v[:,0]-u)**2 + v[:,1]**2 + v[:,2]**2
    return rho/((2*math.pi*T)**1.5) * torch.exp(-c2/(2*T))


def moments(f, v, w):
    # f shape: [nx, nv]
    rho = torch.sum(f*w[None,:], dim=1)
    momx = torch.sum(f*v[None,:,0]*w[None,:], dim=1)
    ux = momx/(rho + 1e-30)

    e2 = torch.sum(f*(v[None,:,0]**2 + v[None,:,1]**2 + v[None,:,2]**2)*w[None,:], dim=1)
    T = (e2/(rho+1e-30) - ux**2)/3.0
    T = torch.clamp(T, min=1e-8)

    cx = v[None,:,0] - ux[:,None]
    cy = v[None,:,1]
    cz = v[None,:,2]
    c2 = cx**2 + cy**2 + cz**2

    qx = 0.5*torch.sum(c2*cx*f*w[None,:], dim=1)
    sig = torch.sum((cx**2 - c2/3.0)*f*w[None,:], dim=1)
    return rho, ux, T, qx, sig


def maxwellian_field(v, rho, ux, T):
    c2 = (v[None,:,0]-ux[:,None])**2 + v[None,:,1]**2 + v[None,:,2]**2
    pref = rho[:,None]/((2*math.pi*T[:,None])**1.5)
    return pref*torch.exp(-c2/(2*T[:,None]))


def find_crossing(x, rho, rho_mid):
    xr = x.detach().cpu().numpy()
    rr = rho.detach().cpu().numpy()
    y = rr - rho_mid
    idx = np.where(y[:-1]*y[1:] <= 0)[0]
    if len(idx) == 0:
        return 0.0
    # choose crossing closest to center
    k = idx[np.argmin(np.abs(xr[idx]))]
    x0, x1 = xr[k], xr[k+1]
    y0, y1 = y[k], y[k+1]
    if abs(y1-y0) < 1e-14:
        return float(x0)
    return float(x0 - y0*(x1-x0)/(y1-y0))


def recenter_f(f, x, shift, ML, MR):
    # New f'(x)=f_old(x+shift), so crossing at old shift moves to new x=0.
    nx, nv = f.shape
    dx = x[1] - x[0]
    xp = x + shift
    s = (xp - x[0])/dx
    i0 = torch.floor(s).long()
    a = (s - i0.to(s.dtype)).clamp(0, 1)

    fnew = torch.empty_like(f)
    left = i0 < 0
    right = i0 >= nx-1
    mid = ~(left | right)

    if torch.any(mid):
        ii = i0[mid]
        aa = a[mid][:,None]
        fnew[mid] = (1-aa)*f[ii] + aa*f[ii+1]
    if torch.any(left):
        fnew[left] = ML[None,:]
    if torch.any(right):
        fnew[right] = MR[None,:]
    return fnew


def save_npz(path, x_scaled, x_mfp, rho, ux, T, qx, sig, states, xhalf_mfp):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        x=x_scaled,
        x_mfp=x_mfp,
        rho=rho,
        ux=ux,
        T=T,
        qx=qx,
        sigma_xx=sig,
        sig=sig,
        states=np.array([states["rho1"], states["u1"], states["T1"], states["rho2"], states["u2"], states["T2"]]),
        xhalf_mfp=np.array(xhalf_mfp),
        kn_eff=np.array(1.0/(2*xhalf_mfp)),
    )
    print(f"[saved] {path}")


def plot_npz(npz_path, out_png, title):
    z = np.load(npz_path)
    x = z["x"]
    vars_ = [("rho", r"$\rho$"), ("ux", r"$u_x$"), ("T", r"$T$"), ("qx", r"$q_x$"), ("sigma_xx", r"$\sigma_{xx}$")]
    fig, axes = plt.subplots(5, 1, figsize=(9.0, 10.5), sharex=True)
    for ax, (k, lab) in zip(axes, vars_):
        ax.plot(x, z[k], lw=2.2)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"scaled shock coordinate $x/(2L_\lambda)$")
    fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    print(f"[plot] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ref/standing_M2_centered_dvm.npz")
    ap.add_argument("--fig", default="figures/standing_M2_centered_dvm_profiles.png")
    ap.add_argument("--M1", type=float, default=2.0)
    ap.add_argument("--gamma", type=float, default=5/3)
    ap.add_argument("--rho1", type=float, default=1.0)
    ap.add_argument("--T1", type=float, default=1.0)
    ap.add_argument("--xhalf-mfp", type=float, default=25.0)
    ap.add_argument("--nx", type=int, default=1400)
    ap.add_argument("--nvx", type=int, default=48)
    ap.add_argument("--nvy", type=int, default=14)
    ap.add_argument("--nvz", type=int, default=14)
    ap.add_argument("--vmax", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--cfl", type=float, default=0.35)
    ap.add_argument("--center-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"[device] {device}, dtype={dtype}")

    st = normal_shock_states(args.M1, args.gamma, args.rho1, args.T1)
    print("[states]", st)

    v, w = make_vgrid(args.nvx, args.nvy, args.nvz, args.vmax, device, dtype)
    nv = v.shape[0]
    print(f"[vgrid] Nv={nv} = {args.nvx}x{args.nvy}x{args.nvz}, vmax={args.vmax}")

    x = torch.linspace(-args.xhalf_mfp, args.xhalf_mfp, args.nx, device=device, dtype=dtype)
    dx = float((x[1]-x[0]).detach().cpu())
    dt = args.cfl*dx/args.vmax
    print(f"[x] [-{args.xhalf_mfp},{args.xhalf_mfp}], nx={args.nx}, dx={dx:.4e}, dt={dt:.4e}")

    ML = maxwellian(v, st["rho1"], st["u1"], st["T1"])
    MR = maxwellian(v, st["rho2"], st["u2"], st["T2"])

    # Initial tanh shock in mean-free-path coordinates.
    width = 3.0
    H = 0.5*(1 + torch.tanh(x/width))
    rho0 = (1-H)*st["rho1"] + H*st["rho2"]
    ux0 = (1-H)*st["u1"] + H*st["u2"]
    T0 = (1-H)*st["T1"] + H*st["T2"]
    f = maxwellian_field(v, rho0, ux0, T0)
    f = torch.clamp(f, min=1e-30)

    pos = v[:,0] > 0
    neg = v[:,0] < 0
    rho_mid = 0.5*(st["rho1"] + st["rho2"])

    last_rho = None
    for step in range(1, args.steps+1):
        f_left = torch.cat([ML[None,:], f[:-1]], dim=0)
        f_right = torch.cat([f[1:], MR[None,:]], dim=0)

        df_pos = (f - f_left)/dx
        df_neg = (f_right - f)/dx
        df = torch.where((v[:,0] >= 0)[None,:], df_pos, df_neg)

        fstar = f - dt*v[None,:,0]*df
        fstar[0, pos] = ML[pos]
        fstar[-1, neg] = MR[neg]
        fstar = torch.clamp(fstar, min=1e-30)

        rho, ux, T, qx, sig = moments(fstar, v, w)
        M = maxwellian_field(v, rho, ux, T)
        f = M + (fstar - M)*math.exp(-dt)  # x is in upstream mean-free-path units; tau=1
        f = torch.clamp(f, min=1e-30)
        f[0, pos] = ML[pos]
        f[-1, neg] = MR[neg]

        if step % args.center_every == 0:
            rho, ux, T, qx, sig = moments(f, v, w)
            xs = find_crossing(x, rho, rho_mid)
            if abs(xs) > 1e-8:
                f = recenter_f(f, x, xs, ML, MR)

        if step == 1 or step % args.save_every == 0 or step == args.steps:
            rho, ux, T, qx, sig = moments(f, v, w)
            xs = find_crossing(x, rho, rho_mid)
            change = 0.0
            if last_rho is not None:
                change = float(torch.max(torch.abs(rho-last_rho)).detach().cpu())
            last_rho = rho.detach().clone()
            print(
                f"[step] {step:8d}/{args.steps} center={xs:+.4e} "
                f"rhoL={float(rho[0]):.5f} rhoR={float(rho[-1]):.5f} "
                f"uL={float(ux[0]):.5f} uR={float(ux[-1]):.5f} "
                f"T_L={float(T[0]):.5f} T_R={float(T[-1]):.5f} "
                f"maxdrho={change:.3e}",
                flush=True
            )

    rho, ux, T, qx, sig = moments(f, v, w)
    xs = find_crossing(x, rho, rho_mid)
    f = recenter_f(f, x, xs, ML, MR)
    rho, ux, T, qx, sig = moments(f, v, w)

    x_mfp = x.detach().cpu().numpy()
    x_scaled = x_mfp/(2*args.xhalf_mfp)

    save_npz(
        args.out,
        x_scaled,
        x_mfp,
        rho.detach().cpu().numpy(),
        ux.detach().cpu().numpy(),
        T.detach().cpu().numpy(),
        qx.detach().cpu().numpy(),
        sig.detach().cpu().numpy(),
        st,
        args.xhalf_mfp
    )
    plot_npz(args.out, args.fig, f"Centered BGK normal shock DVM, M1={args.M1}, gamma={args.gamma:.3g}, Kn_eff={1/(2*args.xhalf_mfp):.4f}")


if __name__ == "__main__":
    main()
