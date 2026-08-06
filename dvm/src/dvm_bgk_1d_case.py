#!/usr/bin/env python
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

def load_json(p):
    return json.loads(Path(p).read_text())

def get_device(dev):
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def build_vgrid(cfg, device, dtype):
    vc = cfg["velocity"]
    vmax = float(vc.get("vmax", 7.0))
    nvx, nvy, nvz = int(vc.get("nvx",32)), int(vc.get("nvy",12)), int(vc.get("nvz",12))
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)
    dvx = 2*vmax/(nvx-1)
    dvy = 2*vmax/(nvy-1)
    dvz = 2*vmax/(nvz-1)
    X,Y,Z = torch.meshgrid(vx,vy,vz, indexing="ij")
    v = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    w = torch.full((v.shape[0],), dvx*dvy*dvz, device=device, dtype=dtype)
    print(f"[dvm] Nv={v.shape[0]} = {nvx}x{nvy}x{nvz}, vmax={vmax}")
    return v, w

def maxwellian(rho, ux, T, v):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    rho = torch.clamp(rho, min=1e-12)
    T = torch.clamp(T, min=1e-8)
    c2 = (vx[None,:]-ux)**2 + vy[None,:]**2 + vz[None,:]**2
    return rho / torch.pow(2.0*math.pi*T, 1.5) * torch.exp(-0.5*c2/T)

def case_state(cfg, x):
    c = cfg["case"]
    L, R = c["left"], c["right"]
    k = float(c.get("smooth_k", 20.0))
    H = 0.5 * (1.0 + torch.tanh(k*x))
    rho = float(L["rho"])*(1-H) + float(R["rho"])*H
    ux  = float(L.get("ux",0.0))*(1-H) + float(R.get("ux",0.0))*H
    T   = float(L["T"])*(1-H) + float(R["T"])*H
    return rho[:,None], ux[:,None], T[:,None]

def moments(f, v, w):
    vx, vy, vz = v[:,0][None,:], v[:,1][None,:], v[:,2][None,:]
    ww = w[None,:]
    rho = torch.sum(f*ww, dim=1)
    ux = torch.sum(f*vx*ww, dim=1) / torch.clamp(rho, min=1e-12)
    cx = vx - ux[:,None]
    c2 = cx*cx + vy*vy + vz*vz
    T = torch.sum(f*c2*ww, dim=1) / torch.clamp(3*rho, min=1e-12)
    qx = 0.5 * torch.sum(f*c2*cx*ww, dim=1)
    sig = torch.sum(f*(cx*cx - c2/3.0)*ww, dim=1)
    return rho, ux, T, qx, sig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nx", type=int, default=600)
    ap.add_argument("--cfl", type=float, default=0.42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_json(args.config)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = get_device(args.device)
    print(f"[dvm] device={device}, dtype={dtype}")

    c = cfg["case"]
    xmin, xmax = float(c["x_min"]), float(c["x_max"])
    tf = float(c["t_final"])
    kn = float(c["kn"])
    nx = args.nx
    dx = (xmax-xmin)/nx
    x = torch.linspace(xmin+0.5*dx, xmax-0.5*dx, nx, device=device, dtype=dtype)
    v, w = build_vgrid(cfg, device, dtype)
    vx = v[:,0]

    rho0, ux0, T0 = case_state(cfg, x)
    f = maxwellian(rho0, ux0, T0, v)

    Ls, Rs = c["left"], c["right"]
    ML = maxwellian(
        torch.tensor([[float(Ls["rho"])]], device=device, dtype=dtype),
        torch.tensor([[float(Ls.get("ux",0.0))]], device=device, dtype=dtype),
        torch.tensor([[float(Ls["T"])]], device=device, dtype=dtype),
        v
    )[0]
    MR = maxwellian(
        torch.tensor([[float(Rs["rho"])]], device=device, dtype=dtype),
        torch.tensor([[float(Rs.get("ux",0.0))]], device=device, dtype=dtype),
        torch.tensor([[float(Rs["T"])]], device=device, dtype=dtype),
        v
    )[0]

    vmax = float(cfg["velocity"]["vmax"])
    dt = args.cfl * dx / vmax
    nsteps = max(1, int(math.ceil(tf/dt)))
    dt = tf/nsteps
    relax = math.exp(-dt/kn)
    print(f"[dvm] nx={nx}, dx={dx:.4e}, dt={dt:.4e}, nsteps={nsteps}, Kn={kn}, t_final={tf}")

    with torch.no_grad():
        for n in range(1, nsteps+1):
            fim1 = torch.empty_like(f)
            fip1 = torch.empty_like(f)
            fim1[1:,:] = f[:-1,:]
            fip1[:-1,:] = f[1:,:]
            fim1[0,:] = torch.where(vx > 0, ML, f[0,:])
            fip1[-1,:] = torch.where(vx < 0, MR, f[-1,:])

            dfdx_pos = (f - fim1)/dx
            dfdx_neg = (fip1 - f)/dx
            dfdx = torch.where(vx[None,:] >= 0, dfdx_pos, dfdx_neg)
            fstar = torch.clamp(f - dt*vx[None,:]*dfdx, min=1e-45)

            rho, ux, T, _, _ = moments(fstar, v, w)
            M = maxwellian(rho[:,None], ux[:,None], T[:,None], v)
            f = torch.clamp(M + (fstar - M)*relax, min=1e-45)

            if args.progress_every and (n == 1 or n % args.progress_every == 0 or n == nsteps):
                print(f"[dvm] step {n}/{nsteps}", flush=True)

        rho, ux, T, qx, sig = moments(f, v, w)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        x=x.detach().cpu().numpy(),
        rho=rho.detach().cpu().numpy(),
        ux=ux.detach().cpu().numpy(),
        T=T.detach().cpu().numpy(),
        qx=qx.detach().cpu().numpy(),
        sigma_xx=sig.detach().cpu().numpy(),
        kn=np.array(kn),
        t_final=np.array(tf),
        nx=np.array(nx)
    )
    print(f"[done] saved {out}")

if __name__ == "__main__":
    main()
