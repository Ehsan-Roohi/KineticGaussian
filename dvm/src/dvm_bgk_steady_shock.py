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
    vmax = float(vc.get("vmax", 10.0))
    nvx, nvy, nvz = int(vc.get("nvx",40)), int(vc.get("nvy",12)), int(vc.get("nvz",12))
    vx = torch.linspace(-vmax, vmax, nvx, device=device, dtype=dtype)
    vy = torch.linspace(-vmax, vmax, nvy, device=device, dtype=dtype)
    vz = torch.linspace(-vmax, vmax, nvz, device=device, dtype=dtype)
    dvx = 2*vmax/(nvx-1)
    dvy = 2*vmax/(nvy-1)
    dvz = 2*vmax/(nvz-1)
    X,Y,Z = torch.meshgrid(vx,vy,vz, indexing="ij")
    v = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    w = torch.full((v.shape[0],), dvx*dvy*dvz, device=device, dtype=dtype)
    print(f"[steady-dvm] Nv={v.shape[0]} = {nvx}x{nvy}x{nvz}, vmax={vmax}")
    return v, w

def maxwellian_2d(rho, ux, T, v):
    vx, vy, vz = v[:,0], v[:,1], v[:,2]
    rho = torch.clamp(rho, min=1e-12)
    T = torch.clamp(T, min=1e-8)
    c2 = (vx[None,:]-ux)**2 + vy[None,:]**2 + vz[None,:]**2
    return rho / torch.pow(2.0*math.pi*T, 1.5) * torch.exp(-0.5*c2/T)

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
    ap.add_argument("--nx", type=int, default=500)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--omega", type=float, default=0.45)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_json(args.config)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = get_device(args.device)
    print(f"[steady-dvm] device={device}, dtype={dtype}")

    c = cfg["case"]
    xmin, xmax = float(c["x_min"]), float(c["x_max"])
    kn = float(c["kn"])
    nx = args.nx
    dx = (xmax-xmin)/nx
    x = torch.linspace(xmin+0.5*dx, xmax-0.5*dx, nx, device=device, dtype=dtype)
    v, w = build_vgrid(cfg, device, dtype)
    vx = v[:,0]
    pos = vx > 0
    neg = vx < 0

    L, R = c["left"], c["right"]
    rhoL = torch.tensor([[float(L["rho"])]], device=device, dtype=dtype)
    uxL  = torch.tensor([[float(L["ux"])]],  device=device, dtype=dtype)
    TL   = torch.tensor([[float(L["T"])]],   device=device, dtype=dtype)
    rhoR = torch.tensor([[float(R["rho"])]], device=device, dtype=dtype)
    uxR  = torch.tensor([[float(R["ux"])]],  device=device, dtype=dtype)
    TR   = torch.tensor([[float(R["T"])]],   device=device, dtype=dtype)

    ML = maxwellian_2d(rhoL, uxL, TL, v)[0]
    MR = maxwellian_2d(rhoR, uxR, TR, v)[0]

    # tanh initial profile
    k = float(c.get("smooth_k", 12.0))
    H = 0.5*(1.0 + torch.tanh(k*x))
    rho0 = float(L["rho"])*(1-H) + float(R["rho"])*H
    ux0  = float(L["ux"])*(1-H)  + float(R["ux"])*H
    T0   = float(L["T"])*(1-H)   + float(R["T"])*H
    f = maxwellian_2d(rho0[:,None], ux0[:,None], T0[:,None], v)

    apv = torch.clamp(vx[pos]*kn/dx, min=1e-12)
    anv = torch.clamp((-vx[neg])*kn/dx, min=1e-12)

    last_res = None
    with torch.no_grad():
        for it in range(1, args.iters+1):
            rho, ux, T, qx, sig = moments(f, v, w)
            M = maxwellian_2d(rho[:,None], ux[:,None], T[:,None], v)
            fnew = torch.empty_like(f)

            # positive velocities: sweep left -> right
            fp = fnew[:,pos]
            Mp = M[:,pos]
            left = ML[pos]
            fp[0,:] = (apv*left + Mp[0,:])/(apv+1.0)
            for i in range(1, nx):
                fp[i,:] = (apv*fp[i-1,:] + Mp[i,:])/(apv+1.0)
            fnew[:,pos] = fp

            # negative velocities: sweep right -> left
            fn = fnew[:,neg]
            Mn = M[:,neg]
            right = MR[neg]
            fn[-1,:] = (anv*right + Mn[-1,:])/(anv+1.0)
            for i in range(nx-2, -1, -1):
                fn[i,:] = (anv*fn[i+1,:] + Mn[i,:])/(anv+1.0)
            fnew[:,neg] = fn

            # if any exact-zero vx exists, set to local M
            zero = ~(pos | neg)
            if torch.any(zero):
                fnew[:,zero] = M[:,zero]

            res = torch.max(torch.abs(fnew-f)/(torch.abs(f)+1e-12)).item()
            f = torch.clamp((1-args.omega)*f + args.omega*fnew, min=1e-45)

            if args.progress_every and (it == 1 or it % args.progress_every == 0 or it == args.iters):
                rho, ux, T, qx, sig = moments(f, v, w)
                print(f"[steady-dvm] iter {it}/{args.iters} rel_change={res:.3e} "
                      f"rho_mid={rho[nx//2].item():.4e} ux_mid={ux[nx//2].item():.4e} T_mid={T[nx//2].item():.4e}",
                      flush=True)
            last_res = res

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
        steady_res=np.array(last_res),
        M1=np.array(c.get("M1", 2.0))
    )
    print(f"[done] saved {out}")

if __name__ == "__main__":
    main()
