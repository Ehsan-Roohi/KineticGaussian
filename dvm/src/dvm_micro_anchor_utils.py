import math
import numpy as np

def _to_np(a):
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def _get(d, names):
    for n in names:
        if n in d:
            return n, d[n]
    return None, None

def _get_velocity(d):
    kvx, vx = _get(d, ["vx", "cx", "v_x", "Vx"])
    kvy, vy = _get(d, ["vy", "cy", "v_y", "Vy"])
    kvz, vz = _get(d, ["vz", "cz", "v_z", "Vz"])
    if vx is not None and vy is not None and vz is not None:
        return kvx, _to_np(vx).reshape(-1), _to_np(vy).reshape(-1), _to_np(vz).reshape(-1)

    kv, v = _get(d, ["v", "vel", "vels", "V", "c"])
    if v is None:
        return None, None, None, None
    arr = _to_np(v)
    if arr.ndim != 2:
        return None, None, None, None
    if arr.shape[1] == 3:
        return kv, arr[:,0].reshape(-1), arr[:,1].reshape(-1), arr[:,2].reshape(-1)
    if arr.shape[0] == 3:
        return kv, arr[0].reshape(-1), arr[1].reshape(-1), arr[2].reshape(-1)
    return None, None, None, None

def _maxwellian(vx, vy, vz, rho, ux, T):
    c2 = (vx - ux)**2 + vy**2 + vz**2
    return rho / ((2.0 * math.pi * T)**1.5) * np.exp(-c2 / (2.0 * T))

def save_micro_anchors_from_locals(
    local_vars,
    out="ref/standing_M2_x40_hmom_dvm_microanchors.npz",
    x_targets=(-12.0, -6.0, -2.0, 0.0, 2.0, 6.0, 12.0),
    n_micro_per_x=256,
):
    kF, F = _get(local_vars, ["F", "f", "Fn", "dist", "distribution"])
    kx, x = _get(local_vars, ["x", "x_scaled", "xc", "xcell", "x_grid", "x_centers"])
    kxm, x_mfp = _get(local_vars, ["x_mfp", "xmfp", "x_scaled"])
    krho, rho = _get(local_vars, ["rho", "rho_np"])
    kux, ux = _get(local_vars, ["ux", "u", "u_x", "ux_np"])
    kT, T = _get(local_vars, ["T", "temp", "temperature", "T_np"])
    kv, vx, vy, vz = _get_velocity(local_vars)
    kw, w = _get(local_vars, ["w", "ww", "weights", "weight"])

    missing = []
    for label, key in [
        ("F/f", kF), ("x/x_scaled", kx), ("rho", krho), ("ux/u", kux),
        ("T", kT), ("velocity grid", kv)
    ]:
        if key is None:
            missing.append(label)

    if missing:
        print("[micro-anchor] missing variables:", missing)
        print("[micro-anchor] available keys sample:", sorted(list(local_vars.keys()))[:160])
        raise RuntimeError("Could not infer all DVM variables for micro-anchor saving.")

    F = _to_np(F)
    x = _to_np(x).reshape(-1)
    rho = _to_np(rho).reshape(-1)
    ux = _to_np(ux).reshape(-1)
    T = _to_np(T).reshape(-1)

    if x_mfp is None:
        x_mfp = x.copy()
    else:
        x_mfp = _to_np(x_mfp).reshape(-1)

    vx = np.asarray(vx).reshape(-1)
    vy = np.asarray(vy).reshape(-1)
    vz = np.asarray(vz).reshape(-1)

    if w is None:
        w = np.ones_like(vx)
    else:
        w = _to_np(w).reshape(-1)

    if F.ndim != 2:
        raise RuntimeError(f"Expected F/f to be 2D, got {F.shape}")

    if F.shape[0] == x.size and F.shape[1] == vx.size:
        pass
    elif F.shape[1] == x.size and F.shape[0] == vx.size:
        F = F.T
    else:
        raise RuntimeError(f"F shape {F.shape} incompatible with Nx={x.size}, Nv={vx.size}")

    ix = np.array([int(np.argmin(np.abs(x_mfp - xt))) for xt in x_targets], dtype=np.int64)
    ix = np.unique(ix)

    yz2 = vy**2 + vz**2
    ymin = float(np.min(yz2))
    iv_line = np.where(yz2 <= ymin + 1e-14)[0]
    if iv_line.size < 8:
        iv_line = np.argsort(yz2)[:min(256, vx.size)]
    iv_line = iv_line[np.argsort(vx[iv_line])]

    f_line = F[ix[:, None], iv_line[None, :]]

    ix_micro = []
    iv_micro = []
    x_micro = []
    f_micro = []
    M_micro = []
    logfM_micro = []
    score_micro = []

    for i in ix:
        rhoi = float(rho[i])
        uxi = float(ux[i])
        Ti = float(max(T[i], 1e-14))
        sqT = math.sqrt(Ti)

        cx = (vx - uxi) / sqT
        cy = vy / sqT
        cz = vz / sqT
        c2 = cx**2 + cy**2 + cz**2

        Hm = cx**3 - (3.0/5.0) * cx * c2
        HR = (c2 - 7.0) * (cx**2 - c2/3.0)

        M = np.maximum(_maxwellian(vx, vy, vz, rhoi, uxi, Ti), 1e-300)
        Fi = np.maximum(F[i, :], 1e-300)

        tail_gate = (c2 > 1.0) & (c2 < 32.0)
        score = (np.abs(HR) + 0.35*np.abs(Hm)) * np.sqrt(M)
        score = np.where(tail_gate, score, 0.0)

        nsel = min(int(n_micro_per_x), vx.size)
        iv = np.argpartition(score, -nsel)[-nsel:]
        iv = iv[np.argsort(score[iv])[::-1]]

        ix_micro.append(np.full(iv.size, i, dtype=np.int64))
        iv_micro.append(iv.astype(np.int64))
        x_micro.append(np.full(iv.size, x[i]))
        f_micro.append(Fi[iv])
        M_micro.append(M[iv])
        logfM_micro.append(np.log(Fi[iv]/M[iv]))
        score_micro.append(score[iv])

    iv_all = np.concatenate(iv_micro)

    np.savez(
        out,
        x=x,
        x_mfp=x_mfp,
        ix=ix,
        x_anchor=x[ix],
        x_mfp_anchor=x_mfp[ix],
        rho_anchor=rho[ix],
        ux_anchor=ux[ix],
        T_anchor=T[ix],
        vx=vx,
        vy=vy,
        vz=vz,
        w=w,
        iv_line=iv_line,
        vx_line=vx[iv_line],
        vy_line=vy[iv_line],
        vz_line=vz[iv_line],
        f_line=f_line,
        ix_micro=np.concatenate(ix_micro),
        iv_micro=iv_all,
        x_micro=np.concatenate(x_micro),
        vx_micro=vx[iv_all],
        vy_micro=vy[iv_all],
        vz_micro=vz[iv_all],
        f_micro=np.concatenate(f_micro),
        M_micro=np.concatenate(M_micro),
        logfM_micro=np.concatenate(logfM_micro),
        score_micro=np.concatenate(score_micro),
    )

    print(f"[micro-anchor saved] {out}", flush=True)
    print(f"[micro-anchor] vars: F={kF}, x={kx}, x_mfp={kxm}, v={kv}, w={kw}", flush=True)
    print(f"[micro-anchor] x_mfp anchors: {x_mfp[ix]}", flush=True)
    print(f"[micro-anchor] f_line shape: {f_line.shape}", flush=True)
    print(f"[micro-anchor] micro points: {iv_all.size}", flush=True)
