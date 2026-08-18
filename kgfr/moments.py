from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from .data import ShockFullState


def moments_from_model(
    model: torch.nn.Module,
    data: ShockFullState,
    ix: torch.Tensor,
    v_chunk: int = 8192,
    f_clip_log_min: float = -90.0,
    f_clip_log_max: float = 30.0,
) -> Dict[str, torch.Tensor]:
    """Differentiably compute moments of the model for selected x indices."""
    device = next(model.parameters()).device
    ix = ix.to(device=device, dtype=torch.long)
    cache = data.torch_cache(device)
    v = cache["v"]
    w = cache["w"]
    v_norm = cache["v_norm"]
    x_norm = cache["x_norm"][ix]
    nx = ix.numel()
    nv = v.shape[0]

    rho = torch.zeros(nx, device=device)
    m1 = torch.zeros(nx, 3, device=device)
    e2 = torch.zeros(nx, device=device)
    # First pass: rho, mean velocity, raw energy around zero.
    for j0 in range(0, nv, v_chunk):
        j1 = min(nv, j0 + v_chunk)
        vc = v[j0:j1]
        wc = w[j0:j1]
        zn = torch.empty((nx, j1 - j0, 4), device=device)
        zn[:, :, 0] = x_norm[:, None]
        zn[:, :, 1:] = v_norm[j0:j1][None, :, :]
        logf = model(zn.reshape(-1, 4)).reshape(nx, j1 - j0)
        f = torch.exp(torch.clamp(logf, f_clip_log_min, f_clip_log_max))
        fw = f * wc[None, :]
        rho = rho + torch.sum(fw, dim=1)
        m1 = m1 + fw @ vc
        e2 = e2 + torch.sum(fw * torch.sum(vc[None, :, :] ** 2, dim=2), dim=1)
    rho_safe = rho.clamp_min(1e-30)
    u = m1 / rho_safe[:, None]
    T = (e2 / rho_safe - torch.sum(u * u, dim=1)) / 3.0

    # Second pass: central moments.
    sxx_raw = torch.zeros(nx, device=device)
    qx = torch.zeros(nx, device=device)
    M300 = torch.zeros(nx, device=device)
    M400 = torch.zeros(nx, device=device)
    for j0 in range(0, nv, v_chunk):
        j1 = min(nv, j0 + v_chunk)
        vc = v[j0:j1]
        wc = w[j0:j1]
        zn = torch.empty((nx, j1 - j0, 4), device=device)
        zn[:, :, 0] = x_norm[:, None]
        zn[:, :, 1:] = v_norm[j0:j1][None, :, :]
        logf = model(zn.reshape(-1, 4)).reshape(nx, j1 - j0)
        f = torch.exp(torch.clamp(logf, f_clip_log_min, f_clip_log_max))
        c = vc[None, :, :] - u[:, None, :]
        c2 = torch.sum(c * c, dim=2)
        cx = c[:, :, 0]
        fw = f * wc[None, :]
        sxx_raw = sxx_raw + torch.sum(fw * cx * cx, dim=1)
        qx = qx + 0.5 * torch.sum(fw * c2 * cx, dim=1)
        M300 = M300 + torch.sum(fw * cx ** 3, dim=1)
        M400 = M400 + torch.sum(fw * cx ** 4, dim=1)
    sig = sxx_raw - rho * T
    return {
        "rho": rho,
        "ux": u[:, 0],
        "uy": u[:, 1],
        "uz": u[:, 2],
        "T": T,
        "qx": qx,
        "sig": sig,
        "sigma_xx": sig,
        "M300": M300,
        "M400": M400,
    }


def moment_loss(
    model: torch.nn.Module,
    data: ShockFullState,
    ix: torch.Tensor,
    keys: Iterable[str],
    v_chunk: int = 8192,
) -> tuple[torch.Tensor, Dict[str, float]]:
    keys = list(keys)
    pred = moments_from_model(model, data, ix, v_chunk=v_chunk)
    ref = data.reference_for_keys(keys, ix, device=next(model.parameters()).device)
    loss = torch.zeros((), device=next(model.parameters()).device)
    terms: Dict[str, float] = {}
    for k in keys:
        kk = "sig" if k == "sigma_xx" else k
        if kk not in pred or kk not in ref:
            continue
        scale = float(data.moment_scales.get(kk, 1.0))
        term = torch.mean(((pred[kk] - ref[kk]) / scale) ** 2)
        loss = loss + term
        terms[kk] = float(term.detach().cpu())
    if terms:
        loss = loss / float(len(terms))
    return loss, terms


def moments_from_fullstate_np(data: ShockFullState, x_chunk: int = 16) -> Dict[str, np.ndarray]:
    """Compute reference moments directly from stored f using NumPy chunks."""
    Nx, Nv = data.Nx, data.Nv
    v = data.v.astype(np.float64)
    w = data.w.astype(np.float64)
    out = {k: np.zeros(Nx, dtype=np.float64) for k in ["rho", "ux", "uy", "uz", "T", "qx", "sig", "M300", "M400"]}
    v2 = np.sum(v * v, axis=1)
    for i0 in range(0, Nx, x_chunk):
        i1 = min(Nx, i0 + x_chunk)
        f = data.f[i0:i1].astype(np.float64)
        fw = f * w[None, :]
        rho = np.sum(fw, axis=1)
        m1 = fw @ v
        u = m1 / np.maximum(rho[:, None], 1e-300)
        e2 = fw @ v2
        T = (e2 / np.maximum(rho, 1e-300) - np.sum(u * u, axis=1)) / 3.0
        sxx_raw = np.zeros(i1 - i0)
        qx = np.zeros(i1 - i0)
        M300 = np.zeros(i1 - i0)
        M400 = np.zeros(i1 - i0)
        # Vectorized over all velocities for this x chunk.
        c = v[None, :, :] - u[:, None, :]
        c2 = np.sum(c * c, axis=2)
        cx = c[:, :, 0]
        sxx_raw = np.sum(fw * cx * cx, axis=1)
        qx = 0.5 * np.sum(fw * c2 * cx, axis=1)
        M300 = np.sum(fw * cx ** 3, axis=1)
        M400 = np.sum(fw * cx ** 4, axis=1)
        out["rho"][i0:i1] = rho
        out["ux"][i0:i1] = u[:, 0]
        out["uy"][i0:i1] = u[:, 1]
        out["uz"][i0:i1] = u[:, 2]
        out["T"][i0:i1] = T
        out["qx"][i0:i1] = qx
        out["sig"][i0:i1] = sxx_raw - rho * T
        out["M300"][i0:i1] = M300
        out["M400"][i0:i1] = M400
    return out


def moment_loss_sampled(
    model: torch.nn.Module,
    data: ShockFullState,
    ix: torch.Tensor,
    keys: Iterable[str],
    vel_count: int = 1024,
    uniform_frac: float = 0.10,
    mass_alpha: float = 0.55,
    f_clip_log_min: float = -90.0,
    f_clip_log_max: float = 30.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable sampled moment loss.

    This is for training only. It estimates velocity quadrature using
    importance-sampled DVM velocity nodes, avoiding full-grid autograd memory.
    Full exact quadrature is still used by evaluate_phase_gaussian.py.
    """
    keys = list(keys)
    device = next(model.parameters()).device
    ix = ix.to(device=device, dtype=torch.long)
    ix_cpu = ix.detach().cpu().numpy().astype(np.int64)

    cache = data.torch_cache(device)
    v = cache["v"]
    w = cache["w"]
    v_norm = cache["v_norm"]
    x_norm = cache["x_norm"][ix]

    nx = len(ix_cpu)
    m = int(vel_count)
    if m <= 0:
        raise ValueError("vel_count must be positive")

    iv_np = np.empty((nx, m), dtype=np.int64)
    prob_np = np.empty((nx, m), dtype=np.float32)

    Nv = data.Nv
    for a, i in enumerate(ix_cpu):
        row = np.maximum(data.f[int(i)], data.f_floor).astype(np.float64)
        mass = row * data.w.astype(np.float64)
        if mass_alpha != 1.0:
            mass = np.power(mass + 1.0e-300, float(mass_alpha))
        ssum = float(np.sum(mass))
        if (not np.isfinite(ssum)) or ssum <= 0.0:
            p = np.full(Nv, 1.0 / Nv, dtype=np.float64)
        else:
            p = mass / ssum
            uf = float(uniform_frac)
            if uf > 0.0:
                p = (1.0 - uf) * p + uf / Nv
            p = np.maximum(p, 1.0e-300)
            p = p / np.sum(p)

        iv = np.random.choice(Nv, size=m, replace=True, p=p)
        iv_np[a, :] = iv
        prob_np[a, :] = p[iv].astype(np.float32)

    iv = torch.from_numpy(iv_np).to(device=device, dtype=torch.long)
    prob = torch.from_numpy(prob_np).to(device=device)

    v_s = v[iv]                  # (nx,m,3), physical velocity
    w_s = w[iv]                  # (nx,m)
    vn_s = v_norm[iv]            # (nx,m,3)

    z = torch.empty((nx, m, 4), device=device)
    z[:, :, 0] = x_norm[:, None]
    z[:, :, 1:] = vn_s

    logf = model(z.reshape(-1, 4)).reshape(nx, m)
    fhat = torch.exp(torch.clamp(logf, f_clip_log_min, f_clip_log_max))

    # Importance quadrature coefficient:
    # sum_v f(v) w_v phi(v) ~= mean_samples f(v_s) w_s phi(v_s) / p(v_s)
    coef = w_s / (float(m) * prob.clamp_min(1.0e-30))
    fw = fhat * coef

    rho = torch.sum(fw, dim=1).clamp_min(1.0e-30)
    m1 = torch.sum(fw[:, :, None] * v_s, dim=1)
    u = m1 / rho[:, None]

    v2 = torch.sum(v_s * v_s, dim=2)
    e2 = torch.sum(fw * v2, dim=1)
    T = (e2 / rho - torch.sum(u * u, dim=1)) / 3.0

    c = v_s - u[:, None, :]
    c2 = torch.sum(c * c, dim=2)
    cx = c[:, :, 0]

    sxx_raw = torch.sum(fw * cx * cx, dim=1)
    qx = 0.5 * torch.sum(fw * c2 * cx, dim=1)
    M300 = torch.sum(fw * cx ** 3, dim=1)
    M400 = torch.sum(fw * cx ** 4, dim=1)

    pred = {
        "rho": rho,
        "ux": u[:, 0],
        "uy": u[:, 1],
        "uz": u[:, 2],
        "T": T,
        "qx": qx,
        "sig": sxx_raw - rho * T,
        "sigma_xx": sxx_raw - rho * T,
        "M300": M300,
        "M400": M400,
    }

    ref = data.reference_for_keys(keys, ix, device=device)
    loss = torch.zeros((), device=device)
    terms: Dict[str, float] = {}

    for k in keys:
        kk = "sig" if k == "sigma_xx" else k
        if kk not in pred or kk not in ref:
            continue
        scale = float(data.moment_scales.get(kk, 1.0))
        term = torch.mean(((pred[kk] - ref[kk]) / scale) ** 2)
        loss = loss + term
        terms[kk] = float(term.detach().cpu())

    if terms:
        loss = loss / float(len(terms))

    return loss, terms
