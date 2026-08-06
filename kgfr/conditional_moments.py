from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from .data import ShockFullState


def moments_from_conditional_model(
    model: torch.nn.Module,
    mach_norm: float | torch.Tensor,
    data: ShockFullState,
    ix: torch.Tensor,
    v_chunk: int = 2048,
    f_clip_log_min: float = -90.0,
    f_clip_log_max: float = 30.0,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    ix = ix.to(device=device, dtype=torch.long)
    cache = data.torch_cache(device)
    v, w = cache["v"], cache["w"]
    v_norm, x_norm = cache["v_norm"], cache["x_norm"][ix]
    nx, nv = ix.numel(), v.shape[0]
    mach = torch.as_tensor(mach_norm, dtype=v.dtype, device=device)

    rho = torch.zeros(nx, device=device)
    m1 = torch.zeros(nx, 3, device=device)
    e2 = torch.zeros(nx, device=device)
    for j0 in range(0, nv, v_chunk):
        j1 = min(nv, j0 + v_chunk)
        vc, wc = v[j0:j1], w[j0:j1]
        z = torch.empty((nx, j1 - j0, 4), device=device)
        z[:, :, 0] = x_norm[:, None]
        z[:, :, 1:] = v_norm[j0:j1][None, :, :]
        logf = model(mach, z.reshape(-1, 4)).reshape(nx, j1 - j0)
        f = torch.exp(torch.clamp(logf, f_clip_log_min, f_clip_log_max))
        fw = f * wc[None, :]
        rho += torch.sum(fw, dim=1)
        m1 += fw @ vc
        e2 += torch.sum(fw * torch.sum(vc[None, :, :] ** 2, dim=2), dim=1)
    rho_safe = rho.clamp_min(1.0e-30)
    u = m1 / rho_safe[:, None]
    temperature = (e2 / rho_safe - torch.sum(u * u, dim=1)) / 3.0

    sxx_raw = torch.zeros(nx, device=device)
    qx = torch.zeros(nx, device=device)
    m300 = torch.zeros(nx, device=device)
    m400 = torch.zeros(nx, device=device)
    for j0 in range(0, nv, v_chunk):
        j1 = min(nv, j0 + v_chunk)
        vc, wc = v[j0:j1], w[j0:j1]
        z = torch.empty((nx, j1 - j0, 4), device=device)
        z[:, :, 0] = x_norm[:, None]
        z[:, :, 1:] = v_norm[j0:j1][None, :, :]
        logf = model(mach, z.reshape(-1, 4)).reshape(nx, j1 - j0)
        f = torch.exp(torch.clamp(logf, f_clip_log_min, f_clip_log_max))
        c = vc[None, :, :] - u[:, None, :]
        c2 = torch.sum(c * c, dim=2)
        cx = c[:, :, 0]
        fw = f * wc[None, :]
        sxx_raw += torch.sum(fw * cx * cx, dim=1)
        qx += 0.5 * torch.sum(fw * c2 * cx, dim=1)
        m300 += torch.sum(fw * cx**3, dim=1)
        m400 += torch.sum(fw * cx**4, dim=1)
    sig = sxx_raw - rho * temperature
    m400neq = m400 - 3.0 * rho * temperature**2
    return {
        "rho": rho,
        "ux": u[:, 0],
        "uy": u[:, 1],
        "uz": u[:, 2],
        "T": temperature,
        "qx": qx,
        "sig": sig,
        "sigma_xx": sig,
        "M300": m300,
        "M400": m400,
        "M400neq": m400neq,
    }


def sampled_conditional_moment_loss(
    model: torch.nn.Module,
    mach_norm: float | torch.Tensor,
    data: ShockFullState,
    ix: torch.Tensor,
    keys: Iterable[str],
    vel_count: int = 1024,
    uniform_frac: float = 0.10,
    mass_alpha: float = 0.55,
) -> tuple[torch.Tensor, Dict[str, float]]:
    keys = list(keys)
    device = next(model.parameters()).device
    ix = ix.to(device=device, dtype=torch.long)
    ix_cpu = ix.detach().cpu().numpy().astype(np.int64)
    cache = data.torch_cache(device)
    v, w = cache["v"], cache["w"]
    v_norm, x_norm = cache["v_norm"], cache["x_norm"][ix]
    nx, sample_count, nv = len(ix_cpu), int(vel_count), data.Nv
    if sample_count <= 0:
        raise ValueError("vel_count must be positive")

    iv_np = np.empty((nx, sample_count), dtype=np.int64)
    prob_np = np.empty((nx, sample_count), dtype=np.float32)
    for row_index, data_index in enumerate(ix_cpu):
        row = np.maximum(data.f[int(data_index)], data.f_floor).astype(np.float64)
        mass = row * data.w.astype(np.float64)
        if mass_alpha != 1.0:
            mass = np.power(mass + 1.0e-300, float(mass_alpha))
        if not np.isfinite(mass.sum()) or mass.sum() <= 0.0:
            prob = np.full(nv, 1.0 / nv, dtype=np.float64)
        else:
            prob = mass / mass.sum()
            prob = (1.0 - uniform_frac) * prob + uniform_frac / nv
            prob = np.maximum(prob, 1.0e-300)
            prob /= prob.sum()
        iv = np.random.choice(nv, size=sample_count, replace=True, p=prob)
        iv_np[row_index] = iv
        prob_np[row_index] = prob[iv]

    iv = torch.from_numpy(iv_np).to(device=device, dtype=torch.long)
    prob = torch.from_numpy(prob_np).to(device=device)
    v_sample = v[iv]
    w_sample = w[iv]
    vn_sample = v_norm[iv]
    z = torch.empty((nx, sample_count, 4), device=device)
    z[:, :, 0] = x_norm[:, None]
    z[:, :, 1:] = vn_sample
    mach = torch.as_tensor(mach_norm, dtype=z.dtype, device=device)
    logf = model(mach, z.reshape(-1, 4)).reshape(nx, sample_count)
    fhat = torch.exp(torch.clamp(logf, -90.0, 30.0))
    fw = fhat * w_sample / (sample_count * prob.clamp_min(1.0e-30))

    rho = torch.sum(fw, dim=1).clamp_min(1.0e-30)
    m1 = torch.sum(fw[:, :, None] * v_sample, dim=1)
    u = m1 / rho[:, None]
    e2 = torch.sum(fw * torch.sum(v_sample * v_sample, dim=2), dim=1)
    temperature = (e2 / rho - torch.sum(u * u, dim=1)) / 3.0
    c = v_sample - u[:, None, :]
    c2 = torch.sum(c * c, dim=2)
    cx = c[:, :, 0]
    sxx_raw = torch.sum(fw * cx * cx, dim=1)
    pred = {
        "rho": rho,
        "ux": u[:, 0],
        "T": temperature,
        "qx": 0.5 * torch.sum(fw * c2 * cx, dim=1),
        "sig": sxx_raw - rho * temperature,
        "M300": torch.sum(fw * cx**3, dim=1),
        "M400": torch.sum(fw * cx**4, dim=1),
    }
    pred["M400neq"] = pred["M400"] - 3.0 * rho * temperature**2
    ref = data.reference_for_keys(keys, ix, device=device)
    loss = torch.zeros((), device=device)
    terms: Dict[str, float] = {}
    for key in keys:
        normalized_key = "sig" if key == "sigma_xx" else key
        if normalized_key not in pred or normalized_key not in ref:
            continue
        scale = float(data.moment_scales.get(normalized_key, 1.0))
        term = torch.mean(((pred[normalized_key] - ref[normalized_key]) / scale) ** 2)
        loss += term
        terms[normalized_key] = float(term.detach().cpu())
    if terms:
        loss /= float(len(terms))
    return loss, terms
