from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ConditionalPhaseGaussianMixture(nn.Module):
    """Positive Gaussian phase-space representation conditioned on Mach number.

    Kernel amplitudes, centers, widths, and optional x-vx correlations are
    smooth low-order Legendre functions of normalized Mach number. Positivity
    is exact because the prediction is a log-sum-exp of positive kernels.
    """

    def __init__(
        self,
        num_kernels: int,
        variant: str = "xvx",
        mach_degree: int = 2,
        dim: int = 4,
        log_scale_min: float = -6.0,
        log_scale_max: float = -0.7,
        init_log_scale: float = -2.4,
        init_log_amp: float = -7.0,
    ) -> None:
        super().__init__()
        if dim != 4:
            raise ValueError("dim must be 4 for (x,vx,vy,vz)")
        if variant not in {"diag", "xvx"}:
            raise ValueError(f"Unknown conditional variant: {variant}")
        if mach_degree < 0 or mach_degree > 5:
            raise ValueError("mach_degree must be between 0 and 5")
        self.num_kernels = int(num_kernels)
        self.variant = variant
        self.mach_degree = int(mach_degree)
        self.dim = int(dim)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        terms = self.mach_degree + 1

        center = torch.zeros(terms, self.num_kernels, self.dim)
        center[0] = 0.2 * torch.randn(self.num_kernels, self.dim)
        self.center_coeff = nn.Parameter(center)

        scale = torch.zeros(terms, self.num_kernels, self.dim)
        scale[0] = self._inverse_bound(torch.full((self.num_kernels, self.dim), float(init_log_scale)))
        scale[0] += 0.05 * torch.randn(self.num_kernels, self.dim)
        self.scale_coeff = nn.Parameter(scale)

        amp = torch.zeros(terms, self.num_kernels)
        amp[0] = float(init_log_amp) + 0.1 * torch.randn(self.num_kernels)
        self.log_amp_coeff = nn.Parameter(amp)

        if self.variant == "xvx":
            self.corr_coeff = nn.Parameter(torch.zeros(terms, self.num_kernels))
        else:
            self.register_parameter("corr_coeff", None)

    def _inverse_bound(self, value: torch.Tensor) -> torch.Tensor:
        lo, hi = self.log_scale_min, self.log_scale_max
        y = ((value - lo) / (hi - lo)).clamp(1.0e-4, 1.0 - 1.0e-4)
        return torch.log(y / (1.0 - y))

    def mach_features(self, mach: torch.Tensor) -> torch.Tensor:
        m = mach.reshape(-1)
        values = [torch.ones_like(m)]
        if self.mach_degree >= 1:
            values.append(m)
        if self.mach_degree >= 2:
            values.append(0.5 * (3.0 * m * m - 1.0))
        for degree in range(3, self.mach_degree + 1):
            pn = ((2 * degree - 1) * m * values[-1] - (degree - 1) * values[-2]) / degree
            values.append(pn)
        return torch.stack(values, dim=1)

    def _conditioned(self, mach: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        phi = self.mach_features(mach)
        centers_raw = torch.einsum("bt,tnd->bnd", phi, self.center_coeff)
        scale_raw = torch.einsum("bt,tnd->bnd", phi, self.scale_coeff)
        log_amp = torch.einsum("bt,tn->bn", phi, self.log_amp_coeff)
        centers = 1.15 * torch.tanh(centers_raw)
        log_scales = self.log_scale_min + (
            self.log_scale_max - self.log_scale_min
        ) * torch.sigmoid(scale_raw)
        scales = torch.exp(log_scales)
        corr = None
        if self.corr_coeff is not None:
            corr = 0.95 * torch.tanh(torch.einsum("bt,tn->bn", phi, self.corr_coeff))
        return centers, scales, log_amp, corr

    def forward(self, mach: torch.Tensor | float, z: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(mach):
            mach = torch.tensor(float(mach), dtype=z.dtype, device=z.device)
        mach = mach.to(device=z.device, dtype=z.dtype).reshape(-1)
        if mach.numel() not in (1, z.shape[0]):
            raise ValueError(f"Mach input must have 1 or {z.shape[0]} entries, got {mach.numel()}")
        centers, scales, log_amp, corr = self._conditioned(mach)
        if mach.numel() == 1:
            dz = z[:, None, :] - centers
        else:
            dz = z[:, None, :] - centers

        if self.variant == "diag":
            q = torch.sum((dz / (scales + 1.0e-20)) ** 2, dim=-1)
        else:
            a = dz[:, :, 0] / (scales[:, :, 0] + 1.0e-20)
            b = dz[:, :, 1] / (scales[:, :, 1] + 1.0e-20)
            denom = 1.0 - corr * corr + 1.0e-6
            q_xvx = (a * a - 2.0 * corr * a * b + b * b) / denom
            q_yz = (dz[:, :, 2] / (scales[:, :, 2] + 1.0e-20)) ** 2
            q_yz += (dz[:, :, 3] / (scales[:, :, 3] + 1.0e-20)) ** 2
            q = q_xvx + q_yz
        return torch.logsumexp(log_amp - 0.5 * q, dim=1)

    @torch.no_grad()
    def initialize_centers_from_samples(self, z_samples: torch.Tensor) -> None:
        if z_samples.ndim != 2 or z_samples.shape[1] != self.dim:
            raise ValueError(f"Expected samples with shape (M,{self.dim})")
        count = z_samples.shape[0]
        if count < self.num_kernels:
            idx = torch.randint(0, count, (self.num_kernels,), device=z_samples.device)
        else:
            idx = torch.randperm(count, device=z_samples.device)[: self.num_kernels]
        centers = (z_samples[idx].clamp(-1.10, 1.10) / 1.15).clamp(-0.999, 0.999)
        self.center_coeff[0].copy_(torch.atanh(centers).to(self.center_coeff.device))
        if self.center_coeff.shape[0] > 1:
            self.center_coeff[1:].zero_()

    def parameter_count(self) -> int:
        return int(sum(item.numel() for item in self.parameters()))

    def metadata(self) -> Dict[str, object]:
        return {
            "model_class": self.__class__.__name__,
            "variant": self.variant,
            "num_kernels": self.num_kernels,
            "mach_degree": self.mach_degree,
            "parameter_count": self.parameter_count(),
            "log_scale_min": self.log_scale_min,
            "log_scale_max": self.log_scale_max,
        }


def optimizer_for_conditional_model(
    model: ConditionalPhaseGaussianMixture,
    cfg: dict,
) -> torch.optim.Optimizer:
    lr = float(cfg.get("lr", 5.0e-4))
    groups = [
        {"params": [model.center_coeff], "lr": lr * float(cfg.get("center_lr_mult", 0.5))},
        {"params": [model.scale_coeff], "lr": lr * float(cfg.get("scale_lr_mult", 0.25))},
        {"params": [model.log_amp_coeff], "lr": lr * float(cfg.get("amp_lr_mult", 1.0))},
    ]
    if model.corr_coeff is not None:
        groups.append({"params": [model.corr_coeff], "lr": lr * float(cfg.get("corr_lr_mult", 0.2))})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=float(cfg.get("weight_decay", 0.0)))
