from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class PhaseGaussianMixture(nn.Module):
    """Positive Gaussian mixture in normalized phase space z=(x,vx,vy,vz).

    The model predicts log(f) through log-sum-exp of positive Gaussian kernels.
    Variants:
      - isotropic: one scale per kernel
      - diag: one scale per kernel and dimension
      - xvx: correlated anisotropy in (x,vx), diagonal in (vy,vz)
    """

    def __init__(
        self,
        num_kernels: int,
        variant: str = "diag",
        dim: int = 4,
        log_scale_min: float = -8.0,
        log_scale_max: float = 1.0,
        init_log_scale: float = -1.2,
        init_log_amp: float = -8.0,
    ):
        super().__init__()
        if dim != 4:
            raise ValueError("This implementation assumes dim=4: x,vx,vy,vz.")
        if variant not in {"isotropic", "diag", "xvx"}:
            raise ValueError(f"Unknown variant: {variant}")
        self.num_kernels = int(num_kernels)
        self.variant = variant
        self.dim = dim
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)

        self.center_raw = nn.Parameter(0.2 * torch.randn(num_kernels, dim))
        if variant == "isotropic":
            scale_shape = (num_kernels, 1)
        else:
            scale_shape = (num_kernels, dim)
        init_raw = self._inverse_bound(torch.full(scale_shape, float(init_log_scale)))
        self.scale_raw = nn.Parameter(init_raw + 0.05 * torch.randn(scale_shape))
        self.log_amp = nn.Parameter(torch.full((num_kernels,), float(init_log_amp)) + 0.1 * torch.randn(num_kernels))
        if variant == "xvx":
            self.corr_raw = nn.Parameter(torch.zeros(num_kernels))
        else:
            self.corr_raw = None

    def _inverse_bound(self, y: torch.Tensor) -> torch.Tensor:
        lo, hi = self.log_scale_min, self.log_scale_max
        y01 = ((y - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.log(y01 / (1 - y01))

    def centers(self) -> torch.Tensor:
        # Allow a small margin outside [-1,1] to reduce edge artifacts.
        return 1.15 * torch.tanh(self.center_raw)

    def log_scales(self) -> torch.Tensor:
        return self.log_scale_min + (self.log_scale_max - self.log_scale_min) * torch.sigmoid(self.scale_raw)

    def scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales())

    @torch.no_grad()
    def initialize_centers_from_samples(self, z_samples: torch.Tensor) -> None:
        """Initialize centers from representative data samples in normalized coordinates."""
        if z_samples.ndim != 2 or z_samples.shape[1] != self.dim:
            raise ValueError(f"Expected z_samples shape (M,{self.dim}), got {tuple(z_samples.shape)}")
        m = z_samples.shape[0]
        if m < self.num_kernels:
            idx = torch.randint(0, m, (self.num_kernels,), device=z_samples.device)
        else:
            idx = torch.randperm(m, device=z_samples.device)[: self.num_kernels]
        c = z_samples[idx].clamp(-1.10, 1.10) / 1.15
        c = c.clamp(-0.999, 0.999)
        self.center_raw.copy_(torch.atanh(c).to(self.center_raw.device))

    def quadratic_form(self, z: torch.Tensor) -> torch.Tensor:
        c = self.centers()
        dz = z[:, None, :] - c[None, :, :]
        if self.variant == "isotropic":
            s = self.scales()[:, 0]
            q = torch.sum(dz * dz, dim=-1) / (s[None, :] ** 2 + 1e-20)
            return q
        s = self.scales()
        if self.variant == "diag":
            q = torch.sum((dz / (s[None, :, :] + 1e-20)) ** 2, dim=-1)
            return q
        # xvx variant: correlated Gaussian in the (x,vx) block.
        a = dz[:, :, 0] / (s[None, :, 0] + 1e-20)
        b = dz[:, :, 1] / (s[None, :, 1] + 1e-20)
        rho = 0.95 * torch.tanh(self.corr_raw)[None, :]
        denom = 1.0 - rho * rho + 1e-6
        q_xvx = (a * a - 2.0 * rho * a * b + b * b) / denom
        q_yz = (dz[:, :, 2] / (s[None, :, 2] + 1e-20)) ** 2 + (dz[:, :, 3] / (s[None, :, 3] + 1e-20)) ** 2
        return q_xvx + q_yz

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q = self.quadratic_form(z)
        log_terms = self.log_amp[None, :] - 0.5 * q
        return torch.logsumexp(log_terms, dim=1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def metadata(self) -> Dict[str, object]:
        return {
            "model_class": self.__class__.__name__,
            "variant": self.variant,
            "num_kernels": self.num_kernels,
            "dim": self.dim,
            "parameter_count": self.parameter_count(),
            "log_scale_min": self.log_scale_min,
            "log_scale_max": self.log_scale_max,
        }


def optimizer_for_model(model: PhaseGaussianMixture, cfg_train: dict) -> torch.optim.Optimizer:
    lr = float(cfg_train.get("lr", 1e-3))
    groups = [
        {"params": [model.center_raw], "lr": lr * float(cfg_train.get("center_lr_mult", 1.0))},
        {"params": [model.scale_raw], "lr": lr * float(cfg_train.get("scale_lr_mult", 1.0))},
        {"params": [model.log_amp], "lr": lr * float(cfg_train.get("amp_lr_mult", 1.0))},
    ]
    if model.corr_raw is not None:
        groups.append({"params": [model.corr_raw], "lr": lr * float(cfg_train.get("corr_lr_mult", 1.0))})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=float(cfg_train.get("weight_decay", 0.0)))
