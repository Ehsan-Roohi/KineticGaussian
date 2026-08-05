from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class MomentGaussian1D(nn.Module):
    """Normalized Gaussian kernel model for 1D moment profiles U(x)."""

    def __init__(
        self,
        num_kernels: int,
        num_outputs: int,
        log_scale_min: float = -8.0,
        log_scale_max: float = 0.0,
        init_log_scale: float = -2.0,
    ):
        super().__init__()
        self.num_kernels = int(num_kernels)
        self.num_outputs = int(num_outputs)
        self.log_scale_min = float(log_scale_min)
        self.log_scale_max = float(log_scale_max)
        centers = torch.linspace(-1.0, 1.0, num_kernels).view(num_kernels, 1)
        self.center_raw = nn.Parameter(torch.atanh(0.95 * centers))
        self.scale_raw = nn.Parameter(self._inverse_bound(torch.full((num_kernels, 1), float(init_log_scale))))
        self.amp = nn.Parameter(0.01 * torch.randn(num_kernels, num_outputs))

    def _inverse_bound(self, y: torch.Tensor) -> torch.Tensor:
        lo, hi = self.log_scale_min, self.log_scale_max
        y01 = ((y - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        return torch.log(y01 / (1 - y01))

    def centers(self) -> torch.Tensor:
        return torch.tanh(self.center_raw)

    def scales(self) -> torch.Tensor:
        log_s = self.log_scale_min + (self.log_scale_max - self.log_scale_min) * torch.sigmoid(self.scale_raw)
        return torch.exp(log_s)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        if x_norm.ndim == 1:
            x_norm = x_norm[:, None]
        dx = x_norm[:, None, :] - self.centers()[None, :, :]
        q = torch.sum((dx / (self.scales()[None, :, :] + 1e-20)) ** 2, dim=-1)
        g = torch.exp(-0.5 * q)
        w = g / (torch.sum(g, dim=1, keepdim=True) + 1e-12)
        return w @ self.amp

    def metadata(self) -> Dict[str, object]:
        return {
            "model_class": self.__class__.__name__,
            "num_kernels": self.num_kernels,
            "num_outputs": self.num_outputs,
            "parameter_count": int(sum(p.numel() for p in self.parameters())),
            "log_scale_min": self.log_scale_min,
            "log_scale_max": self.log_scale_max,
        }
