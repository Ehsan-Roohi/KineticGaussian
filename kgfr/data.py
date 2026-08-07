from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .utils import rms_scale


@dataclass
class CoordinateNormalizer:
    center: np.ndarray
    halfwidth: np.ndarray

    @classmethod
    def from_x_v(cls, x: np.ndarray, v: np.ndarray) -> "CoordinateNormalizer":
        z_min = np.array([x.min(), v[:, 0].min(), v[:, 1].min(), v[:, 2].min()], dtype=np.float32)
        z_max = np.array([x.max(), v[:, 0].max(), v[:, 1].max(), v[:, 2].max()], dtype=np.float32)
        center = 0.5 * (z_min + z_max)
        halfwidth = 0.5 * (z_max - z_min)
        halfwidth = np.where(halfwidth <= 0, 1.0, halfwidth).astype(np.float32)
        return cls(center=center.astype(np.float32), halfwidth=halfwidth.astype(np.float32))

    @classmethod
    def from_bounds(cls, z_min: np.ndarray, z_max: np.ndarray) -> "CoordinateNormalizer":
        z_min = np.asarray(z_min, dtype=np.float32)
        z_max = np.asarray(z_max, dtype=np.float32)
        center = 0.5 * (z_min + z_max)
        halfwidth = 0.5 * (z_max - z_min)
        halfwidth = np.where(halfwidth <= 0, 1.0, halfwidth).astype(np.float32)
        return cls(center=center.astype(np.float32), halfwidth=halfwidth)

    @classmethod
    def from_state_dict(cls, state: Dict[str, list]) -> "CoordinateNormalizer":
        return cls(
            center=np.asarray(state["center"], dtype=np.float32),
            halfwidth=np.asarray(state["halfwidth"], dtype=np.float32),
        )

    def normalize_z(self, z: np.ndarray) -> np.ndarray:
        return ((z - self.center[None, :]) / self.halfwidth[None, :]).astype(np.float32)

    def normalize_xv(self, x_vals: np.ndarray, v_vals: np.ndarray) -> np.ndarray:
        z = np.empty((len(x_vals), 4), dtype=np.float32)
        z[:, 0] = x_vals
        z[:, 1:] = v_vals
        return self.normalize_z(z)

    def state_dict(self) -> Dict[str, list]:
        return {"center": self.center.tolist(), "halfwidth": self.halfwidth.tolist()}


class ShockFullState:
    """Loads one full-state DVM shock NPZ and provides phase-space sampling."""

    def __init__(
        self,
        path: str | Path,
        moment_path: str | Path | None = None,
        f_floor: float = 1e-35,
        normalizer: CoordinateNormalizer | None = None,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Full-state file not found: {self.path}")
        print(f"[kgfr] loading full-state file: {self.path}")
        d = np.load(self.path, allow_pickle=True)
        self.keys = list(d.files)
        required = ["x", "f", "v", "w", "rho", "ux", "T", "qx"]
        missing = [k for k in required if k not in d.files]
        if missing:
            raise KeyError(f"Missing keys in {self.path}: {missing}")

        self.x = np.asarray(d["x"], dtype=np.float32)
        self.f = np.asarray(d["f"], dtype=np.float32)
        self.v = np.asarray(d["v"], dtype=np.float32)
        self.w = np.asarray(d["w"], dtype=np.float32)
        self.rho = np.asarray(d["rho"], dtype=np.float32)
        self.ux = np.asarray(d["ux"], dtype=np.float32)
        self.T = np.asarray(d["T"], dtype=np.float32)
        self.qx = np.asarray(d["qx"], dtype=np.float32)
        self.sig = np.asarray(d["sig"] if "sig" in d.files else d["sigma_xx"], dtype=np.float32)
        self.Nx, self.Nv = self.f.shape
        if self.v.shape != (self.Nv, 3):
            raise ValueError(f"Expected v shape ({self.Nv},3), got {self.v.shape}")
        if self.w.shape[0] != self.Nv:
            raise ValueError(f"Expected w shape ({self.Nv},), got {self.w.shape}")
        self.f_floor = float(f_floor)
        self.norm = normalizer or CoordinateNormalizer.from_x_v(self.x, self.v)
        self.v_norm = ((self.v - self.norm.center[None, 1:]) / self.norm.halfwidth[None, 1:]).astype(np.float32)
        self.x_norm = ((self.x - self.norm.center[0]) / self.norm.halfwidth[0]).astype(np.float32)
        self.moment_ref: Dict[str, np.ndarray] = {
            "rho": self.rho,
            "ux": self.ux,
            "T": self.T,
            "qx": self.qx,
            "sig": self.sig,
        }
        if moment_path is not None and Path(moment_path).exists():
            print(f"[kgfr] loading optional high-moment file: {moment_path}")
            md = np.load(moment_path, allow_pickle=True)
            for k in md.files:
                arr = md[k]
                if np.ndim(arr) == 1 and arr.shape[0] == self.Nx:
                    self.moment_ref[k] = np.asarray(arr, dtype=np.float32)
            if "sigma_xx" in self.moment_ref:
                self.moment_ref["sig"] = self.moment_ref["sigma_xx"]
            # High-Mach DVM references may provide moments of f-Md, where Md
            # is the local conservative discrete Maxwellian.  Prefer these
            # quadrature-offset-free diagnostics over raw continuum
            # subtractions when they are available.
            if "qx_neq_discrete" in self.moment_ref:
                self.moment_ref["qx"] = self.moment_ref["qx_neq_discrete"]
                self.qx = self.moment_ref["qx"]
            if "sig_neq_discrete" in self.moment_ref:
                self.moment_ref["sig"] = self.moment_ref["sig_neq_discrete"]
                self.sig = self.moment_ref["sig"]
            # Aliases so phase-space moment_loss can train model M300/M400
            # against high-moment diagnostics from the DVM file.
            if "M300_neq" in self.moment_ref:
                self.moment_ref["M300"] = self.moment_ref["M300_neq"]
            if "M400_raw" in self.moment_ref:
                self.moment_ref["M400"] = self.moment_ref["M400_raw"]
        if "M300" not in self.moment_ref or "M400" not in self.moment_ref:
            derived_m300, derived_m400 = self._derive_longitudinal_moments()
            self.moment_ref.setdefault("M300", derived_m300)
            self.moment_ref.setdefault("M400", derived_m400)
        if "M400_neq" in self.moment_ref:
            self.moment_ref["M400neq"] = self.moment_ref["M400_neq"]
        else:
            self.moment_ref["M400neq"] = (
                self.moment_ref["M400"] - 3.0 * self.rho * self.T**2
            ).astype(np.float32)
        self.moment_scales = {k: rms_scale(v) for k, v in self.moment_ref.items() if np.ndim(v) == 1}
        self._x_sampling_p = self._build_x_sampling_weights()
        self._torch_cache: Dict[str, torch.Tensor] = {}

    def _derive_longitudinal_moments(self, velocity_chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
        """Integrate central third/fourth streamwise moments without a large temporary phase array."""
        m300 = np.zeros(self.Nx, dtype=np.float64)
        m400 = np.zeros(self.Nx, dtype=np.float64)
        ux = self.ux.astype(np.float64)
        for j0 in range(0, self.Nv, velocity_chunk):
            j1 = min(self.Nv, j0 + velocity_chunk)
            cx = self.v[None, j0:j1, 0].astype(np.float64) - ux[:, None]
            fw = self.f[:, j0:j1].astype(np.float64) * self.w[None, j0:j1].astype(np.float64)
            m300 += np.sum(fw * cx**3, axis=1)
            m400 += np.sum(fw * cx**4, axis=1)
        return m300.astype(np.float32), m400.astype(np.float32)

    def set_coordinate_normalizer(self, normalizer: CoordinateNormalizer) -> None:
        self.norm = normalizer
        self.v_norm = ((self.v - self.norm.center[None, 1:]) / self.norm.halfwidth[None, 1:]).astype(np.float32)
        self.x_norm = ((self.x - self.norm.center[0]) / self.norm.halfwidth[0]).astype(np.float32)
        self._torch_cache.clear()

    def _build_x_sampling_weights(self) -> np.ndarray:
        gr = np.gradient(self.rho.astype(np.float64))
        score = np.abs(gr) / (np.max(np.abs(gr)) + 1e-30)
        score += np.abs(self.qx) / (np.max(np.abs(self.qx)) + 1e-30)
        score += np.abs(self.sig) / (np.max(np.abs(self.sig)) + 1e-30)
        p = 1.0 + 3.0 * score
        p = p / p.sum()
        return p.astype(np.float64)

    def sample_x_indices(self, n: int, shock_weighted_frac: float = 0.70) -> np.ndarray:
        n_weighted = int(round(n * shock_weighted_frac))
        n_uniform = n - n_weighted
        parts = []
        if n_weighted > 0:
            parts.append(np.random.choice(self.Nx, size=n_weighted, replace=True, p=self._x_sampling_p))
        if n_uniform > 0:
            parts.append(np.random.randint(0, self.Nx, size=n_uniform))
        ix = np.concatenate(parts) if len(parts) > 1 else parts[0]
        np.random.shuffle(ix)
        return ix.astype(np.int64)

    def sample_phase_batch(
        self,
        x_batch: int,
        vel_per_x: int,
        uniform_vel_frac: float = 0.20,
        mass_alpha: float = 0.70,
        device: torch.device | str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return normalized z, target log(f), and optional sample weights."""
        ix_all = self.sample_x_indices(x_batch)
        n_uniform = int(round(vel_per_x * uniform_vel_frac))
        n_biased = vel_per_x - n_uniform
        total = x_batch * vel_per_x
        z = np.empty((total, 4), dtype=np.float32)
        logf = np.empty((total,), dtype=np.float32)
        sw = np.ones((total,), dtype=np.float32)

        pos = 0
        for ix in ix_all:
            row = self.f[ix]
            iv_parts = []
            if n_biased > 0:
                mass = np.maximum(row, 0.0).astype(np.float64) * self.w.astype(np.float64)
                if mass_alpha != 1.0:
                    mass = np.power(mass + 1e-300, mass_alpha)
                s = float(mass.sum())
                if not np.isfinite(s) or s <= 0:
                    p = None
                else:
                    p = mass / s
                iv_parts.append(np.random.choice(self.Nv, size=n_biased, replace=True, p=p))
            if n_uniform > 0:
                iv_parts.append(np.random.randint(0, self.Nv, size=n_uniform))
            iv = np.concatenate(iv_parts) if len(iv_parts) > 1 else iv_parts[0]
            np.random.shuffle(iv)
            sl = slice(pos, pos + vel_per_x)
            z[sl, 0] = self.x_norm[ix]
            z[sl, 1:] = self.v_norm[iv]
            logf[sl] = np.log(np.maximum(row[iv], self.f_floor)).astype(np.float32)
            # Give non-negligible tails some influence but avoid exploding weights.
            fval = np.maximum(row[iv], self.f_floor)
            sw[sl] = np.clip(np.sqrt(fval / (np.mean(fval) + self.f_floor)), 0.25, 4.0).astype(np.float32)
            pos += vel_per_x

        device = torch.device(device)
        return (
            torch.from_numpy(z).to(device=device),
            torch.from_numpy(logf).to(device=device),
            torch.from_numpy(sw).to(device=device),
        )

    def sample_init_points(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        x_batch = max(1, min(512, int(np.sqrt(n))))
        vel_per_x = max(1, int(np.ceil(n / x_batch)))
        z, _, _ = self.sample_phase_batch(
            x_batch=x_batch,
            vel_per_x=vel_per_x,
            uniform_vel_frac=0.05,
            mass_alpha=0.50,
            device=device,
        )
        if z.shape[0] > n:
            z = z[torch.randperm(z.shape[0], device=z.device)[:n]]
        return z

    def torch_cache(self, device: torch.device | str) -> Dict[str, torch.Tensor]:
        device = torch.device(device)
        key = str(device)
        if key in self._torch_cache:
            return self._torch_cache[key]
        cache = {
            "v": torch.from_numpy(self.v).to(device=device),
            "w": torch.from_numpy(self.w).to(device=device),
            "v_norm": torch.from_numpy(self.v_norm).to(device=device),
            "x_norm": torch.from_numpy(self.x_norm).to(device=device),
        }
        for k, arr in self.moment_ref.items():
            if np.ndim(arr) == 1 and arr.shape[0] == self.Nx:
                cache[k] = torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device=device)
        self._torch_cache[key] = cache
        return cache

    def reference_for_keys(self, keys: Iterable[str], ix: np.ndarray | torch.Tensor, device: torch.device | str) -> Dict[str, torch.Tensor]:
        cache = self.torch_cache(device)
        if isinstance(ix, np.ndarray):
            ixt = torch.from_numpy(ix.astype(np.int64)).to(device=device)
        else:
            ixt = ix.to(device=device, dtype=torch.long)
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            kk = "sig" if k == "sigma_xx" else k
            if kk in cache:
                out[kk] = cache[kk][ixt]
        return out

    def summary(self) -> str:
        return (
            f"ShockFullState(path={self.path}, Nx={self.Nx}, Nv={self.Nv}, "
            f"f=[{float(np.nanmin(self.f)):.3e},{float(np.nanmax(self.f)):.3e}], "
            f"x=[{float(self.x.min()):.3g},{float(self.x.max()):.3g}])"
        )
