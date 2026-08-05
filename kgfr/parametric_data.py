from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from .data import ShockFullState


@dataclass(frozen=True)
class ShockCaseSpec:
    name: str
    mach: float
    data_path: Path
    moment_path: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict, base_dir: Path) -> "ShockCaseSpec":
        def resolve(value: str | None) -> Path | None:
            if value is None:
                return None
            expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
            return expanded if expanded.is_absolute() else (base_dir / expanded).resolve()

        if "name" not in raw or "mach" not in raw or "data_path" not in raw:
            raise KeyError("Each manifest case requires name, mach, and data_path")
        return cls(
            name=str(raw["name"]),
            mach=float(raw["mach"]),
            data_path=resolve(raw["data_path"]),
            moment_path=resolve(raw.get("moment_path")),
        )

    def state_dict(self) -> dict:
        return {
            "name": self.name,
            "mach": self.mach,
            "data_path": str(self.data_path),
            "moment_path": None if self.moment_path is None else str(self.moment_path),
        }


def load_case_manifest(path: str | Path) -> tuple[List[ShockCaseSpec], dict]:
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [ShockCaseSpec.from_dict(item, manifest_path.parent) for item in raw.get("cases", [])]
    if not cases:
        raise ValueError(f"No cases found in {manifest_path}")
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError(f"Case names must be unique: {names}")
    machs = [case.mach for case in cases]
    if len(machs) != len(set(machs)):
        raise ValueError(f"Mach values must be unique: {machs}")
    return cases, raw


def select_case_specs(specs: Sequence[ShockCaseSpec], names: Iterable[str]) -> List[ShockCaseSpec]:
    requested = list(names)
    by_name = {case.name: case for case in specs}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise KeyError(f"Unknown cases {missing}; available cases are {sorted(by_name)}")
    return [by_name[name] for name in requested]


class ParametricShockDataset:
    """A collection of DVM shocks sampled uniformly by flow case.

    Each case keeps its original physical quadrature and its own affine map of
    (x,vx,vy,vz) to [-1,1]^4. Mach number uses one fixed map shared by every
    training and held-out case. No held-out distribution values are loaded.
    """

    def __init__(
        self,
        specs: Sequence[ShockCaseSpec],
        mach_bounds: Sequence[float],
        f_floor: float = 1.0e-35,
    ) -> None:
        if len(specs) < 1:
            raise ValueError("At least one training case is required")
        if len(mach_bounds) != 2 or not float(mach_bounds[1]) > float(mach_bounds[0]):
            raise ValueError(f"Invalid mach_bounds: {mach_bounds}")
        self.specs = list(specs)
        self.mach_min = float(mach_bounds[0])
        self.mach_max = float(mach_bounds[1])
        self.mach_center = 0.5 * (self.mach_min + self.mach_max)
        self.mach_halfwidth = 0.5 * (self.mach_max - self.mach_min)
        self.cases: Dict[str, ShockFullState] = {}
        for spec in self.specs:
            self.cases[spec.name] = ShockFullState(
                spec.data_path,
                moment_path=spec.moment_path,
                f_floor=f_floor,
            )

    def normalize_mach(self, mach: float) -> float:
        return float((float(mach) - self.mach_center) / self.mach_halfwidth)

    def sample_phase_batch(
        self,
        x_batch: int,
        vel_per_x: int,
        uniform_vel_frac: float,
        mass_alpha: float,
        device: torch.device | str,
        case_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        if case_name is None:
            spec = self.specs[int(np.random.randint(0, len(self.specs)))]
        else:
            spec = next((item for item in self.specs if item.name == case_name), None)
            if spec is None:
                raise KeyError(f"Case {case_name!r} is not loaded")
        z, target_logf, sample_w = self.cases[spec.name].sample_phase_batch(
            x_batch=x_batch,
            vel_per_x=vel_per_x,
            uniform_vel_frac=uniform_vel_frac,
            mass_alpha=mass_alpha,
            device=device,
        )
        mach = torch.tensor(self.normalize_mach(spec.mach), dtype=z.dtype, device=z.device)
        return mach, z, target_logf, sample_w, spec.name

    def sample_init_points(
        self,
        n: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = [n // len(self.specs)] * len(self.specs)
        for i in range(n % len(self.specs)):
            counts[i] += 1
        mach_parts: List[torch.Tensor] = []
        z_parts: List[torch.Tensor] = []
        for spec, count in zip(self.specs, counts):
            if count <= 0:
                continue
            z = self.cases[spec.name].sample_init_points(count, device=device)
            mach_parts.append(torch.full((len(z),), self.normalize_mach(spec.mach), device=z.device))
            z_parts.append(z)
        mach = torch.cat(mach_parts)
        z = torch.cat(z_parts)
        order = torch.randperm(len(z), device=z.device)
        return mach[order], z[order]

    def coordinate_state(self) -> dict:
        return {
            "mach_bounds": [self.mach_min, self.mach_max],
            "mach_center": self.mach_center,
            "mach_halfwidth": self.mach_halfwidth,
            "cases": {
                spec.name: {
                    "mach": spec.mach,
                    "coordinate_normalizer": self.cases[spec.name].norm.state_dict(),
                }
                for spec in self.specs
            },
        }

    def summary(self) -> str:
        rows = [
            f"{spec.name}: M={spec.mach:g}, Nx={self.cases[spec.name].Nx}, "
            f"Nv={self.cases[spec.name].Nv}"
            for spec in self.specs
        ]
        return "ParametricShockDataset(" + "; ".join(rows) + ")"
