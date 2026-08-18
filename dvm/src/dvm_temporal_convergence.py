"""Temporal steady-state diagnostics for fixed-step DVM calculations.

The tracker is deliberately independent of PyTorch and does not stop a solver.
It records whether several consecutive saved states changed by less than the
configured relative-L2 tolerances.  The caller decides whether a completed
fixed-step calculation is eligible for numerical certification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


MACRO_KEYS = ("rho", "ux", "T")
NONEQ_KEYS = ("qx", "sig", "M300", "M400neq")


def _array(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def relative_change(current: object, previous: object) -> float:
    """Return a symmetric relative L2 change with a finite near-zero floor."""

    current_array = _array(current)
    previous_array = _array(previous)
    denominator = max(
        float(np.linalg.norm(current_array)),
        float(np.linalg.norm(previous_array)),
        1.0e-30,
    )
    return float(np.linalg.norm(current_array - previous_array) / denominator)


@dataclass
class TemporalConvergenceTracker:
    macro_tolerance: float
    noneq_tolerance: float
    required_consecutive_checks: int
    min_step: int
    previous: dict[str, np.ndarray] | None = None
    consecutive_passes: int = 0
    history: list[dict[str, object]] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.required_consecutive_checks > 0

    @property
    def converged(self) -> bool:
        return self.enabled and self.consecutive_passes >= self.required_consecutive_checks

    def update(self, step: int, profiles: Mapping[str, object]) -> dict[str, object]:
        missing = [key for key in MACRO_KEYS + NONEQ_KEYS if key not in profiles]
        if missing:
            raise KeyError(f"Temporal convergence profiles missing {missing}")

        current = {key: _array(profiles[key]).copy() for key in MACRO_KEYS + NONEQ_KEYS}
        macro_change = None
        noneq_change = None
        passed = False
        eligible = bool(self.enabled and step >= self.min_step and self.previous is not None)
        if self.previous is not None:
            macro_change = max(relative_change(current[key], self.previous[key]) for key in MACRO_KEYS)
            noneq_change = max(relative_change(current[key], self.previous[key]) for key in NONEQ_KEYS)
        if eligible:
            passed = bool(
                macro_change is not None
                and noneq_change is not None
                and macro_change <= self.macro_tolerance
                and noneq_change <= self.noneq_tolerance
            )
            self.consecutive_passes = self.consecutive_passes + 1 if passed else 0

        entry = {
            "step": int(step),
            "eligible": eligible,
            "macro_relative_l2_max": macro_change,
            "noneq_relative_l2_max": noneq_change,
            "passed": passed,
            "consecutive_passes": int(self.consecutive_passes),
        }
        self.history.append(entry)
        self.previous = current
        return entry

    def metadata(self, completed_steps: int) -> dict[str, np.ndarray]:
        latest = self.history[-1] if self.history else {}
        return {
            "temporal_gate_enabled": np.array(self.enabled),
            "temporal_converged": np.array(self.converged),
            "temporal_completed_steps": np.array(int(completed_steps)),
            "temporal_min_step": np.array(int(self.min_step)),
            "temporal_macro_tolerance": np.array(float(self.macro_tolerance)),
            "temporal_noneq_tolerance": np.array(float(self.noneq_tolerance)),
            "temporal_required_consecutive_checks": np.array(int(self.required_consecutive_checks)),
            "temporal_consecutive_passes": np.array(int(self.consecutive_passes)),
            "temporal_latest_macro_relative_l2_max": np.array(
                np.nan if latest.get("macro_relative_l2_max") is None else latest["macro_relative_l2_max"]
            ),
            "temporal_latest_noneq_relative_l2_max": np.array(
                np.nan if latest.get("noneq_relative_l2_max") is None else latest["noneq_relative_l2_max"]
            ),
            "temporal_history_json": np.array(json.dumps(self.history, separators=(",", ":"))),
        }
