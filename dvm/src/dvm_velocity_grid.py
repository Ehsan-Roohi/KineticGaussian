#!/usr/bin/env python
"""Velocity-grid construction shared by the DVM solver and JCP audits.

The legacy shock calculations used one uniform cube ``[-vmax, vmax]^3``.
That becomes both wasteful and inaccurate at high Mach number: the grid must
simultaneously resolve a narrow upstream Maxwellian and a much hotter
downstream Maxwellian.  This module builds a deterministic composite tensor
grid with fine patches around both Rankine--Hugoniot states and coarser tail
coverage.  It is NumPy-only so the grid can be audited on a login/CPU node
before a long GPU calculation is released.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class CompositeGridSpec:
    gauss_order: int = 3
    core_sigma: float = 4.0
    tail_sigma: float = 6.0
    interval_sigma: float = 1.0

    def validate(self) -> None:
        if self.gauss_order < 1:
            raise ValueError("gauss_order must be positive")
        if self.core_sigma <= 0.0:
            raise ValueError("core_sigma must be positive")
        if self.tail_sigma <= self.core_sigma:
            raise ValueError("tail_sigma must be larger than core_sigma")
        if self.interval_sigma <= 0.0:
            raise ValueError("interval_sigma must be positive")


def normal_shock_states(
    mach: float,
    gamma: float = 5.0 / 3.0,
    rho1: float = 1.0,
    temperature1: float = 1.0,
) -> Dict[str, float]:
    """Return upstream/downstream normal-shock states in solver units."""
    a1 = math.sqrt(gamma * temperature1)
    u1 = float(mach) * a1
    ratio = ((gamma + 1.0) * mach**2) / ((gamma - 1.0) * mach**2 + 2.0)
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mach**2 - 1.0)
    return {
        "rho1": float(rho1),
        "u1": u1,
        "T1": float(temperature1),
        "rho2": float(rho1 * ratio),
        "u2": float(u1 / ratio),
        "T2": float(temperature1 * p2_p1 / ratio),
        "gamma": float(gamma),
        "M1": float(mach),
    }


def trapezoid_weights(nodes: np.ndarray) -> np.ndarray:
    """Trapezoidal weights for a strictly increasing, nonuniform 1-D grid."""
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 1 or nodes.size < 2:
        raise ValueError("nodes must be a one-dimensional array with at least two entries")
    delta = np.diff(nodes)
    if np.any(delta <= 0.0):
        raise ValueError("nodes must be strictly increasing")
    weights = np.empty_like(nodes)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    return weights


def _unique_sorted(parts: Iterable[np.ndarray], digits: int = 13) -> np.ndarray:
    joined = np.concatenate([np.asarray(part, dtype=np.float64) for part in parts])
    return np.unique(np.round(joined, decimals=digits))


def composite_axis_quadrature(
    centers: Iterable[float],
    thermal_sigmas: Iterable[float],
    spec: CompositeGridSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one composite Gauss--Legendre axis and positive weights.

    Every Maxwellian contributes break points spaced by ``interval_sigma`` in
    its thermal core.  Gauss--Legendre nodes are then placed independently in
    every interval of the merged partition.  This is substantially more
    accurate than a trapezoid rule on an irregular union of fine/coarse nodes.
    """
    spec.validate()
    centers = np.asarray(tuple(centers), dtype=np.float64)
    sigmas = np.asarray(tuple(thermal_sigmas), dtype=np.float64)
    if centers.shape != sigmas.shape or centers.size == 0:
        raise ValueError("centers and thermal_sigmas must have equal nonzero length")
    if np.any(sigmas <= 0.0):
        raise ValueError("thermal_sigmas must be positive")

    parts = []
    for center, sigma in zip(centers, sigmas):
        scaled = np.arange(
            -spec.core_sigma,
            spec.core_sigma + 0.5 * spec.interval_sigma,
            spec.interval_sigma,
            dtype=np.float64,
        )
        parts.append(center + sigma * scaled)
        parts.append(np.array([center - spec.tail_sigma * sigma, center + spec.tail_sigma * sigma]))
    breaks = _unique_sorted(parts)
    canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(spec.gauss_order)
    nodes = []
    weights = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        half = 0.5 * (right - left)
        midpoint = 0.5 * (right + left)
        nodes.append(midpoint + half * canonical_nodes)
        weights.append(half * canonical_weights)
    return np.concatenate(nodes), np.concatenate(weights)


def composite_velocity_axes(
    states: Dict[str, float],
    spec: CompositeGridSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return streamwise and two transverse composite velocity axes."""
    sigmas = (math.sqrt(states["T1"]), math.sqrt(states["T2"]))
    vx, _ = composite_axis_quadrature((states["u1"], states["u2"]), sigmas, spec)
    transverse, _ = composite_axis_quadrature((0.0, 0.0), sigmas, spec)
    return vx, transverse, transverse.copy()


def composite_velocity_quadrature(
    states: Dict[str, float],
    spec: CompositeGridSpec,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return composite axes and their matching one-dimensional weights."""
    sigmas = (math.sqrt(states["T1"]), math.sqrt(states["T2"]))
    vx, wx = composite_axis_quadrature((states["u1"], states["u2"]), sigmas, spec)
    vy, wy = composite_axis_quadrature((0.0, 0.0), sigmas, spec)
    return (vx, vy, vy.copy()), (wx, wy, wy.copy())


def grid_metadata(
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    spec: CompositeGridSpec,
) -> Dict[str, object]:
    vx, vy, vz = axes
    return {
        "mode": "composite",
        "quadrature": "composite_gauss_legendre",
        "gauss_order": spec.gauss_order,
        "core_sigma": spec.core_sigma,
        "tail_sigma": spec.tail_sigma,
        "interval_sigma": spec.interval_sigma,
        "shape": [int(vx.size), int(vy.size), int(vz.size)],
        "velocity_count": int(vx.size * vy.size * vz.size),
        "bounds": [[float(a[0]), float(a[-1])] for a in axes],
        "minimum_spacing": [float(np.min(np.diff(a))) for a in axes],
        "maximum_spacing": [float(np.max(np.diff(a))) for a in axes],
    }
