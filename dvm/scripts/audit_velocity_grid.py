#!/usr/bin/env python
"""Certify high-Mach DVM velocity quadrature before expensive GPU runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dvm" / "src"))

from dvm_velocity_grid import (  # noqa: E402
    CompositeGridSpec,
    composite_velocity_quadrature,
    grid_metadata,
    normal_shock_states,
    trapezoid_weights,
)


DEFAULT_LEVELS = {
    "coarse": CompositeGridSpec(gauss_order=2, core_sigma=3.0, tail_sigma=6.0),
    "medium": CompositeGridSpec(gauss_order=3, core_sigma=4.0, tail_sigma=6.0),
    "fine": CompositeGridSpec(gauss_order=3, core_sigma=5.0, tail_sigma=7.0),
}


def _axis_integrals(nodes: np.ndarray, weights: np.ndarray, center: float, temperature: float) -> Tuple[np.ndarray, ...]:
    density = np.exp(-0.5 * (nodes - center) ** 2 / temperature) / math.sqrt(2.0 * math.pi * temperature)
    weighted = density * weights
    return tuple(np.sum(weighted * nodes**power) for power in range(5))


def _discrete_mean_temperature(
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_weights: Tuple[np.ndarray, np.ndarray, np.ndarray],
    exponent_u: float,
    exponent_temperature: float,
) -> Tuple[float, float]:
    ix = _axis_integrals(axes[0], axis_weights[0], exponent_u, exponent_temperature)
    iy = _axis_integrals(axes[1], axis_weights[1], 0.0, exponent_temperature)
    iz = _axis_integrals(axes[2], axis_weights[2], 0.0, exponent_temperature)
    mean = ix[1] / ix[0]
    energy = ix[2] / ix[0] + iy[2] / iy[0] + iz[2] / iz[0]
    temperature = (energy - mean**2) / 3.0
    return float(mean), float(temperature)


def _conservative_parameters(
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_weights: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_u: float,
    target_temperature: float,
    iterations: int = 12,
) -> Tuple[float, float]:
    u = float(target_u)
    log_temperature = math.log(float(target_temperature))
    for _ in range(iterations):
        temperature = math.exp(log_temperature)
        mean, discrete_temperature = _discrete_mean_temperature(axes, axis_weights, u, temperature)
        residual = np.array([mean - target_u, discrete_temperature - target_temperature])
        if np.linalg.norm(residual, ord=np.inf) < 2.0e-13:
            break
        du_eps = max(1.0e-6, 1.0e-6 * abs(u))
        dl_eps = 1.0e-6
        mean_u, temp_u = _discrete_mean_temperature(axes, axis_weights, u + du_eps, temperature)
        mean_l, temp_l = _discrete_mean_temperature(axes, axis_weights, u, math.exp(log_temperature + dl_eps))
        jacobian = np.array(
            [
                [(mean_u - mean) / du_eps, (mean_l - mean) / dl_eps],
                [(temp_u - discrete_temperature) / du_eps, (temp_l - discrete_temperature) / dl_eps],
            ]
        )
        delta = np.linalg.solve(jacobian, -residual)
        u += float(np.clip(delta[0], -0.5, 0.5))
        log_temperature += float(np.clip(delta[1], -0.5, 0.5))
    return u, math.exp(log_temperature)


def audit_state(
    axes: Tuple[np.ndarray, np.ndarray, np.ndarray],
    axis_weights: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rho: float,
    target_u: float,
    target_temperature: float,
) -> Dict[str, float]:
    exponent_u, exponent_temperature = _conservative_parameters(axes, axis_weights, target_u, target_temperature)
    raw_x = _axis_integrals(axes[0], axis_weights[0], exponent_u, exponent_temperature)
    raw_y = _axis_integrals(axes[1], axis_weights[1], 0.0, exponent_temperature)
    raw_z = _axis_integrals(axes[2], axis_weights[2], 0.0, exponent_temperature)
    normalization = raw_x[0] * raw_y[0] * raw_z[0]
    scale = rho / normalization

    mean = raw_x[1] / raw_x[0]
    # Recompute central one-dimensional moments about the discrete mean.
    def central(nodes: np.ndarray, weights: np.ndarray, center: float, temperature: float, shift: float) -> Tuple[float, ...]:
        density = np.exp(-0.5 * (nodes - center) ** 2 / temperature) / math.sqrt(2.0 * math.pi * temperature)
        weighted = density * weights
        return tuple(float(np.sum(weighted * (nodes - shift) ** power)) for power in range(5))

    cx = central(axes[0], axis_weights[0], exponent_u, exponent_temperature, mean)
    cy = central(axes[1], axis_weights[1], 0.0, exponent_temperature, 0.0)
    cz = central(axes[2], axis_weights[2], 0.0, exponent_temperature, 0.0)
    rho_d = scale * cx[0] * cy[0] * cz[0]
    m2x = scale * cx[2] * cy[0] * cz[0]
    m2y = scale * cx[0] * cy[2] * cz[0]
    m2z = scale * cx[0] * cy[0] * cz[2]
    temperature_d = (m2x + m2y + m2z) / (3.0 * rho_d)
    qx = 0.5 * scale * (
        cx[3] * cy[0] * cz[0]
        + cx[1] * cy[2] * cz[0]
        + cx[1] * cy[0] * cz[2]
    )
    sig = m2x - (m2x + m2y + m2z) / 3.0
    m300 = scale * cx[3] * cy[0] * cz[0]
    m400 = scale * cx[4] * cy[0] * cz[0]
    return {
        "rho_relative_error": abs(rho_d - rho) / rho,
        "ux_relative_error": abs(mean - target_u) / max(abs(target_u), 1.0),
        "T_relative_error": abs(temperature_d - target_temperature) / target_temperature,
        "qx_normalized": abs(qx) / (rho * target_temperature**1.5),
        "sig_normalized": abs(sig) / (rho * target_temperature),
        "M300_normalized": abs(m300) / (rho * target_temperature**1.5),
        "M400neq_normalized": abs(m400 - 3.0 * rho * target_temperature**2) / (rho * target_temperature**2),
        "exponent_u": exponent_u,
        "exponent_T": exponent_temperature,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machs", default="6,7,8,10,12")
    parser.add_argument("--output-dir", default="dvm/audit")
    parser.add_argument("--noneq-tolerance", type=float, default=5.0e-3)
    parser.add_argument("--low-moment-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--require-level", default="medium", choices=tuple(DEFAULT_LEVELS))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    payload = {"thresholds": vars(args), "levels": {}, "pass": True}
    for level, spec in DEFAULT_LEVELS.items():
        payload["levels"][level] = {}
        for mach in [float(token) for token in args.machs.split(",") if token.strip()]:
            states = normal_shock_states(mach)
            axes, axis_weights = composite_velocity_quadrature(states, spec)
            metadata = grid_metadata(axes, spec)
            case = {"grid": metadata, "states": {}}
            for side, rho_key, u_key, t_key in (
                ("upstream", "rho1", "u1", "T1"),
                ("downstream", "rho2", "u2", "T2"),
            ):
                metrics = audit_state(axes, axis_weights, states[rho_key], states[u_key], states[t_key])
                low_max = max(metrics[key] for key in ("rho_relative_error", "ux_relative_error", "T_relative_error"))
                noneq_max = max(metrics[key] for key in ("qx_normalized", "sig_normalized", "M300_normalized", "M400neq_normalized"))
                metrics["pass"] = bool(low_max <= args.low_moment_tolerance and noneq_max <= args.noneq_tolerance)
                case["states"][side] = metrics
                rows.append({"level": level, "mach": mach, "side": side, **metadata, **metrics})
            case["pass"] = bool(all(item["pass"] for item in case["states"].values()))
            payload["levels"][level][f"M{mach:g}"] = case
            if level == args.require_level and not case["pass"]:
                payload["pass"] = False

    json_path = output_dir / "velocity_grid_audit.json"
    csv_path = output_dir / "velocity_grid_audit.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for level in DEFAULT_LEVELS:
        for name, case in payload["levels"][level].items():
            worst = max(
                max(state[key] for key in ("qx_normalized", "sig_normalized", "M300_normalized", "M400neq_normalized"))
                for state in case["states"].values()
            )
            print(f"{level:6s} {name:4s} Nv={case['grid']['velocity_count']:8d} worst_eq_defect={100*worst:9.5f}% pass={case['pass']}")
    print(f"WROTE: {json_path}")
    print(f"WROTE: {csv_path}")
    if not payload["pass"]:
        raise SystemExit(f"Required level {args.require_level!r} failed velocity-grid certification")


if __name__ == "__main__":
    main()
