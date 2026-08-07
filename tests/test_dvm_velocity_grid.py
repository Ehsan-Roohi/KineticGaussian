from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dvm" / "src"))
sys.path.insert(0, str(REPO_ROOT / "dvm" / "scripts"))

from dvm_velocity_grid import (  # noqa: E402
    CompositeGridSpec,
    composite_velocity_quadrature,
    normal_shock_states,
    trapezoid_weights,
)
from audit_velocity_grid import audit_state  # noqa: E402


class DvmVelocityGridTest(unittest.TestCase):
    def test_nonuniform_trapezoid_weights_integrate_constant(self) -> None:
        nodes = np.array([-3.0, -1.0, -0.25, 0.5, 4.0])
        weights = trapezoid_weights(nodes)
        self.assertAlmostEqual(float(weights.sum()), 7.0, places=13)

    def test_composite_grid_resolves_mach12_equilibria(self) -> None:
        states = normal_shock_states(12.0)
        axes, axis_weights = composite_velocity_quadrature(states, CompositeGridSpec())
        self.assertLess(np.min(np.diff(axes[0])), 0.35)
        self.assertLess(np.min(np.diff(axes[1])), 0.35)
        for rho_key, u_key, t_key in (("rho1", "u1", "T1"), ("rho2", "u2", "T2")):
            metrics = audit_state(axes, axis_weights, states[rho_key], states[u_key], states[t_key])
            self.assertLess(metrics["sig_normalized"], 5.0e-3)
            self.assertLess(metrics["M400neq_normalized"], 5.0e-3)


if __name__ == "__main__":
    unittest.main()
