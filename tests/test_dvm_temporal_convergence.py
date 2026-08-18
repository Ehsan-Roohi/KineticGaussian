from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dvm" / "src"))

from dvm_temporal_convergence import TemporalConvergenceTracker, relative_change


def profiles(scale: float = 1.0) -> dict[str, np.ndarray]:
    x = np.linspace(-2.0, 2.0, 17)
    return {
        "rho": scale * (1.5 + 0.25 * np.tanh(x)),
        "ux": scale * (2.0 - 0.1 * np.tanh(x)),
        "T": scale * (1.0 + 0.2 * np.tanh(x)),
        "qx": scale * np.exp(-(x**2)),
        "sig": -scale * np.exp(-0.5 * x**2),
        "M300": 0.5 * scale * np.exp(-(x**2)),
        "M400neq": 0.75 * scale * np.exp(-(x**2)),
    }


class DvmTemporalConvergenceTest(unittest.TestCase):
    def test_relative_change_is_symmetric(self) -> None:
        a = np.array([1.0, 2.0])
        b = np.array([1.1, 2.2])
        self.assertAlmostEqual(relative_change(a, b), relative_change(b, a))

    def test_requires_consecutive_eligible_checks(self) -> None:
        tracker = TemporalConvergenceTracker(1.0e-3, 1.0e-3, 2, min_step=20)
        tracker.update(10, profiles(1.0))
        first = tracker.update(20, profiles(1.0001))
        second = tracker.update(30, profiles(1.00015))
        self.assertTrue(first["passed"])
        self.assertTrue(second["passed"])
        self.assertTrue(tracker.converged)
        self.assertEqual(int(tracker.metadata(30)["temporal_consecutive_passes"]), 2)

    def test_large_change_resets_streak(self) -> None:
        tracker = TemporalConvergenceTracker(1.0e-3, 1.0e-3, 2, min_step=0)
        tracker.update(1, profiles(1.0))
        tracker.update(2, profiles(1.0001))
        failed = tracker.update(3, profiles(1.2))
        self.assertFalse(failed["passed"])
        self.assertEqual(tracker.consecutive_passes, 0)
        self.assertFalse(tracker.converged)

    def test_zero_required_checks_preserves_legacy_mode(self) -> None:
        tracker = TemporalConvergenceTracker(0.0, 0.0, 0, min_step=0)
        tracker.update(1, profiles())
        tracker.update(2, profiles())
        self.assertFalse(tracker.enabled)
        self.assertFalse(tracker.converged)


if __name__ == "__main__":
    unittest.main()
