from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_result(path: Path, perturbation: float, temporal_converged: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5.0, 5.0, 101)
    base = 1.0 + np.tanh(x)
    np.savez(
        path,
        x_mfp=x,
        rho=base * (1.0 + perturbation),
        ux=(3.0 - 0.5 * base) * (1.0 + perturbation),
        T=(1.0 + 0.2 * base) * (1.0 + perturbation),
        qx_neq_discrete=np.exp(-x**2) * (1.0 + perturbation),
        sig_neq_discrete=-np.exp(-0.5 * x**2) * (1.0 + perturbation),
        M300_neq=0.5 * np.exp(-x**2) * (1.0 + perturbation),
        M400_neq=0.75 * np.exp(-x**2) * (1.0 + perturbation),
        temporal_gate_enabled=np.array(True),
        temporal_converged=np.array(temporal_converged),
        temporal_completed_steps=np.array(100),
        temporal_consecutive_passes=np.array(3 if temporal_converged else 0),
        temporal_latest_macro_relative_l2_max=np.array(1.0e-7),
        temporal_latest_noneq_relative_l2_max=np.array(1.0e-5),
    )


class JcpGridConvergenceTest(unittest.TestCase):
    def test_medium_fine_gate_passes_small_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for level, perturbation in (("coarse", 0.02), ("medium", 0.002), ("fine", 0.0)):
                path = root / f"M6_{level}.npz"
                write_result(path, perturbation)
                rows.append({"case": "M6", "level": level, "moments_path": str(path)})
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"temporal_convergence": {"required_consecutive_checks": 3}, "convergence": rows})
            )
            report = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "dvm" / "scripts" / "check_jcp_grid_convergence.py"),
                    "--suite-manifest",
                    str(manifest),
                    "--output",
                    str(report),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            self.assertTrue(json.loads(report.read_text())["pass"])

    def test_temporal_failure_blocks_grid_certification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for level, converged in (("coarse", True), ("medium", False), ("fine", True)):
                path = root / f"M12_{level}.npz"
                write_result(path, 0.0, temporal_converged=converged)
                rows.append({"case": "M12", "level": level, "moments_path": str(path)})
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"temporal_convergence": {"required_consecutive_checks": 3}, "convergence": rows})
            )
            report = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "dvm" / "scripts" / "check_jcp_grid_convergence.py"),
                    "--suite-manifest",
                    str(manifest),
                    "--output",
                    str(report),
                ],
                cwd=REPO_ROOT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            parsed = json.loads(report.read_text())
            self.assertFalse(parsed["pass"])
            self.assertFalse(parsed["cases"]["M12"]["temporal"]["medium"]["pass"])


if __name__ == "__main__":
    unittest.main()
