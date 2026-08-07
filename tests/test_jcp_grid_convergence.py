from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_result(path: Path, perturbation: float) -> None:
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
            manifest.write_text(json.dumps({"convergence": rows}))
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


if __name__ == "__main__":
    unittest.main()

