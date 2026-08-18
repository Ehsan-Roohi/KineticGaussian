from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_fullstate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-1.0, 1.0, 3, dtype=np.float32)
    v = np.zeros((4, 3), dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    f = np.ones((3, 4), dtype=np.float32)
    macro = np.ones(3, dtype=np.float32)
    np.savez(path, x=x, f=f, v=v, w=w, rho=macro, ux=macro, T=macro, qx=macro, sig=macro)


def write_moments(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ones(3, dtype=np.float32)
    np.savez(path, M300_neq=values, M400_raw=values, Rxx_neq=values)


class FullMachManifestTest(unittest.TestCase):
    def test_prefers_authoritative_double_fullstate_and_rejects_lite_and_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "ref" / "mach_sweep_extra" / "M8"
            data = case_dir / "standing_M8_hmom_x120_nx2800_v171_31_31_vmax32_fullstate_fullstate.npz"
            companion = case_dir / "standing_M8_hmom_x120_nx2800_v171_31_31_vmax32_fullstate.npz"
            write_fullstate(data)
            write_moments(companion)
            write_fullstate(
                root
                / "ref"
                / "mach_sweep_extra"
                / "M8lite"
                / "standing_M8lite_hmom_fullstate_fullstate.npz"
            )
            write_fullstate(
                case_dir
                / "smoke_200_steps_20260613"
                / "standing_M8_hmom_x120_nx2800_v171_31_31_vmax32_fullstate_fullstate.npz"
            )

            output = root / "manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "make_full_mach_manifest.py"),
                    "--bgk-root",
                    str(root),
                    "--output",
                    str(output),
                    "--tags",
                    "M8",
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            payload = json.loads(output.read_text())
            self.assertEqual(payload["mach_bounds"], [8.0, 8.0])
            self.assertEqual(len(payload["cases"]), 1)
            case = payload["cases"][0]
            self.assertEqual(Path(case["data_path"]), data.resolve())
            self.assertEqual(Path(case["moment_path"]), companion.resolve())
            self.assertIn("M300_neq", case["high_moment_keys"])


if __name__ == "__main__":
    unittest.main()
