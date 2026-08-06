from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareGeneralizationSuiteTest(unittest.TestCase):
    def test_custom_run_prefix_is_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "mach_bounds": [2.5, 5.0],
                        "cases": [
                            {"name": "M2p5", "mach": 2.5, "data_path": "M2p5.npz"},
                            {"name": "M3", "mach": 3.0, "data_path": "M3.npz"},
                            {"name": "M5", "mach": 5.0, "data_path": "M5.npz"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "generated"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/prepare_generalization_suite.py"),
                    "--manifest", str(manifest),
                    "--output-dir", str(output),
                    "--holdouts", "M3",
                    "--capacities", "256",
                    "--seeds", "1234,2026,3407",
                    "--objectives", "moment",
                    "--conditionings", "all",
                    "--coordinate-normalization", "shared_training",
                    "--run-prefix", "ablation_all_shared",
                    "--skip-baselines",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            task_paths = [
                Path(line) for line in (output / "tasks.txt").read_text().splitlines() if line
            ]
            self.assertEqual(len(task_paths), 3)
            configs = [json.loads(path.read_text()) for path in task_paths]
            self.assertEqual(len({cfg["run_name"] for cfg in configs}), 3)
            for cfg in configs:
                self.assertTrue(cfg["run_name"].startswith("ablation_all_shared_"))
                self.assertEqual(cfg["model"]["conditioning"], "all")
                self.assertEqual(cfg["data"]["coordinate_normalization"], "shared_training")
                self.assertEqual(cfg["model"]["num_kernels"], 256)


if __name__ == "__main__":
    unittest.main()
