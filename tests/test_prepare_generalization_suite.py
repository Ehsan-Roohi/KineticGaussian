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

    def test_multiple_degrees_use_training_only_bounds_for_blind_extrapolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            case_values = [
                ("M1p5", 1.5),
                ("M2", 2.0),
                ("M2p5", 2.5),
                ("M3", 3.0),
                ("M4", 4.0),
                ("M5", 5.0),
                ("M6", 6.0),
                ("M8", 8.0),
                ("M12", 12.0),
            ]
            manifest.write_text(
                json.dumps(
                    {
                        "mach_bounds": [1.5, 12.0],
                        "cases": [
                            {"name": name, "mach": mach, "data_path": f"{name}.npz"}
                            for name, mach in case_values
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "generated"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "prepare_generalization_suite.py"),
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                    "--holdouts",
                    "M12",
                    "--capacities",
                    "512",
                    "--mach-degrees",
                    "2,3",
                    "--seeds",
                    "1234,2026,3407",
                    "--objectives",
                    "moment",
                    "--conditionings",
                    "all",
                    "--mach-bounds-source",
                    "training",
                    "--skip-baselines",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            task_paths = [
                Path(line) for line in (output / "tasks.txt").read_text().splitlines() if line
            ]
            self.assertEqual(len(task_paths), 6)
            configs = [json.loads(path.read_text()) for path in task_paths]
            self.assertEqual({cfg["model"]["mach_degree"] for cfg in configs}, {2, 3})
            for cfg in configs:
                self.assertEqual(cfg["mach_bounds"], [1.5, 8.0])
                self.assertEqual(cfg["mach_bounds_source"], "training")
                self.assertNotIn("M12", cfg["train_cases"])
                self.assertIn(f"_D{cfg['model']['mach_degree']}_", cfg["run_name"])


if __name__ == "__main__":
    unittest.main()
