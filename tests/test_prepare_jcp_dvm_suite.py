from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PrepareJcpDvmSuiteTest(unittest.TestCase):
    def test_generates_six_convergence_and_three_production_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "generated"
            data = root / "data"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "dvm" / "scripts" / "prepare_jcp_dvm_suite.py"),
                    "--config",
                    str(REPO_ROOT / "dvm" / "configs" / "jcp_high_mach_cases.json"),
                    "--output-dir",
                    str(generated),
                    "--data-root",
                    str(data),
                ],
                cwd=REPO_ROOT,
                check=True,
            )
            manifest = json.loads((generated / "suite_manifest.json").read_text())
            self.assertEqual(len(manifest["convergence"]), 6)
            self.assertEqual(len(manifest["production"]), 3)
            self.assertEqual({row["case"] for row in manifest["convergence"]}, {"M6", "M12"})
            self.assertEqual({row["case"] for row in manifest["production"]}, {"M7", "M8", "M10"})
            all_rows = manifest["convergence"] + manifest["production"]
            self.assertEqual(len({row["fullstate_path"] for row in all_rows}), 9)
            temporal = manifest["temporal_convergence"]
            self.assertEqual(temporal["required_consecutive_checks"], 3)
            for row in all_rows:
                self.assertEqual(row["temporal_convergence"], temporal)


if __name__ == "__main__":
    unittest.main()
