from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]


def synthetic_case(path: Path, mach: float) -> None:
    x = np.linspace(-4.0, 4.0, 12, dtype=np.float32)
    axis_x = np.linspace(-5.0, 7.0, 7, dtype=np.float32)
    axis_t = np.linspace(-4.0, 4.0, 5, dtype=np.float32)
    vx, vy, vz = np.meshgrid(axis_x, axis_t, axis_t, indexing="ij")
    v = np.column_stack([vx.ravel(), vy.ravel(), vz.ravel()]).astype(np.float32)
    dv = float(axis_x[1] - axis_x[0]) * float(axis_t[1] - axis_t[0]) ** 2
    w = np.full(len(v), dv, dtype=np.float32)
    rho = (1.0 + 0.5 * (1.0 + np.tanh(x / 1.3))).astype(np.float32)
    ux = (0.45 * mach - 0.16 * np.tanh(x / 1.5)).astype(np.float32)
    temperature = (1.0 + 0.15 * mach + 0.20 * (1.0 + np.tanh(x / 1.6))).astype(np.float32)
    f = np.empty((len(x), len(v)), dtype=np.float32)
    for index in range(len(x)):
        c = v - np.array([ux[index], 0.0, 0.0], dtype=np.float32)
        c2 = np.sum(c * c, axis=1)
        prefactor = rho[index] / ((2.0 * np.pi * temperature[index]) ** 1.5)
        f[index] = prefactor * np.exp(-c2 / (2.0 * temperature[index]))
    qx = np.zeros_like(x)
    sig = np.zeros_like(x)
    np.savez_compressed(path, x=x, f=f, v=v, w=w, rho=rho, ux=ux, T=temperature, qx=qx, sig=sig)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="kinetic-gaussian-smoke-") as temporary:
        root = Path(temporary)
        cases = []
        for name, mach in (("M2p5", 2.5), ("M3", 3.0), ("M5", 5.0)):
            path = root / f"{name}_fullstate.npz"
            synthetic_case(path, mach)
            cases.append({"name": name, "mach": mach, "data_path": str(path), "moment_path": None})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"mach_bounds": [2.5, 5.0], "cases": cases}), encoding="utf-8")
        output = root / "runs"
        config = {
            "run_name": "smoke",
            "manifest_path": str(manifest),
            "output_dir": str(output),
            "device": "cpu",
            "seed": 7,
            "mach_bounds": [2.5, 5.0],
            "train_cases": ["M2p5", "M5"],
            "holdout_cases": ["M3"],
            "eval_cases": ["M3"],
            "data": {"coordinate_normalization": "shared_training"},
            "model": {
                "variant": "xvx",
                "mach_degree": 1,
                "conditioning": "amplitude",
                "num_kernels": 8,
                "init_samples": 64,
                "log_scale_min": -5.0,
                "log_scale_max": 0.0,
                "init_log_scale": -1.5,
                "init_log_amp": -4.0,
            },
            "sampling": {"x_batch": 2, "vel_per_x": 12, "uniform_vel_frac": 0.2, "mass_alpha": 0.7},
            "train": {"steps": 3, "lr": 0.001, "warmup_steps": 1, "log_every": 1, "save_every": 3, "lambda_moment": 0.0},
            "evaluation": {"moment_keys": ["rho", "ux", "T", "qx", "sig", "M400", "M400neq"]},
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        subprocess.run([sys.executable, "train_conditional_shocks.py", "--config", str(config_path)], cwd=REPO, check=True)
        checkpoint = output / "smoke" / "best.pt"
        subprocess.run(
            [
                sys.executable,
                "evaluate_conditional_shocks.py",
                "--config", str(config_path),
                "--checkpoint", str(checkpoint),
                "--x-stride", "3",
                "--x-chunk", "2",
                "--v-chunk", "64",
                "--sample-points", "128",
            ],
            cwd=REPO,
            check=True,
        )
        metrics_path = output / "smoke" / "eval" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["cases"]["M3"]["split"] == "holdout"
        assert metrics["cases"]["M3"]["sample_points"] == 128
        assert "M400neq" in metrics["cases"]["M3"]["moment_relative_l2"]
        checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert checkpoint_state["coordinate_state"]["coordinate_normalization"] == "shared_training"
        assert "common_normalizer" in checkpoint_state["coordinate_state"]
        print(f"SMOKE_OK {metrics_path}")


if __name__ == "__main__":
    main()
