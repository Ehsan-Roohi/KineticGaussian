#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch

from kgfr.conditional_models import ConditionalPhaseGaussianMixture
from kgfr.conditional_moments import moments_from_conditional_model
from kgfr.data import CoordinateNormalizer, ShockFullState
from kgfr.parametric_data import load_case_manifest, select_case_specs
from kgfr.plots import plot_ladder, plot_moment_profiles
from kgfr.utils import choose_device, load_json, set_seed


def model_from_config(cfg: dict) -> ConditionalPhaseGaussianMixture:
    model_cfg = cfg.get("model", {})
    return ConditionalPhaseGaussianMixture(
        num_kernels=int(model_cfg.get("num_kernels", 512)),
        variant=str(model_cfg.get("variant", "xvx")),
        mach_degree=int(model_cfg.get("mach_degree", 2)),
        conditioning=str(model_cfg.get("conditioning", "all")),
        log_scale_min=float(model_cfg.get("log_scale_min", -6.0)),
        log_scale_max=float(model_cfg.get("log_scale_max", -0.7)),
        init_log_scale=float(model_cfg.get("init_log_scale", -2.4)),
        init_log_amp=float(model_cfg.get("init_log_amp", -7.0)),
    )


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(prediction, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    denominator = np.linalg.norm(np.asarray(reference, dtype=np.float64)) + 1.0e-300
    return float(numerator / denominator)


def evaluate_moments(
    model: torch.nn.Module,
    mach_norm: float,
    data: ShockFullState,
    keys: Iterable[str],
    x_stride: int,
    x_chunk: int,
    v_chunk: int,
) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    keys = list(keys)
    indices = np.arange(0, data.Nx, x_stride, dtype=np.int64)
    prediction = {key: np.zeros(len(indices), dtype=np.float64) for key in keys}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), x_chunk):
            block = indices[start : start + x_chunk]
            ix = torch.from_numpy(block).to(device=next(model.parameters()).device, dtype=torch.long)
            values = moments_from_conditional_model(model, mach_norm, data, ix, v_chunk=v_chunk)
            for key in keys:
                if key in values:
                    prediction[key][start : start + len(block)] = values[key].detach().cpu().numpy()
    reference: Dict[str, np.ndarray] = {}
    for key in keys:
        normalized = "sig" if key == "sigma_xx" else key
        if normalized in data.moment_ref:
            reference[key] = np.asarray(data.moment_ref[normalized][indices], dtype=np.float64)
    errors = {key: relative_l2(prediction[key], reference[key]) for key in keys if key in reference}
    return data.x[indices], prediction, reference, errors


def evaluate_phase_samples(
    model: torch.nn.Module,
    mach_norm: float,
    data: ShockFullState,
    sample_points: int,
    device: torch.device,
) -> dict:
    squared_log_error = 0.0
    squared_f_error = 0.0
    squared_f_reference = 0.0
    weight_sum = 0.0
    completed = 0
    batch_points = 4096
    model.eval()
    with torch.no_grad():
        while completed < sample_points:
            count = min(batch_points, sample_points - completed)
            x_batch = min(16, count)
            vel_per_x = max(1, int(np.ceil(count / x_batch)))
            z, target, sample_weight = data.sample_phase_batch(
                x_batch=x_batch,
                vel_per_x=vel_per_x,
                uniform_vel_frac=0.25,
                mass_alpha=0.60,
                device=device,
            )
            z, target, sample_weight = z[:count], target[:count], sample_weight[:count]
            pred = model(torch.tensor(mach_norm, device=device), z)
            delta = torch.clamp(pred - target, -40.0, 40.0)
            squared_log_error += float(torch.sum(sample_weight * delta * delta).cpu())
            weight_sum += float(torch.sum(sample_weight).cpu())
            f_pred = torch.exp(torch.clamp(pred, -80.0, 30.0))
            f_ref = torch.exp(torch.clamp(target, -80.0, 30.0))
            squared_f_error += float(torch.sum((f_pred - f_ref) ** 2).cpu())
            squared_f_reference += float(torch.sum(f_ref * f_ref).cpu())
            completed += count
    return {
        "sample_points": int(completed),
        "weighted_logf_rmse": float(np.sqrt(squared_log_error / max(weight_sum, 1.0e-300))),
        "sampled_f_relative_l2": float(np.sqrt(squared_f_error / max(squared_f_reference, 1.0e-300))),
    }


def write_profile_csv(path: Path, x: np.ndarray, prediction: dict, reference: dict) -> None:
    keys = [key for key in prediction if key in reference]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", *[f"{key}_pred" for key in keys], *[f"{key}_ref" for key in keys]])
        writer.writerows(zip(x, *[prediction[key] for key in keys], *[reference[key] for key in keys]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out Mach shocks for the conditional Gaussian model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--x-stride", type=int, default=5)
    parser.add_argument("--x-chunk", type=int, default=16)
    parser.add_argument("--v-chunk", type=int, default=2048)
    parser.add_argument("--sample-points", type=int, default=50000)
    parser.add_argument(
        "--output-subdir",
        default="eval",
        help="Run-directory subfolder for metrics and plots (use a new name to preserve prior evaluation output)",
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 1234)) + 991)
    device = choose_device(cfg.get("device", "cuda"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    specs, manifest = load_case_manifest(cfg["manifest_path"])
    requested = args.cases or list(cfg.get("eval_cases", cfg.get("holdout_cases", [])))
    if not requested:
        requested = [case.name for case in specs]
    selected = select_case_specs(specs, requested)
    mach_bounds = checkpoint.get("coordinate_state", {}).get("mach_bounds")
    if mach_bounds is None:
        mach_bounds = cfg.get("mach_bounds", manifest.get("mach_bounds"))
    mach_center = 0.5 * (float(mach_bounds[0]) + float(mach_bounds[1]))
    mach_halfwidth = 0.5 * (float(mach_bounds[1]) - float(mach_bounds[0]))
    train_names = set(cfg.get("train_cases", []))
    holdout_names = set(cfg.get("holdout_cases", []))

    run_dir = Path(cfg.get("output_dir", "runs/conditional")) / cfg["run_name"]
    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_count = model.parameter_count()
    representation_bytes = int(sum(t.numel() * t.element_size() for t in model.state_dict().values()))
    summary = {
        "run_name": cfg["run_name"],
        "checkpoint": str(Path(args.checkpoint)),
        "model": model.metadata(),
        "representation_bytes_float32": representation_bytes,
        "cases": {},
    }

    keys = list(cfg.get("evaluation", {}).get("moment_keys", ["rho", "ux", "T", "qx", "sig", "M300", "M400"]))
    if "M400" in keys and "M400neq" not in keys:
        keys.append("M400neq")
    coordinate_state = checkpoint.get("coordinate_state", {})
    normalizer_state = coordinate_state.get("common_normalizer")
    shared_normalizer = None
    if normalizer_state is not None:
        shared_normalizer = CoordinateNormalizer.from_state_dict(normalizer_state)
    for spec in selected:
        print(f"[kinetic-gaussian] evaluating {spec.name} (M={spec.mach:g})", flush=True)
        data = ShockFullState(
            spec.data_path,
            moment_path=spec.moment_path,
            f_floor=float(cfg.get("sampling", {}).get("f_floor", 1.0e-35)),
            normalizer=shared_normalizer,
        )
        mach_norm = (spec.mach - mach_center) / mach_halfwidth
        x, prediction, reference, errors = evaluate_moments(
            model,
            mach_norm,
            data,
            keys,
            x_stride=max(1, args.x_stride),
            x_chunk=max(1, args.x_chunk),
            v_chunk=max(1, args.v_chunk),
        )
        phase_metrics = evaluate_phase_samples(model, mach_norm, data, args.sample_points, device)
        split = "holdout" if spec.name in holdout_names else ("train" if spec.name in train_names else "unspecified")
        case_metrics = {
            "name": spec.name,
            "mach": spec.mach,
            "normalized_mach": mach_norm,
            "split": split,
            "Nx": data.Nx,
            "Nv": data.Nv,
            "raw_phase_values": int(data.Nx * data.Nv),
            "raw_f_bytes": int(data.f.nbytes),
            "parameter_count": parameter_count,
            "nominal_compression": float(data.Nx * data.Nv / parameter_count),
            "moment_relative_l2": errors,
            **phase_metrics,
        }
        summary["cases"][spec.name] = case_metrics
        write_profile_csv(output_dir / f"moments_{spec.name}.csv", x, prediction, reference)
        plot_moment_profiles(x, prediction, reference, errors.keys(), output_dir / f"moments_{spec.name}.png")
        plot_ladder(errors, output_dir / f"ladder_{spec.name}.png")
        print(json.dumps(case_metrics, indent=2), flush=True)

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[kinetic-gaussian] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
