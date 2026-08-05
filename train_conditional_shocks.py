#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from kgfr.conditional_models import ConditionalPhaseGaussianMixture, optimizer_for_conditional_model
from kgfr.conditional_moments import sampled_conditional_moment_loss
from kgfr.parametric_data import ParametricShockDataset, load_case_manifest, select_case_specs
from kgfr.plots import plot_training_history
from kgfr.utils import choose_device, ensure_dir, load_json, save_json, set_seed


def logf_loss_fn(pred: torch.Tensor, target: torch.Tensor, sample_w: torch.Tensor, cfg: dict) -> torch.Tensor:
    clip = float(cfg.get("logf_error_clip", 30.0))
    error = torch.clamp(pred - target, -clip, clip)
    if cfg.get("logf_loss", "huber") == "huber":
        absolute = torch.abs(error)
        per_point = torch.where(absolute < 1.0, 0.5 * error * error, absolute - 0.5)
    else:
        per_point = error * error
    return torch.mean(sample_w * per_point)


def model_from_config(cfg: dict) -> ConditionalPhaseGaussianMixture:
    model_cfg = cfg.get("model", {})
    return ConditionalPhaseGaussianMixture(
        num_kernels=int(model_cfg.get("num_kernels", 512)),
        variant=str(model_cfg.get("variant", "xvx")),
        mach_degree=int(model_cfg.get("mach_degree", 2)),
        log_scale_min=float(model_cfg.get("log_scale_min", -6.0)),
        log_scale_max=float(model_cfg.get("log_scale_max", -0.7)),
        init_log_scale=float(model_cfg.get("init_log_scale", -2.4)),
        init_log_amp=float(model_cfg.get("init_log_amp", -7.0)),
    )


def save_checkpoint(
    path: Path,
    model: ConditionalPhaseGaussianMixture,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: dict,
    data: ParametricShockDataset,
    step: int,
    best_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "best_loss": float(best_loss),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_meta": model.metadata(),
            "coordinate_state": data.coordinate_state(),
            "config": cfg,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


@torch.no_grad()
def balanced_probe_loss(
    model: ConditionalPhaseGaussianMixture,
    data: ParametricShockDataset,
    train_cfg: dict,
    sampling_cfg: dict,
    device: torch.device,
    seed: int,
) -> float:
    """Deterministic equal-case probe used only for checkpoint selection."""
    numpy_state = np.random.get_state()
    np.random.seed(seed)
    was_training = model.training
    model.eval()
    values = []
    try:
        for spec in data.specs:
            mach, z, target, sample_w, _ = data.sample_phase_batch(
                x_batch=int(sampling_cfg.get("probe_x_batch", 12)),
                vel_per_x=int(sampling_cfg.get("probe_vel_per_x", 256)),
                uniform_vel_frac=float(sampling_cfg.get("uniform_vel_frac", 0.15)),
                mass_alpha=float(sampling_cfg.get("mass_alpha", 0.55)),
                device=device,
                case_name=spec.name,
            )
            values.append(float(logf_loss_fn(model(mach, z), target, sample_w, train_cfg).cpu()))
    finally:
        np.random.set_state(numpy_state)
        model.train(was_training)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Mach-conditional positive Gaussian shock representation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Optional last.pt checkpoint")
    args = parser.parse_args()

    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 1234)))
    device = choose_device(cfg.get("device", "cuda"))
    manifest_specs, manifest = load_case_manifest(cfg["manifest_path"])
    all_names = [case.name for case in manifest_specs]
    holdout_names = list(cfg.get("holdout_cases", []))
    train_names = list(cfg.get("train_cases", [name for name in all_names if name not in holdout_names]))
    if set(train_names) & set(holdout_names):
        raise ValueError("train_cases and holdout_cases must be disjoint")
    train_specs = select_case_specs(manifest_specs, train_names)
    mach_bounds = cfg.get("mach_bounds", manifest.get("mach_bounds"))
    if mach_bounds is None:
        all_machs = [case.mach for case in manifest_specs]
        mach_bounds = [min(all_machs), max(all_machs)]

    out_root = Path(cfg.get("output_dir", "runs/conditional"))
    run_dir = ensure_dir(out_root / cfg["run_name"])
    save_json(cfg, run_dir / "config.json")
    sampling_cfg = cfg.get("sampling", {})
    data = ParametricShockDataset(
        train_specs,
        mach_bounds=mach_bounds,
        f_floor=float(sampling_cfg.get("f_floor", 1.0e-35)),
    )
    print(data.summary(), flush=True)

    model = model_from_config(cfg).to(device)
    print(f"[kinetic-gaussian] model: {model.metadata()}", flush=True)
    init_count = int(cfg.get("model", {}).get("init_samples", max(10000, 20 * model.num_kernels)))
    _, z_init = data.sample_init_points(init_count, device=device)
    model.initialize_centers_from_samples(z_init)
    del z_init
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_cfg = cfg.get("train", {})
    optimizer = optimizer_for_conditional_model(model, train_cfg)
    steps = int(train_cfg.get("steps", 80000))
    warmup = min(int(train_cfg.get("warmup_steps", 1000)), max(0, steps - 1))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))

    def schedule_factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1.0e-3, float(step + 1) / warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_factor)
    start_step, best_loss = 1, math.inf
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_step = int(checkpoint["step"]) + 1
        best_loss = float(checkpoint.get("best_loss", math.inf))
        print(f"[kinetic-gaussian] resumed from step {start_step - 1}", flush=True)

    x_batch = int(sampling_cfg.get("x_batch", 24))
    vel_per_x = int(sampling_cfg.get("vel_per_x", 384))
    uniform_vel_frac = float(sampling_cfg.get("uniform_vel_frac", 0.15))
    mass_alpha = float(sampling_cfg.get("mass_alpha", 0.55))
    lambda_moment = float(train_cfg.get("lambda_moment", 0.0))
    moment_every = max(1, int(train_cfg.get("moment_every", 10)))
    moment_x_count = int(train_cfg.get("moment_x_count", 12))
    moment_keys = list(train_cfg.get("moment_keys", ["rho", "ux", "T", "qx", "sig", "M300"]))
    log_every = int(train_cfg.get("log_every", 50))
    save_every = int(train_cfg.get("save_every", 2000))
    eval_every = int(train_cfg.get("eval_every", 1000))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    history_path = run_dir / "history.csv"
    if start_step == 1 or not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ["step", "case", "loss_total", "loss_logf", "loss_moment", "loss_probe", "lr", "seconds"]
            )

    started = time.time()
    progress = trange(start_step, steps + 1, desc=cfg["run_name"])
    for step in progress:
        model.train()
        mach, z, target_logf, sample_w, case_name = data.sample_phase_batch(
            x_batch=x_batch,
            vel_per_x=vel_per_x,
            uniform_vel_frac=uniform_vel_frac,
            mass_alpha=mass_alpha,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(mach, z)
        loss_logf = logf_loss_fn(prediction, target_logf, sample_w, train_cfg)
        loss_moment = torch.zeros((), device=device)
        if lambda_moment > 0.0 and step % moment_every == 0:
            case = data.cases[case_name]
            ix_np = case.sample_x_indices(moment_x_count)
            ix = torch.from_numpy(ix_np).to(device=device, dtype=torch.long)
            loss_moment, _ = sampled_conditional_moment_loss(
                model,
                mach,
                case,
                ix,
                moment_keys,
                vel_count=int(train_cfg.get("moment_vel_count", 1536)),
                uniform_frac=float(train_cfg.get("moment_uniform_vel_frac", 0.10)),
                mass_alpha=float(train_cfg.get("moment_mass_alpha", 0.55)),
            )
        loss = loss_logf + lambda_moment * loss_moment
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        probe_value = math.nan
        if step == start_step or step % eval_every == 0 or step == steps:
            probe_value = balanced_probe_loss(
                model,
                data,
                train_cfg,
                sampling_cfg,
                device,
                seed=int(cfg.get("seed", 1234)) + 104729,
            )
            if probe_value < best_loss:
                best_loss = probe_value
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, cfg, data, step, best_loss)
        if step % log_every == 0 or step == start_step:
            row = [
                step,
                case_name,
                loss_value,
                float(loss_logf.detach().cpu()),
                float(loss_moment.detach().cpu()),
                probe_value,
                optimizer.param_groups[0]["lr"],
                time.time() - started,
            ]
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(row)
            progress.set_postfix(loss=f"{loss_value:.3e}", case=case_name)
        if step % save_every == 0:
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, cfg, data, step, best_loss)
            plot_training_history(history_path, run_dir / "training_history.png")

    save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, cfg, data, steps, best_loss)
    plot_training_history(history_path, run_dir / "training_history.png")
    print(f"[kinetic-gaussian] finished; best balanced probe loss={best_loss:.6e}", flush=True)
    print(f"[kinetic-gaussian] run directory: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
