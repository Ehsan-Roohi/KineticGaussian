from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from tqdm import trange

from kgfr.data import ShockFullState
from kgfr.models import PhaseGaussianMixture, optimizer_for_model
from kgfr.moments import moment_loss, moment_loss_sampled
from kgfr.plots import plot_training_history
from kgfr.utils import choose_device, ensure_dir, load_json, save_json, set_seed


def logf_loss_fn(pred: torch.Tensor, target: torch.Tensor, sample_w: torch.Tensor, cfg: dict) -> torch.Tensor:
    clip = float(cfg.get("logf_error_clip", 30.0))
    err = torch.clamp(pred - target, -clip, clip)
    if cfg.get("logf_loss", "mse") == "huber":
        abs_err = torch.abs(err)
        loss = torch.where(abs_err < 1.0, 0.5 * err * err, abs_err - 0.5)
    else:
        loss = err * err
    return torch.mean(sample_w * loss)


def save_checkpoint(path: Path, model: PhaseGaussianMixture, cfg: dict, data: ShockFullState, step: int, best_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_loss": best_loss,
            "model_state": model.state_dict(),
            "model_meta": model.metadata(),
            "coord_normalizer": data.norm.state_dict(),
            "config": cfg,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 1)))
    device = choose_device(cfg.get("device", "cuda"))
    out_root = Path(cfg.get("output_dir", "runs"))
    run_dir = ensure_dir(out_root / cfg["run_name"])
    save_json(cfg, run_dir / "config.json")

    sampling_cfg = cfg.get("sampling", {})
    data = ShockFullState(cfg["data_path"], moment_path=cfg.get("moment_path"), f_floor=float(sampling_cfg.get("f_floor", 1e-35)))
    print(data.summary())

    model_cfg = cfg.get("model", {})
    model = PhaseGaussianMixture(
        num_kernels=int(model_cfg.get("num_kernels", 256)),
        variant=model_cfg.get("variant", "diag"),
        log_scale_min=float(model_cfg.get("log_scale_min", -8.0)),
        log_scale_max=float(model_cfg.get("log_scale_max", 1.0)),
        init_log_scale=float(model_cfg.get("init_log_scale", -1.2)),
        init_log_amp=float(model_cfg.get("init_log_amp", -8.0)),
    ).to(device)
    print(f"[kgfr] model: {model.metadata()}")

    n_init = int(model_cfg.get("init_samples", max(10000, 10 * model.num_kernels)))
    print(f"[kgfr] initializing centers from {n_init} sampled phase points")
    z_init = data.sample_init_points(n_init, device=device)
    model.initialize_centers_from_samples(z_init)
    del z_init
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_cfg = cfg.get("train", {})
    opt = optimizer_for_model(model, train_cfg)
    steps = int(train_cfg.get("steps", 10000))
    x_batch = int(sampling_cfg.get("x_batch", 32))
    vel_per_x = int(sampling_cfg.get("vel_per_x", 256))
    uniform_vel_frac = float(sampling_cfg.get("uniform_vel_frac", 0.20))
    mass_alpha = float(sampling_cfg.get("mass_alpha", 0.70))
    log_every = int(train_cfg.get("log_every", 25))
    save_every = int(train_cfg.get("save_every", 1000))
    moment_every = max(1, int(train_cfg.get("moment_every", 5)))
    lambda_moment = float(train_cfg.get("lambda_moment", 0.0))
    moment_x_count = int(train_cfg.get("moment_x_count", 8))
    moment_v_chunk = int(train_cfg.get("moment_v_chunk", 8192))
    moment_keys = list(train_cfg.get("moment_keys", ["rho", "ux", "T", "qx", "sig"]))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_total", "loss_logf", "loss_moment", "seconds"])

    best_loss = math.inf
    t0 = time.time()
    pbar = trange(1, steps + 1, desc=cfg["run_name"])
    for step in pbar:
        model.train()
        z, target_logf, sample_w = data.sample_phase_batch(
            x_batch=x_batch,
            vel_per_x=vel_per_x,
            uniform_vel_frac=uniform_vel_frac,
            mass_alpha=mass_alpha,
            device=device,
        )
        opt.zero_grad(set_to_none=True)
        pred_logf = model(z)
        loss_logf = logf_loss_fn(pred_logf, target_logf, sample_w, train_cfg)
        loss_m = torch.zeros((), device=device)
        moment_terms = {}
        if lambda_moment > 0.0 and (step % moment_every == 0):
            ix_np = data.sample_x_indices(moment_x_count)
            ix = torch.from_numpy(ix_np).to(device=device, dtype=torch.long)
            if train_cfg.get("moment_loss_mode", "exact") == "sampled":
                loss_m, moment_terms = moment_loss_sampled(
                    model,
                    data,
                    ix,
                    moment_keys,
                    vel_count=int(train_cfg.get("moment_vel_count", 1024)),
                    uniform_frac=float(train_cfg.get("moment_uniform_vel_frac", 0.10)),
                    mass_alpha=float(train_cfg.get("moment_mass_alpha", 0.55)),
                )
            else:
                loss_m, moment_terms = moment_loss(model, data, ix, moment_keys, v_chunk=moment_v_chunk)
        loss = loss_logf + lambda_moment * loss_m
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        loss_float = float(loss.detach().cpu())
        logf_float = float(loss_logf.detach().cpu())
        moment_float = float(loss_m.detach().cpu())
        if loss_float < best_loss:
            best_loss = loss_float
            save_checkpoint(run_dir / "best.pt", model, cfg, data, step, best_loss)

        if step % log_every == 0 or step == 1:
            elapsed = time.time() - t0
            with open(history_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([step, loss_float, logf_float, moment_float, elapsed])
            msg = {"loss": f"{loss_float:.3e}", "logf": f"{logf_float:.3e}", "mom": f"{moment_float:.3e}"}
            pbar.set_postfix(msg)

        if step % save_every == 0:
            save_checkpoint(run_dir / "last.pt", model, cfg, data, step, best_loss)
            plot_training_history(history_path, run_dir / "training_history.png")

    save_checkpoint(run_dir / "last.pt", model, cfg, data, steps, best_loss)
    plot_training_history(history_path, run_dir / "training_history.png")
    print(f"[kgfr] finished. best loss={best_loss:.6e}")
    print(f"[kgfr] run directory: {run_dir}")


if __name__ == "__main__":
    main()
