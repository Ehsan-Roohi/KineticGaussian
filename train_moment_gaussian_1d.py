from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from kgfr.moment1d_model import MomentGaussian1D
from kgfr.utils import choose_device, ensure_dir, load_json, rel_l2, rms_scale, save_json, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_json(args.config)
    set_seed(int(cfg.get("seed", 1)))
    device = choose_device(cfg.get("device", "cuda"))
    run_dir = ensure_dir(Path(cfg.get("output_dir", "runs")) / cfg["run_name"])
    save_json(cfg, run_dir / "config.json")

    d = np.load(cfg["data_path"], allow_pickle=True)
    x = np.asarray(d["x"], dtype=np.float32)
    x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-30) * 2.0 - 1.0).astype(np.float32)
    variables = [v for v in cfg.get("variables", []) if v in d.files]
    if not variables:
        raise ValueError("No requested variables were found in the moment NPZ file.")
    y_raw = np.column_stack([np.asarray(d[k], dtype=np.float32) for k in variables])
    y_mean = y_raw.mean(axis=0)
    y_scale = np.array([rms_scale(y_raw[:, j] - y_mean[j]) for j in range(y_raw.shape[1])], dtype=np.float32)
    y = ((y_raw - y_mean[None, :]) / y_scale[None, :]).astype(np.float32)

    x_t = torch.from_numpy(x_norm).to(device)
    y_t = torch.from_numpy(y).to(device)
    model_cfg = cfg.get("model", {})
    model = MomentGaussian1D(
        num_kernels=int(model_cfg.get("num_kernels", 64)),
        num_outputs=len(variables),
        log_scale_min=float(model_cfg.get("log_scale_min", -8.0)),
        log_scale_max=float(model_cfg.get("log_scale_max", 0.0)),
        init_log_scale=float(model_cfg.get("init_log_scale", -2.0)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("train", {}).get("lr", 1e-3)), weight_decay=1e-8)
    steps = int(cfg.get("train", {}).get("steps", 10000))
    batch_size = int(cfg.get("train", {}).get("batch_size", 512))
    log_every = int(cfg.get("train", {}).get("log_every", 50))
    save_every = int(cfg.get("train", {}).get("save_every", 1000))
    hist_path = run_dir / "history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "loss", "seconds"])
    t0 = time.time()
    best = float("inf")
    pbar = trange(1, steps + 1, desc=cfg["run_name"])
    for step in pbar:
        idx = torch.randint(0, x_t.numel(), (batch_size,), device=device)
        pred = model(x_t[idx])
        loss = torch.mean((pred - y_t[idx]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lf = float(loss.detach().cpu())
        if lf < best:
            best = lf
            torch.save({"model_state": model.state_dict(), "model_meta": model.metadata(), "variables": variables, "y_mean": y_mean, "y_scale": y_scale, "config": cfg}, run_dir / "best.pt")
        if step % log_every == 0 or step == 1:
            with open(hist_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([step, lf, time.time() - t0])
            pbar.set_postfix({"loss": f"{lf:.3e}"})
        if step % save_every == 0:
            torch.save({"model_state": model.state_dict(), "model_meta": model.metadata(), "variables": variables, "y_mean": y_mean, "y_scale": y_scale, "config": cfg}, run_dir / "last.pt")
    model.eval()
    with torch.no_grad():
        pred_n = model(x_t).detach().cpu().numpy()
    pred_raw = pred_n * y_scale[None, :] + y_mean[None, :]
    errors = {variables[j]: rel_l2(pred_raw[:, j], y_raw[:, j]) for j in range(len(variables))}
    with open(run_dir / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    np.savetxt(run_dir / "moment1d_predictions.csv", np.column_stack([x, y_raw, pred_raw]), delimiter=",", header=",".join(["x"] + [f"ref_{k}" for k in variables] + [f"pred_{k}" for k in variables]), comments="")

    n = len(variables)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, max(2.2 * n, 4.0)), sharex=True)
    if n == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(x, y_raw[:, j], label="DVM")
        ax.plot(x, pred_raw[:, j], "--", label="Gaussian 1D")
        ax.set_ylabel(variables[j])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(run_dir / "moment1d_profiles.png", dpi=220)
    plt.close(fig)
    print(json.dumps(errors, indent=2))
    print(f"[kgfr] run directory: {run_dir}")


if __name__ == "__main__":
    main()
