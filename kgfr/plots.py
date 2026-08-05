from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history_csv: str | Path, out_png: str | Path) -> None:
    path = Path(history_csv)
    if not path.exists():
        return
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        return
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 4.5))
    step = data["step"]
    for name in data.dtype.names:
        if name.startswith("loss"):
            plt.semilogy(step, data[name], label=name)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_moment_profiles(x: np.ndarray, pred: Dict[str, np.ndarray], ref: Dict[str, np.ndarray], keys: Iterable[str], out_png: str | Path) -> None:
    keys = [k for k in keys if k in pred and k in ref]
    if not keys:
        return
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, max(2.2 * n, 4.0)), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(x, ref[k], label="DVM")
        ax.plot(x, pred[k], "--", label="Gaussian")
        ax.set_ylabel(k)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_ladder(errors: Dict[str, float], out_png: str | Path) -> None:
    if not errors:
        return
    keys = list(errors.keys())
    vals = np.array([errors[k] for k in keys], dtype=float)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 4.2))
    x = np.arange(len(keys))
    plt.bar(x, vals)
    plt.yscale("log")
    plt.xticks(x, keys, rotation=35, ha="right")
    plt.ylabel("relative L2 error")
    plt.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
