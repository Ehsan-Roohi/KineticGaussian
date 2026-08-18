from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[kgfr] CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def rel_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-30) -> float:
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.sqrt(np.sum((pred - ref) ** 2) / (np.sum(ref ** 2) + eps)))


def rms_scale(a: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64)
    s = float(np.sqrt(np.mean(a * a)))
    return max(s, eps)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_float32(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.float32:
        return a.astype(np.float32)
    return a
