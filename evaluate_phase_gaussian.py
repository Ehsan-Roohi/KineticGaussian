from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kgfr.data import ShockFullState
from kgfr.io_table import save_dict_csv
from kgfr.models import PhaseGaussianMixture
from kgfr.moments import moments_from_model
from kgfr.plots import plot_ladder, plot_moment_profiles
from kgfr.utils import choose_device, ensure_dir, load_json, rel_l2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--x-stride", type=int, default=5)
    ap.add_argument("--v-chunk", type=int, default=8192)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_json(args.config)
    device = choose_device(cfg.get("device", "cuda"))
    data = ShockFullState(cfg["data_path"], moment_path=cfg.get("moment_path"), f_floor=float(cfg.get("sampling", {}).get("f_floor", 1e-35)))
    ckpt = torch.load(args.checkpoint, map_location=device)
    meta = ckpt["model_meta"]
    model = PhaseGaussianMixture(
        num_kernels=int(meta["num_kernels"]),
        variant=str(meta["variant"]),
        log_scale_min=float(meta.get("log_scale_min", cfg.get("model", {}).get("log_scale_min", -8.0))),
        log_scale_max=float(meta.get("log_scale_max", cfg.get("model", {}).get("log_scale_max", 1.0))),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    run_dir = Path(cfg.get("output_dir", "runs")) / cfg["run_name"]
    out_dir = ensure_dir(args.out_dir or (run_dir / "eval"))
    ix_np = np.arange(0, data.Nx, max(1, args.x_stride), dtype=np.int64)
    pred_cols = {"rho": [], "ux": [], "T": [], "qx": [], "sig": [], "M300": [], "M400": []}
    batch_x = 8
    with torch.no_grad():
        for i0 in range(0, len(ix_np), batch_x):
            ix = torch.from_numpy(ix_np[i0:i0 + batch_x]).to(device=device, dtype=torch.long)
            mom = moments_from_model(model, data, ix, v_chunk=args.v_chunk)
            for k in pred_cols:
                pred_cols[k].append(mom[k].detach().cpu().numpy())
    pred = {k: np.concatenate(v) for k, v in pred_cols.items()}
    x_sel = data.x[ix_np]
    save_dict_csv(out_dir / "predicted_moments.csv", x_sel, pred)

    ref = {}
    for k in ["rho", "ux", "T", "qx", "sig"]:
        if k in data.moment_ref:
            ref[k] = data.moment_ref[k][ix_np]
    errors = {}
    for k, r in ref.items():
        if k in pred:
            errors[k] = rel_l2(pred[k], r)
    # Optional comparisons to high-moment file if definitions are compatible.
    if "M300_neq" in data.moment_ref:
        ref["M300_neq"] = data.moment_ref["M300_neq"][ix_np]
        errors["M300_vs_M300_neq"] = rel_l2(pred["M300"], ref["M300_neq"])
    with open(out_dir / "moment_errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2)
    plot_moment_profiles(x_sel, pred, ref, ["rho", "ux", "T", "qx", "sig", "M300_neq"], out_dir / "moment_profiles.png")
    plot_ladder(errors, out_dir / "nonequilibrium_ladder.png")
    print(json.dumps(errors, indent=2))
    print(f"[kgfr] wrote evaluation to {out_dir}")


if __name__ == "__main__":
    main()
