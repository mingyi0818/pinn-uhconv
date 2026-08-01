"""Run robustness analysis: noise, missing data, and unseen-basin generalisation.

1. Noise robustness: add Gaussian noise to forcing inputs at inference time
   (std = 0.0, 0.05, 0.10, 0.20, 0.30 of normalised forcing).
2. Missing-data robustness: randomly zero out forcing time-steps
   (rate = 0.0, 0.05, 0.10, 0.20, 0.30).
3. Unseen-basin: train on seed-A split, evaluate on seed-B's held-out test basins
   (regionalisation to ungaged basins).

Trains one baseline PINN_UHConv (seed=42), then evaluates robustness.
Results saved to results/experiment4_robustness/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_CFG, MODEL_CFG, TRAIN_CFG, RESULTS_DIR, set_seed
from data_loader import load_full_pipeline, split_basins, CAMELSDataset, FORCING_USE
from models import build_model, count_parameters
from train import train_one_model, collate_fn
from evaluate import compute_all_metrics


@torch.no_grad()
def evaluate_with_perturbation(
    model, dataloader, device, model_name,
    noise_std: float = 0.0, missing_rate: float = 0.0, seed: int = 42,
) -> Dict:
    """Evaluate model with Gaussian noise on forcing and/or random missing forcing."""
    model.eval()
    rng = np.random.RandomState(seed)
    all_pred, all_obs, all_basin = [], [], []

    for batch in dataloader:
        forcing = batch["forcing"].to(device)
        static = batch["static"].to(device)
        q_norm = batch["target_norm"].to(device)
        q_mean = batch["q_mean"]
        q_std = batch["q_std"]
        basin_id = batch["basin_id"]

        # Add Gaussian noise to forcing
        if noise_std > 0:
            noise = torch.randn_like(forcing) * noise_std
            forcing = forcing + noise

        # Randomly zero out forcing time-steps (set to 0 = mean after normalisation)
        if missing_rate > 0:
            mask = torch.rand_like(forcing[..., :1]) < missing_rate  # mask whole time-step
            forcing = forcing * ~mask

        out = model(forcing, static)
        q_raw_pred = out.get("q_raw")
        if q_raw_pred is not None and model_name in ("PINN_UHConv", "UH_LSTM"):
            pred_raw = q_raw_pred.detach().cpu().numpy()
        else:
            pred_norm = out["q_norm"]
            pred_raw = (pred_norm.detach().cpu() * q_std + q_mean).numpy()
        obs_raw = (q_norm.detach().cpu() * q_std + q_mean).numpy()
        all_pred.append(pred_raw)
        all_obs.append(obs_raw)
        all_basin.append(np.array(basin_id))

    pred = np.concatenate(all_pred)
    obs = np.concatenate(all_obs)
    basins = np.concatenate(all_basin)
    metrics = compute_all_metrics(pred, obs, basins)
    metrics["model"] = model_name
    metrics["n_samples"] = int(len(pred))
    return metrics


def run_robustness(
    seed: int = 42,
    n_basins: int = 50,
    epochs: int = 15,
    batch_size: int = 256,
    seq_length: int = 180,
    device: str = "cuda",
) -> Dict:
    """Train baseline then evaluate robustness."""
    save_dir = RESULTS_DIR / "experiment4_robustness"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Train baseline PINN_UHConv inline (capture model object) ----
    print("=" * 100)
    print("STEP 1: Train baseline PINN_UHConv for robustness analysis")
    print("=" * 100)
    set_seed(seed)
    DATA_CFG.n_basins = n_basins
    DATA_CFG.seq_length = seq_length
    DATA_CFG.batch_size = batch_size

    basin_data, attrs, stats, basin_ids = load_full_pipeline(n_basins=n_basins, cache=True)
    train_basins, val_basins, test_basins = split_basins(basin_ids, seed=seed)
    train_ds = CAMELSDataset(train_basins, "train", basin_data, stats, attrs, seq_length=seq_length)
    val_ds = CAMELSDataset(val_basins, "val", basin_data, stats, attrs, seq_length=seq_length)
    test_ds = CAMELSDataset(test_basins, "test", basin_data, stats, attrs, seq_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    n_forcing = len(FORCING_USE) + 2
    n_static = attrs.shape[1]
    MODEL_CFG.static_input_size = n_static
    model = build_model("PINN_UHConv", input_size=n_forcing, static_input_size=n_static).to(device)
    n_params = count_parameters(model)
    print(f"[train] PINN_UHConv | params: {n_params:,} | device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG.learning_rate, weight_decay=TRAIN_CFG.weight_decay)
    from losses import PINNUHConvLoss
    loss_fn = PINNUHConvLoss(
        lambda_mass=MODEL_CFG.lambda_mass, lambda_extreme=MODEL_CFG.lambda_extreme,
        use_mass_balance=True, use_extreme_weighting=True,
    )
    use_amp = False  # PINN model: disable AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_nse = -np.inf
    best_state = None
    from evaluate import evaluate_model
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            forcing = batch["forcing"].to(device)
            static = batch["static"].to(device)
            target_norm = batch["target_norm"].to(device)
            q_mean = batch["q_mean"].to(device)
            q_std = batch["q_std"].to(device)
            precip_raw = batch["precip_raw"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(forcing, static)
                q_raw = out.get("q_raw")
                pred_norm = (q_raw - q_mean) / (q_std + 1e-8) if q_raw is not None else out["q_norm"]
                et = out.get("et_last", torch.zeros_like(q_raw))
                ds_dt = out.get("ds_dt_last", torch.zeros_like(q_raw))
                loss, _ = loss_fn(pred_norm, target_norm, q_mean, q_std,
                                  precip=precip_raw, et=et, storage_diff=ds_dt, q_raw=q_raw)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        # Validate
        vm = evaluate_model(model, val_loader, device=device, model_name="PINN_UHConv")
        val_nse = vm["NSE_median"]
        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch}/{epochs} | val_NSE_med={val_nse:.4f} | best={best_val_nse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val NSE: {best_val_nse:.4f}")

    # ---- Step 2: Noise robustness ----
    print()
    print("=" * 100)
    print("STEP 2: Noise robustness (Gaussian noise on forcing)")
    print("=" * 100)
    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30]
    noise_results = {}
    for nl in noise_levels:
        m = evaluate_with_perturbation(model, test_loader, device, "PINN_UHConv", noise_std=nl)
        noise_results[f"noise_{nl}"] = m
        print(f"  noise_std={nl:.2f} | NSE_med={m['NSE_median']:.4f} | KGE_med={m['KGE_median']:.4f} | PBIAS={m['PBIAS_median']:.2f}%")

    # ---- Step 3: Missing-data robustness ----
    print()
    print("=" * 100)
    print("STEP 3: Missing-data robustness (random zeroed time-steps)")
    print("=" * 100)
    missing_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    missing_results = {}
    for mr in missing_rates:
        m = evaluate_with_perturbation(model, test_loader, device, "PINN_UHConv", missing_rate=mr)
        missing_results[f"missing_{mr}"] = m
        print(f"  missing_rate={mr:.2f} | NSE_med={m['NSE_median']:.4f} | KGE_med={m['KGE_median']:.4f} | PBIAS={m['PBIAS_median']:.2f}%")

    # ---- Step 4: Unseen-basin (regionalisation) ----
    print()
    print("=" * 100)
    print("STEP 4: Unseen-basin robustness (evaluate on different seed's test split)")
    print("=" * 100)
    unseen_results = {}
    for eval_seed in [2024, 7, 123, 999]:
        _, _, other_test_basins = split_basins(basin_ids, seed=eval_seed)
        # Only keep basins NOT in original train set
        unseen_basins = [b for b in other_test_basins if b not in train_basins]
        if not unseen_basins:
            continue
        ds_unseen = CAMELSDataset(unseen_basins, "test", basin_data, stats, attrs, seq_length=seq_length)
        loader_unseen = DataLoader(ds_unseen, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        m = evaluate_with_perturbation(model, loader_unseen, device, "PINN_UHConv")
        unseen_results[f"unseen_seed{eval_seed}"] = m
        print(f"  unseen_basins (seed={eval_seed}, n={len(unseen_basins)}) | NSE_med={m['NSE_median']:.4f} | KGE_med={m['KGE_median']:.4f}")

    summary = {
        "noise_robustness": noise_results,
        "missing_robustness": missing_results,
        "unseen_basin_robustness": unseen_results,
        "config": {
            "seed": seed, "n_basins": n_basins, "epochs": epochs,
            "batch_size": batch_size, "seq_length": seq_length,
            "best_val_nse": best_val_nse,
        },
    }

    out_file = RESULTS_DIR / "experiment4_robustness_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run robustness analysis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_basins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_robustness(
        seed=args.seed,
        n_basins=args.n_basins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
