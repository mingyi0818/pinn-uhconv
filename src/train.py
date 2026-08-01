"""Training loop for the PINN-UHconv hydrology model (and all baselines).

This module provides:
  - `train_one_model(...)`: train a single model on the CAMELS-US dataset
  - `collate_fn`: PyTorch collate function for the CAMELSDataset
  - Built-in early stopping on validation NSE
  - Mixed-precision training (AMP)
  - Per-epoch logging to results/logs/

Usage:
  python train.py --model PINN_UHConv --seed 42 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Suppress noisy deprecation warnings that flood the log
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_CFG, MODEL_CFG, TRAIN_CFG, EXP_CFG,
    set_seed, RESULTS_DIR, LOG_DIR,
)
from data_loader import (
    CAMELSDataset, load_full_pipeline, split_basins,
    FORCING_USE,
)
from models import build_model, count_parameters
from losses import PINNUHConvLoss, nse_loss
from evaluate import evaluate_model


# -----------------------------------------------------------------------------
# Collate function (variable-length basin_ids are strings)
# -----------------------------------------------------------------------------
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack numeric fields; keep basin_id as list."""
    out = {
        "forcing": torch.from_numpy(np.stack([b["forcing"] for b in batch])),
        "target_norm": torch.from_numpy(np.stack([b["target_norm"] for b in batch])),
        "target_raw": torch.from_numpy(np.stack([b["target_raw"] for b in batch])),
        "q_mean": torch.from_numpy(np.stack([b["q_mean"] for b in batch])),
        "q_std": torch.from_numpy(np.stack([b["q_std"] for b in batch])),
        "static": torch.from_numpy(np.stack([b["static"] for b in batch])),
        "basin_id": [b["basin_id"] for b in batch],
        "date_str": [b["date_str"] for b in batch],
        "precip_raw": torch.from_numpy(np.stack([b["precip_raw"] for b in batch])),
        "precip_series_raw": torch.from_numpy(np.stack([b["precip_series_raw"] for b in batch])),
    }
    return out


# -----------------------------------------------------------------------------
# Train one model
# -----------------------------------------------------------------------------
def train_one_model(
    model_name: str,
    seed: int = TRAIN_CFG.seed,
    epochs: int = TRAIN_CFG.epochs,
    n_basins: Optional[int] = DATA_CFG.n_basins,
    learning_rate: float = TRAIN_CFG.learning_rate,
    batch_size: int = DATA_CFG.batch_size,
    device: str = TRAIN_CFG.device,
    cache_data: bool = True,
    save_dir: Optional[Path] = None,
    lambda_mass: float = MODEL_CFG.lambda_mass,
    lambda_extreme: float = MODEL_CFG.lambda_extreme,
    use_mass_balance: bool = True,
    use_extreme_weighting: bool = True,
    use_uhconv: bool = True,
    use_static_modulation: bool = True,
    uh_kernel_size: int = MODEL_CFG.uh_kernel_size,
    hidden_size: int = MODEL_CFG.hidden_size,
    seq_length: int = DATA_CFG.seq_length,
    log_every_n_steps: int = 100,
    quiet: bool = False,
    run_name: Optional[str] = None,
) -> Dict:
    """Train a single model and return metrics dict.

    Returns a dict containing:
      - 'model_name', 'seed', 'n_params'
      - 'train_metrics' (last epoch train NSE)
      - 'val_metrics' (best epoch val metrics)
      - 'test_metrics' (final test metrics)
      - 'history' (per-epoch train/val NSE)
      - 'best_epoch'
    """
    # Set seed
    set_seed(seed)

    # Override config
    DATA_CFG.n_basins = n_basins
    DATA_CFG.seq_length = seq_length
    DATA_CFG.batch_size = batch_size
    MODEL_CFG.hidden_size = hidden_size
    MODEL_CFG.uh_kernel_size = uh_kernel_size
    MODEL_CFG.lambda_mass = lambda_mass
    MODEL_CFG.lambda_extreme = lambda_extreme

    # Load data
    if not quiet:
        print(f"[train] Loading CAMELS-US data (n_basins={n_basins})...")
    basin_data, attrs, stats, basin_ids = load_full_pipeline(n_basins=n_basins, cache=cache_data)
    if not quiet:
        print(f"[train] Loaded {len(basin_ids)} basins, {attrs.shape[1]} static attrs")

    # Split basins
    train_basins, val_basins, test_basins = split_basins(basin_ids, seed=seed)
    if not quiet:
        print(f"[train] Basin split: {len(train_basins)} train / {len(val_basins)} val / {len(test_basins)} test")

    # Build datasets
    train_ds = CAMELSDataset(train_basins, "train", basin_data, stats, attrs, seq_length=seq_length)
    val_ds = CAMELSDataset(val_basins, "val", basin_data, stats, attrs, seq_length=seq_length)
    test_ds = CAMELSDataset(test_basins, "test", basin_data, stats, attrs, seq_length=seq_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=TRAIN_CFG.num_workers, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=TRAIN_CFG.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=TRAIN_CFG.num_workers, collate_fn=collate_fn)

    # Build model
    n_forcing = len(FORCING_USE) + 2  # +2 for sin/cos day-of-year
    n_static = attrs.shape[1] if not attrs.empty else 16
    MODEL_CFG.static_input_size = n_static

    # Override flags for ablation
    if model_name == "PINN_UHConv":
        model = build_model(model_name, input_size=n_forcing, static_input_size=n_static)
        # Apply overrides
        model.use_uhconv = use_uhconv
        model.use_mass_balance = use_mass_balance
        model.use_static_modulation = use_static_modulation
    elif model_name == "UH_LSTM":
        # Already configured with use_mass_balance=False in registry
        model = build_model(model_name, input_size=n_forcing, static_input_size=n_static)
        model.use_uhconv = use_uhconv
        model.use_static_modulation = use_static_modulation
    elif model_name == "Phys_LSTM":
        # Phys_LSTM = LSTM model but trained with physics loss
        model = build_model("LSTM", input_size=n_forcing, static_input_size=n_static)
    else:
        model = build_model(model_name, input_size=n_forcing, static_input_size=n_static)

    model = model.to(device)
    n_params = count_parameters(model)
    if not quiet:
        print(f"[train] Model: {model_name} | params: {n_params:,} | device: {device}")

    # Loss & optimiser
    loss_fn = PINNUHConvLoss(
        lambda_mass=lambda_mass if (use_mass_balance and model_name in ("PINN_UHConv", "Phys_LSTM")) else 0.0,
        lambda_extreme=lambda_extreme if use_extreme_weighting else 0.0,
        use_mass_balance=use_mass_balance and model_name in ("PINN_UHConv", "Phys_LSTM"),
        use_extreme_weighting=use_extreme_weighting,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=TRAIN_CFG.weight_decay)
    # Disable AMP for physics-based models: the mass-balance loss involves large
    # values in mm/day that overflow in fp16, producing NaN gradients.
    use_amp = TRAIN_CFG.use_amp and device == "cuda" and model_name not in ("PINN_UHConv", "UH_LSTM", "Phys_LSTM")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_val_nse = -np.inf
    best_epoch = 0
    history = []
    best_state = None
    patience_counter = 0

    log_lines = []
    def log(msg: str):
        if not quiet:
            print(msg, flush=True)
        log_lines.append(msg)

    log(f"[train] Starting training: {model_name} seed={seed} epochs={epochs} lr={learning_rate}")
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nse_loss = 0.0
        epoch_mass_loss = 0.0
        epoch_extreme_loss = 0.0
        n_batches = 0

        for step, batch in enumerate(train_loader):
            forcing = batch["forcing"].to(device)
            static = batch["static"].to(device)
            target_norm = batch["target_norm"].to(device)
            q_mean = batch["q_mean"].to(device)
            q_std = batch["q_std"].to(device)
            # Raw PRCP (mm/day) at the prediction day — used for mass-balance loss
            precip_raw = batch["precip_raw"].to(device)
            # Encode basin_id as integer for per-basin NSE
            # (build mapping on the fly for stability)
            # Skip per-basin NSE in batch; use global batch NSE for speed
            basin_idx = None

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(forcing, static)
                # For physics-based models (PINN_UHConv, UH_LSTM), the model outputs
                # q_raw in mm/day. We normalise it using per-basin (q_mean, q_std) to
                # produce q_norm for the regression loss. q_raw is used for mass balance.
                q_raw = out.get("q_raw")
                if q_raw is not None and model_name in ("PINN_UHConv", "UH_LSTM"):
                    # Clamp q_raw to a physically reasonable range RELATIVE to the
                    # basin's discharge climatology. Without this, the model can
                    # predict q_raw >> q_mean + 3*q_std on low-flow basins (e.g.
                    # q_raw=50 mm/day when q_mean=0.1, q_std=0.2), which makes
                    # pred_norm explode to ~250 and causes MSE loss spikes of
                    # 10-45 and gradient explosions. The clamp at q_mean+10*q_std
                    # allows predicting up to 10 std above the mean (sufficient
                    # for extreme floods) while preventing numerical instability.
                    # NOTE: torch.clamp cannot mix Number min with Tensor max, so
                    # use maximum/minimum which handle tensors cleanly.
                    upper = q_mean + 10.0 * q_std
                    q_raw = torch.maximum(q_raw, torch.zeros_like(q_raw))
                    q_raw = torch.minimum(q_raw, upper)
                    pred_norm = (q_raw - q_mean) / (q_std + 1e-8)
                else:
                    pred_norm = out["q_norm"]
                    q_raw = out.get("q_raw", pred_norm * q_std + q_mean)
                # Pull physics-related outputs if available
                ds_dt_last = out.get("ds_dt_last")
                et_last = out.get("et_last")
                # Use the real raw PRCP (mm/day) for mass-balance loss.
                precip = precip_raw  # real PRCP at prediction day (mm/day)
                # If model produces ET, use it
                if et_last is not None:
                    et = et_last
                else:
                    et = torch.zeros_like(q_raw)
                if ds_dt_last is not None:
                    storage_diff = ds_dt_last
                else:
                    storage_diff = torch.zeros_like(q_raw)

                loss, log_dict = loss_fn(
                    pred_norm, target_norm, q_mean, q_std,
                    precip=precip, et=et, storage_diff=storage_diff,
                    basin_idx=basin_idx,
                    q_raw=q_raw,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu())
            epoch_nse_loss += log_dict.get("loss_nse", 0.0)
            epoch_mass_loss += log_dict.get("loss_mass", 0.0)
            epoch_extreme_loss += log_dict.get("loss_extreme", 0.0)
            n_batches += 1

            if (step + 1) % log_every_n_steps == 0 and not quiet:
                elapsed = time.time() - train_start
                log(f"  Epoch {epoch} step {step+1}/{len(train_loader)} | loss={loss.item():.4f} nse={log_dict.get('loss_nse',0):.4f} mass={log_dict.get('loss_mass',0):.4f} | elapsed={elapsed:.0f}s")

        avg_loss = epoch_loss / max(1, n_batches)
        avg_nse = epoch_nse_loss / max(1, n_batches)
        avg_mass = epoch_mass_loss / max(1, n_batches)
        avg_ext = epoch_extreme_loss / max(1, n_batches)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device=device, model_name=model_name)
        val_nse = val_metrics["NSE_median"]
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_nse_loss": avg_nse,
            "train_mass_loss": avg_mass,
            "train_extreme_loss": avg_ext,
            "val_nse_median": val_nse,
            "val_nse_extreme_median": val_metrics.get("NSE_extreme_median", float("nan")),
            "val_kge_median": val_metrics.get("KGE_median", float("nan")),
        })

        log(f"[train] Epoch {epoch}/{epochs} | train_loss={avg_loss:.4f} (nse={avg_nse:.4f} mass={avg_mass:.4f} ext={avg_ext:.4f}) | val_NSE_med={val_nse:.4f} val_NSE_ext_med={val_metrics.get('NSE_extreme_median', float('nan')):.4f}")

        # Early stopping
        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CFG.early_stopping_patience:
                log(f"[train] Early stopping at epoch {epoch} (patience={TRAIN_CFG.early_stopping_patience})")
                break

    train_time = time.time() - train_start
    log(f"[train] Training finished in {train_time/60:.1f} min. Best epoch={best_epoch} val_NSE_med={best_val_nse:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test metrics
    test_metrics = evaluate_model(model, test_loader, device=device, model_name=model_name)
    val_metrics_final = evaluate_model(model, val_loader, device=device, model_name=model_name)

    result = {
        "model_name": model_name,
        "seed": seed,
        "n_params": n_params,
        "n_basins_used": len(basin_ids),
        "n_train_basins": len(train_basins),
        "n_val_basins": len(val_basins),
        "n_test_basins": len(test_basins),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
        "best_epoch": best_epoch,
        "best_val_nse_median": best_val_nse,
        "train_time_sec": train_time,
        "config": {
            "epochs": epochs,
            "lr": learning_rate,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "hidden_size": hidden_size,
            "uh_kernel_size": uh_kernel_size,
            "lambda_mass": lambda_mass,
            "lambda_extreme": lambda_extreme,
            "use_mass_balance": use_mass_balance,
            "use_extreme_weighting": use_extreme_weighting,
            "use_uhconv": use_uhconv,
            "use_static_modulation": use_static_modulation,
        },
        "val_metrics": val_metrics_final,
        "test_metrics": test_metrics,
        "history": history,
    }

    # Save
    if save_dir is None:
        save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    file_tag = run_name if run_name else f"{model_name}_seed{seed}"
    out_file = save_dir / f"train_{file_tag}.json"
    with open(out_file, "w") as f:
        # Convert non-serializable items
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
            return str(o)
        json.dump(result, f, indent=2, default=_default)
    log(f"[train] Results saved to {out_file}")

    # Save log
    log_file = LOG_DIR / f"train_{file_tag}.log"
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines))
    return result


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PINN_UHConv",
                        choices=["LSTM", "EA_LSTM", "MTS_LSTM", "Transformer",
                                 "Phys_LSTM", "UH_LSTM", "PINN_UHConv"])
    parser.add_argument("--seed", type=int, default=TRAIN_CFG.seed)
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    parser.add_argument("--n_basins", type=int, default=DATA_CFG.n_basins)
    parser.add_argument("--lr", type=float, default=TRAIN_CFG.learning_rate)
    parser.add_argument("--batch_size", type=int, default=DATA_CFG.batch_size)
    parser.add_argument("--device", type=str, default=TRAIN_CFG.device)
    parser.add_argument("--seq_length", type=int, default=DATA_CFG.seq_length)
    parser.add_argument("--hidden_size", type=int, default=MODEL_CFG.hidden_size)
    parser.add_argument("--uh_kernel_size", type=int, default=MODEL_CFG.uh_kernel_size)
    parser.add_argument("--lambda_mass", type=float, default=MODEL_CFG.lambda_mass)
    parser.add_argument("--lambda_extreme", type=float, default=MODEL_CFG.lambda_extreme)
    parser.add_argument("--no_mass_balance", action="store_true")
    parser.add_argument("--no_extreme_weighting", action="store_true")
    parser.add_argument("--no_uhconv", action="store_true")
    parser.add_argument("--no_static_modulation", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    result = train_one_model(
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        n_basins=args.n_basins,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        cache_data=not args.no_cache,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        uh_kernel_size=args.uh_kernel_size,
        lambda_mass=args.lambda_mass,
        lambda_extreme=args.lambda_extreme,
        use_mass_balance=not args.no_mass_balance,
        use_extreme_weighting=not args.no_extreme_weighting,
        use_uhconv=not args.no_uhconv,
        use_static_modulation=not args.no_static_modulation,
    )
    print()
    print("=" * 80)
    print(f"FINAL TEST METRICS — {args.model} (seed={args.seed})")
    print("=" * 80)
    for k, v in result["test_metrics"].items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}")
        else:
            print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    main()
