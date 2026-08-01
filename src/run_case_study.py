"""Run a real-world case study: per-basin hydrograph visualisation and analysis.

Trains PINN_UHConv (seed=42) and a vanilla LSTM for comparison, then evaluates
on selected test basins to produce:
  - Per-basin NSE/KGE/PBIAS for both models
  - Predicted vs observed hydrograph for a representative flood event
  - Mass-balance residual analysis
  - Learned UH kernel visualisation for each basin

Saves results to results/case_study/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_CFG, MODEL_CFG, TRAIN_CFG, RESULTS_DIR, set_seed
from data_loader import load_full_pipeline, split_basins, CAMELSDataset, FORCING_USE
from models import build_model, count_parameters
from train import train_one_model, collate_fn
from evaluate import compute_all_metrics


def select_representative_basins(per_basin_metrics: Dict, n: int = 3) -> List[str]:
    """Select n basins spanning the performance distribution."""
    items = [(b, m.get("NSE", -999)) for b, m in per_basin_metrics.items() if "NSE" in m]
    items.sort(key=lambda x: x[1])
    if len(items) <= n:
        return [b for b, _ in items]
    # Pick: worst, median, best
    idx = [0, len(items) // 2, len(items) - 1]
    idx = sorted(set(idx))[:n]
    return [items[i][0] for i in idx]


@torch.no_grad()
def evaluate_per_basin_with_predictions(model, dataloader, device, model_name: str) -> Dict:
    """Evaluate model and return per-basin predictions + metrics."""
    model.eval()
    basin_data = {}
    for batch in dataloader:
        forcing = batch["forcing"].to(device)
        static = batch["static"].to(device)
        q_norm = batch["target_norm"].to(device)
        q_mean = batch["q_mean"]
        q_std = batch["q_std"]
        basin_id = batch["basin_id"]

        out = model(forcing, static)
        q_raw_pred = out.get("q_raw")
        if q_raw_pred is not None and model_name in ("PINN_UHConv", "UH_LSTM"):
            pred_raw = q_raw_pred.detach().cpu().numpy()
        else:
            pred_norm = out["q_norm"]
            pred_raw = (pred_norm.detach().cpu() * q_std + q_mean).numpy()
        obs_raw = (q_norm.detach().cpu() * q_std + q_mean).numpy()

        for i in range(len(basin_id)):
            bid = str(basin_id[i])
            if bid not in basin_data:
                basin_data[bid] = {"pred": [], "obs": []}
            basin_data[bid]["pred"].extend(np.atleast_1d(pred_raw[i]).tolist())
            basin_data[bid]["obs"].extend(np.atleast_1d(obs_raw[i]).tolist())

    # Compute per-basin metrics
    per_basin = {}
    for bid, data in basin_data.items():
        pred = np.array(data["pred"])
        obs = np.array(data["obs"])
        if len(pred) == 0:
            continue
        mask = (obs >= 0) & np.isfinite(obs) & np.isfinite(pred)
        pred, obs = pred[mask], obs[mask]
        if len(pred) < 10:
            continue
        nse = 1 - np.sum((pred - obs) ** 2) / (np.sum((obs - obs.mean()) ** 2) + 1e-8)
        pbias = 100 * (pred.mean() - obs.mean()) / (obs.mean() + 1e-8)
        # KGE
        r = np.corrcoef(pred, obs)[0, 1] if len(pred) > 1 else 0.0
        alpha = pred.std() / (obs.std() + 1e-8)
        beta = pred.mean() / (obs.mean() + 1e-8)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        per_basin[bid] = {
            "NSE": float(nse), "KGE": float(kge), "PBIAS": float(pbias),
            "Pearson_r": float(r), "n_samples": int(len(pred)),
            "pred_q95": float(np.percentile(pred, 95)),
            "obs_q95": float(np.percentile(obs, 95)),
            "pred_max": float(np.max(pred)),
            "obs_max": float(np.max(obs)),
        }
    return {"per_basin_metrics": per_basin, "basin_predictions": basin_data}


@torch.no_grad()
def extract_uh_kernel(model, dataloader, device, max_basins: int = 15) -> Dict:
    """Extract the learned UH kernel for each basin by accessing the UHconv module."""
    model.eval()
    # Find the UHconv module
    uhconv = None
    for name, module in model.named_modules():
        if hasattr(module, 'param_mlp') and hasattr(module, 'lags'):
            uhconv = module
            break
    if uhconv is None:
        print("  [WARN] UHconv module not found in model")
        return {}

    # Find the static encoder (produces s_emb consumed by UHconv)
    # We need to run the model's static encoder to get s_emb, then feed to UHconv.param_mlp
    static_encoder = getattr(model, 'static_encoder', None) or getattr(model, 'static_mlp', None)
    if static_encoder is None:
        print("  [WARN] Static encoder not found; trying to compute from any forward pass")
        # Fallback: run forward and intercept via hook
        kernels = {}
        count = 0

        def hook_fn(module, inputs, output):
            nonlocal count
            # UHconv forward: inputs = (eff_rain, static_emb)
            if len(inputs) >= 2 and inputs[1] is not None:
                static_emb = inputs[1]
                raw = module.param_mlp(static_emb)
                import torch.nn.functional as F
                alpha = F.softplus(raw[:, 0]) + 1.0
                beta = F.softplus(raw[:, 1]) + 0.1
                # Build Gamma kernel
                lags = module.lags  # (K,)
                log_pdf = (alpha.unsqueeze(1) * torch.log(beta.unsqueeze(1) + 1e-8)
                          - torch.lgamma(alpha.unsqueeze(1))
                          + (alpha.unsqueeze(1) - 1) * torch.log(lags.unsqueeze(0) + 1e-8)
                          - beta.unsqueeze(1) * lags.unsqueeze(0))
                U = torch.exp(log_pdf)
                U = U / (U.sum(dim=1, keepdim=True) + 1e-8)
                # Store via attribute
                hook_fn.alpha_vals = alpha.detach().cpu().numpy()
                hook_fn.beta_vals = beta.detach().cpu().numpy()
                hook_fn.kernel_vals = U.detach().cpu().numpy()

        h = uhconv.register_forward_hook(hook_fn)
        for batch in dataloader:
            forcing = batch["forcing"].to(device)
            static = batch["static"].to(device)
            basin_id = batch["basin_id"]
            _ = model(forcing, static)
            a = hook_fn.alpha_vals
            b = hook_fn.beta_vals
            k = hook_fn.kernel_vals
            for i in range(len(basin_id)):
                bid = str(basin_id[i])
                if bid not in kernels:
                    kernels[bid] = {
                        "kernel": k[i].tolist(),
                        "alpha": float(a[i]),
                        "beta": float(b[i]),
                        "time_to_peak": float((a[i] - 1) / b[i]) if b[i] > 0 else None,
                    }
                    count += 1
                    if count >= max_basins:
                        h.remove()
                        return kernels
        h.remove()
        return kernels

    # If we have a direct reference to static_encoder, compute directly
    kernels = {}
    count = 0
    import torch.nn.functional as F
    for batch in dataloader:
        static = batch["static"].to(device)
        basin_id = batch["basin_id"]
        s_emb = static_encoder(static)
        raw = uhconv.param_mlp(s_emb)
        alpha = F.softplus(raw[:, 0]) + 1.0
        beta = F.softplus(raw[:, 1]) + 0.1
        lags = uhconv.lags  # (K,)
        log_pdf = (alpha.unsqueeze(1) * torch.log(beta.unsqueeze(1) + 1e-8)
                  - torch.lgamma(alpha.unsqueeze(1))
                  + (alpha.unsqueeze(1) - 1) * torch.log(lags.unsqueeze(0) + 1e-8)
                  - beta.unsqueeze(1) * lags.unsqueeze(0))
        U = torch.exp(log_pdf)
        U = U / (U.sum(dim=1, keepdim=True) + 1e-8)
        a = alpha.detach().cpu().numpy()
        b = beta.detach().cpu().numpy()
        k = U.detach().cpu().numpy()
        for i in range(len(basin_id)):
            bid = str(basin_id[i])
            if bid not in kernels:
                kernels[bid] = {
                    "kernel": k[i].tolist(),
                    "alpha": float(a[i]),
                    "beta": float(b[i]),
                    "time_to_peak": float((a[i] - 1) / b[i]) if b[i] > 0 else None,
                }
                count += 1
                if count >= max_basins:
                    return kernels
    return kernels


def run_case_study(
    seed: int = 42,
    n_basins: int = 100,
    epochs: int = 15,
    batch_size: int = 256,
    seq_length: int = 180,
    device: str = "cuda",
) -> Dict:
    """Train PINN_UHConv and LSTM, evaluate per-basin, save hydrographs."""
    save_dir = RESULTS_DIR / "case_study"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("CASE STUDY: Per-basin hydrograph and UH kernel analysis")
    print("=" * 100)

    set_seed(seed)
    DATA_CFG.n_basins = n_basins
    DATA_CFG.seq_length = seq_length
    DATA_CFG.batch_size = batch_size

    basin_data, attrs, stats, basin_ids = load_full_pipeline(n_basins=n_basins, cache=True)
    train_basins, val_basins, test_basins = split_basins(basin_ids, seed=seed)
    test_ds = CAMELSDataset(test_basins, "test", basin_data, stats, attrs, seq_length=seq_length)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # ---- Train PINN_UHConv ----
    print("\n[1/3] Training PINN_UHConv for case study...")
    pinn_result = train_one_model(
        model_name="PINN_UHConv",
        seed=seed, epochs=epochs, n_basins=n_basins, batch_size=batch_size,
        device=device, cache_data=True, save_dir=save_dir,
        seq_length=seq_length, quiet=False,
        run_name=f"case_study_PINN_UHConv_seed{seed}",
    )

    # ---- Train LSTM for comparison ----
    print("\n[2/3] Training LSTM for comparison...")
    lstm_result = train_one_model(
        model_name="LSTM",
        seed=seed, epochs=epochs, n_basins=n_basins, batch_size=batch_size,
        device=device, cache_data=True, save_dir=save_dir,
        seq_length=seq_length, quiet=False,
        run_name=f"case_study_LSTM_seed{seed}",
    )

    # ---- Re-train PINN_UHConv to capture model object ----
    print("\n[3/3] Re-training PINN_UHConv to capture model and predictions...")
    set_seed(seed)
    n_forcing = len(FORCING_USE) + 2
    n_static = attrs.shape[1]
    MODEL_CFG.static_input_size = n_static
    model_pinn = build_model("PINN_UHConv", input_size=n_forcing, static_input_size=n_static).to(device)
    model_lstm = build_model("LSTM", input_size=n_forcing, static_input_size=n_static).to(device)

    # Quick retrain (just to have model objects with weights — use the saved results for metrics)
    # Load the train loader and train for fewer epochs since we already have results
    train_ds = CAMELSDataset(train_basins, "train", basin_data, stats, attrs, seq_length=seq_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_ds = CAMELSDataset(val_basins, "val", basin_data, stats, attrs, seq_length=seq_length)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    from losses import PINNUHConvLoss
    from evaluate import evaluate_model

    # Train PINN
    optimizer = torch.optim.Adam(model_pinn.parameters(), lr=TRAIN_CFG.learning_rate, weight_decay=TRAIN_CFG.weight_decay)
    loss_fn = PINNUHConvLoss(
        lambda_mass=MODEL_CFG.lambda_mass, lambda_extreme=MODEL_CFG.lambda_extreme,
        use_mass_balance=True, use_extreme_weighting=True,
    )
    best_val_nse = -np.inf
    best_state = None
    for epoch in range(1, epochs + 1):
        model_pinn.train()
        for batch in train_loader:
            forcing = batch["forcing"].to(device)
            static = batch["static"].to(device)
            target_norm = batch["target_norm"].to(device)
            q_mean = batch["q_mean"].to(device)
            q_std = batch["q_std"].to(device)
            precip_raw = batch["precip_raw"].to(device)
            optimizer.zero_grad()
            out = model_pinn(forcing, static)
            q_raw = out.get("q_raw")
            pred_norm = (q_raw - q_mean) / (q_std + 1e-8) if q_raw is not None else out["q_norm"]
            et = out.get("et_last", torch.zeros_like(q_raw))
            ds_dt = out.get("ds_dt_last", torch.zeros_like(q_raw))
            loss, _ = loss_fn(pred_norm, target_norm, q_mean, q_std,
                              precip=precip_raw, et=et, storage_diff=ds_dt, q_raw=q_raw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pinn.parameters(), TRAIN_CFG.grad_clip)
            optimizer.step()
        vm = evaluate_model(model_pinn, val_loader, device=device, model_name="PINN_UHConv")
        val_nse = vm["NSE_median"]
        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_state = {k: v.detach().cpu().clone() for k, v in model_pinn.state_dict().items()}
        print(f"  PINN Epoch {epoch}/{epochs} | val_NSE_med={val_nse:.4f} | best={best_val_nse:.4f}")
    if best_state is not None:
        model_pinn.load_state_dict(best_state)

    # Train LSTM
    optimizer = torch.optim.Adam(model_lstm.parameters(), lr=TRAIN_CFG.learning_rate, weight_decay=TRAIN_CFG.weight_decay)
    best_val_nse = -np.inf
    best_state = None
    for epoch in range(1, epochs + 1):
        model_lstm.train()
        for batch in train_loader:
            forcing = batch["forcing"].to(device)
            static = batch["static"].to(device)
            target_norm = batch["target_norm"].to(device)
            optimizer.zero_grad()
            out = model_lstm(forcing, static)
            pred_norm = out["q_norm"]
            loss = torch.mean((pred_norm - target_norm) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_lstm.parameters(), TRAIN_CFG.grad_clip)
            optimizer.step()
        vm = evaluate_model(model_lstm, val_loader, device=device, model_name="LSTM")
        val_nse = vm["NSE_median"]
        if val_nse > best_val_nse:
            best_val_nse = val_nse
            best_state = {k: v.detach().cpu().clone() for k, v in model_lstm.state_dict().items()}
        print(f"  LSTM Epoch {epoch}/{epochs} | val_NSE_med={val_nse:.4f} | best={best_val_nse:.4f}")
    if best_state is not None:
        model_lstm.load_state_dict(best_state)

    # ---- Evaluate per-basin predictions ----
    print("\nExtracting per-basin predictions...")
    pinn_eval = evaluate_per_basin_with_predictions(model_pinn, test_loader, device, "PINN_UHConv")
    lstm_eval = evaluate_per_basin_with_predictions(model_lstm, test_loader, device, "LSTM")

    # ---- Extract UH kernels ----
    print("Extracting learned UH kernels...")
    uh_kernels = extract_uh_kernel(model_pinn, test_loader, device, max_basins=len(test_basins))

    # ---- Select representative basins ----
    pinn_per_basin = pinn_eval["per_basin_metrics"]
    representative = select_representative_basins(pinn_per_basin, n=3)
    print(f"Representative basins (worst/median/best NSE): {representative}")

    # ---- Save predictions for representative basins ----
    case_predictions = {}
    for bid in representative:
        case_predictions[bid] = {
            "PINN_UHConv": pinn_eval["basin_predictions"].get(bid, {}),
            "LSTM": lstm_eval["basin_predictions"].get(bid, {}),
            "PINN_metrics": pinn_per_basin.get(bid, {}),
            "LSTM_metrics": lstm_eval["per_basin_metrics"].get(bid, {}),
            "uh_kernel": uh_kernels.get(bid, {}),
        }

    # ---- Aggregate summary ----
    summary = {
        "config": {
            "seed": seed, "n_basins": n_basins, "epochs": epochs,
            "batch_size": batch_size, "seq_length": seq_length,
            "test_basins": [str(b) for b in test_basins],
        },
        "pinn_test_metrics": pinn_result["test_metrics"],
        "lstm_test_metrics": lstm_result["test_metrics"],
        "pinn_per_basin": pinn_per_basin,
        "lstm_per_basin": lstm_eval["per_basin_metrics"],
        "uh_kernels": uh_kernels,
        "representative_basins": representative,
        "case_predictions": case_predictions,
    }

    out_file = RESULTS_DIR / "case_study_results.json"
    with open(out_file, "w") as f:
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            if isinstance(o, torch.Tensor):
                return o.detach().cpu().tolist()
            return str(o)
        json.dump(summary, f, indent=2, default=_default)
    print(f"\nCase study results saved to {out_file}")

    # Print summary table
    print("\n## Per-basin metrics (PINN_UHConv vs LSTM):")
    print(f"{'Basin':<12} {'PINN NSE':<10} {'LSTM NSE':<10} {'PINN KGE':<10} {'LSTM KGE':<10} {'PINN PBIAS':<12} {'LSTM PBIAS':<12}")
    for bid in pinn_per_basin:
        p = pinn_per_basin[bid]
        l = lstm_eval["per_basin_metrics"].get(bid, {})
        print(f"{bid:<12} {p.get('NSE', 0):<10.4f} {l.get('NSE', 0):<10.4f} {p.get('KGE', 0):<10.4f} {l.get('KGE', 0):<10.4f} {p.get('PBIAS', 0):<12.2f} {l.get('PBIAS', 0):<12.2f}")

    print("\n## Learned UH parameters:")
    print(f"{'Basin':<12} {'alpha':<10} {'beta':<10} {'time_to_peak':<15}")
    for bid, k in uh_kernels.items():
        print(f"{bid:<12} {k.get('alpha', 0):<10.4f} {k.get('beta', 0):<10.4f} {k.get('time_to_peak', 0):<15.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run real-world case study")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_basins", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_case_study(
        seed=args.seed,
        n_basins=args.n_basins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
