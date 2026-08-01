"""Evaluation metrics for hydrology runoff prediction.

Primary metrics:
  - NSE  (Nash-Sutcliffe Efficiency)  — standard hydrology metric
  - NSE_extreme (NSE on top-quantile high-flow events) — extreme-flood performance
  - KGE  (Kling-Gupta Efficiency)    — decomposes into correlation, bias, variability
  - Alpha-NSE  (flow variability ratio)
  - Beta-NSE   (bias ratio)
  - Pearson R  (linear correlation)
  - RMSE       (root mean squared error)
  - FHV / FLV  (high/low flow bias — percent bias in top/bottom 30% flows)
  - Peak Q error (error at peak flow events)

All metrics computed per-basin, then median/mean across basins for reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def nse(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-8) -> float:
    """Nash-Sutcliffe Efficiency. obs, pred: 1D arrays."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) < 2:
        return float("nan")
    denom = np.sum((obs - obs.mean()) ** 2) + eps
    numer = np.sum((pred - obs) ** 2)
    return float(1.0 - numer / denom)


def nse_extreme(pred: np.ndarray, obs: np.ndarray, quantile: float = 0.95) -> float:
    """NSE on high-flow events (obs >= 95th percentile)."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) < 5:
        return float("nan")
    thresh = np.quantile(obs, quantile)
    mask = obs >= thresh
    if mask.sum() < 2:
        return float("nan")
    return nse(pred[mask], obs[mask])


def kge(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-8) -> Tuple[float, float, float, float]:
    """Kling-Gupta Efficiency. Returns (KGE, r, alpha, beta)."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else 0.0
    alpha = float(np.std(pred) / (np.std(obs) + eps))
    beta = float(np.mean(pred) / (np.mean(obs) + eps))
    kge_val = 1.0 - float(np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return kge_val, r, alpha, beta


def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def pbias(pred: np.ndarray, obs: np.ndarray, eps: float = 1e-8) -> float:
    """Percent bias: 100 * sum(pred - obs) / sum(obs)."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(100.0 * np.sum(pred - obs) / (np.sum(np.abs(obs)) + eps))


def fhv(pred: np.ndarray, obs: np.ndarray, high_quantile: float = 0.70) -> float:
    """High-flow bias: percent bias in top 30% flows (Yilmaz et al. 2008)."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) < 5:
        return float("nan")
    thresh = np.quantile(obs, high_quantile)
    mask = obs >= thresh
    if mask.sum() < 2:
        return float("nan")
    return float(100.0 * np.sum(pred[mask] - obs[mask]) / (np.sum(obs[mask]) + 1e-8))


def flv(pred: np.ndarray, obs: np.ndarray, low_quantile: float = 0.30) -> float:
    """Low-flow bias: percent bias in bottom 30% flows."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) < 5:
        return float("nan")
    thresh = np.quantile(obs, low_quantile)
    mask = obs <= thresh
    if mask.sum() < 2:
        return float("nan")
    return float(100.0 * np.sum(pred[mask] - obs[mask]) / (np.sum(obs[mask]) + 1e-8))


def peak_error(pred: np.ndarray, obs: np.ndarray) -> float:
    """Relative error at the maximum-observed flow time-step."""
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if len(obs) == 0:
        return float("nan")
    idx = np.argmax(obs)
    obs_peak = obs[idx]
    pred_peak = pred[idx]
    if obs_peak == 0:
        return float("nan")
    return float((pred_peak - obs_peak) / obs_peak)


# -----------------------------------------------------------------------------
# Aggregated metrics across basins
# -----------------------------------------------------------------------------
def compute_all_metrics(
    predictions: np.ndarray, observations: np.ndarray, basin_ids: Optional[np.ndarray] = None,
    extreme_quantile: float = 0.95,
) -> Dict[str, float]:
    """Compute all metrics.

    If basin_ids is provided, compute per-basin metrics then aggregate (median + mean).
    Otherwise compute global metrics over the flattened arrays.
    """
    predictions = _to_numpy(predictions).ravel()
    observations = _to_numpy(observations).ravel()
    if basin_ids is not None:
        basin_ids = _to_numpy(basin_ids).ravel()

    if basin_ids is None:
        # Global
        nse_v = nse(predictions, observations)
        kge_v, r, alpha, beta = kge(predictions, observations)
        return {
            "NSE": nse_v,
            "NSE_extreme": nse_extreme(predictions, observations, extreme_quantile),
            "KGE": kge_v,
            "Pearson_r": r,
            "Alpha_NSE": alpha,
            "Beta_NSE": beta,
            "RMSE": rmse(predictions, observations),
            "PBIAS": pbias(predictions, observations),
            "FHV": fhv(predictions, observations),
            "FLV": flv(predictions, observations),
            "Peak_Error": peak_error(predictions, observations),
        }

    # Per-basin
    unique_basins = np.unique(basin_ids)
    per_basin = {m: [] for m in ["NSE", "NSE_extreme", "KGE", "Pearson_r", "Alpha_NSE", "Beta_NSE",
                                  "RMSE", "PBIAS", "FHV", "FLV", "Peak_Error"]}
    n_basins_valid = 0
    for b in unique_basins:
        mask = basin_ids == b
        if mask.sum() < 30:    # need at least 30 days for stable NSE
            continue
        p = predictions[mask]
        o = observations[mask]
        if np.isnan(p).any() or np.isnan(o).any() or np.std(o) == 0:
            continue
        per_basin["NSE"].append(nse(p, o))
        per_basin["NSE_extreme"].append(nse_extreme(p, o, extreme_quantile))
        kge_v, r, alpha, beta = kge(p, o)
        per_basin["KGE"].append(kge_v)
        per_basin["Pearson_r"].append(r)
        per_basin["Alpha_NSE"].append(alpha)
        per_basin["Beta_NSE"].append(beta)
        per_basin["RMSE"].append(rmse(p, o))
        per_basin["PBIAS"].append(pbias(p, o))
        per_basin["FHV"].append(fhv(p, o))
        per_basin["FLV"].append(flv(p, o))
        per_basin["Peak_Error"].append(peak_error(p, o))
        n_basins_valid += 1

    # Aggregate: report median (robust) and mean
    agg = {"n_basins_valid": n_basins_valid}
    for m, vals in per_basin.items():
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            agg[f"{m}_median"] = float("nan")
            agg[f"{m}_mean"] = float("nan")
        else:
            agg[f"{m}_median"] = float(np.median(vals))
            agg[f"{m}_mean"] = float(np.mean(vals))
    return agg


# -----------------------------------------------------------------------------
# Convenience: evaluate a trained model on a dataset
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model, dataloader, device="cuda", model_name: str = "model",
    return_per_basin: bool = False,
) -> Dict:
    """Run model over dataloader, collect predictions, compute metrics."""
    model.eval()
    all_pred = []
    all_obs = []
    all_basin = []
    all_q_mean = []
    all_q_std = []
    for batch in dataloader:
        forcing = batch["forcing"].to(device)
        static = batch["static"].to(device)
        q_norm = batch["target_norm"].to(device)
        q_mean = batch["q_mean"]
        q_std = batch["q_std"]
        basin_id = batch["basin_id"]

        out = model(forcing, static)
        # For physics-based models (PINN_UHConv, UH_LSTM), the model outputs
        # q_raw in mm/day directly. Use it as the raw prediction.
        q_raw_pred = out.get("q_raw")
        if q_raw_pred is not None and model_name in ("PINN_UHConv", "UH_LSTM"):
            # Apply the SAME relative clamp as in train.py: q_raw in [0, q_mean+10*q_std].
            # Without this the model can predict absurdly large Q on low-flow basins,
            # producing NSE << -10 that drags down NSE_mean even when NSE_median is fine.
            q_mean_dev = q_mean.to(q_raw_pred.device)
            q_std_dev = q_std.to(q_raw_pred.device)
            upper = q_mean_dev + 10.0 * q_std_dev
            q_raw_pred = torch.maximum(q_raw_pred, torch.zeros_like(q_raw_pred))
            q_raw_pred = torch.minimum(q_raw_pred, upper)
            pred_raw = q_raw_pred.detach().cpu().numpy()
        else:
            pred_norm = out["q_norm"]
            # Convert normalised prediction back to raw mm/day
            pred_raw = (pred_norm.detach().cpu() * q_std + q_mean).numpy()
        obs_raw = (q_norm.detach().cpu() * q_std + q_mean).numpy()
        all_pred.append(pred_raw)
        all_obs.append(obs_raw)
        all_basin.append(np.array(basin_id))
        all_q_mean.append(q_mean.numpy())
        all_q_std.append(q_std.numpy())

    pred = np.concatenate(all_pred)
    obs = np.concatenate(all_obs)
    basins = np.concatenate(all_basin)

    metrics = compute_all_metrics(pred, obs, basins)
    metrics["model"] = model_name
    metrics["n_samples"] = int(len(pred))
    if return_per_basin:
        metrics["_predictions"] = pred
        metrics["_observations"] = obs
        metrics["_basins"] = basins
    return metrics


if __name__ == "__main__":
    # Quick test
    np.random.seed(42)
    obs = np.random.exponential(2.0, 1000)
    pred = obs * 0.9 + np.random.normal(0, 0.5, 1000)
    m = compute_all_metrics(pred, obs)
    print("Test metrics:")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
