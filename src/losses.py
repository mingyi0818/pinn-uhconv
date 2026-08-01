"""Loss functions for the PINN-UHconv hydrology model.

Key losses:
  1. NSE loss (negative Nash-Sutcliffe Efficiency) — primary regression loss
  2. Extreme-event weighted NSE loss — emphasises high-flow periods
  3. Mass-balance constraint loss — physics soft constraint
  4. Optional regularisation

NSE is the standard hydrology metric:
  NSE = 1 - sum((Q_pred - Q_obs)^2) / sum((Q_obs - mean(Q_obs))^2)
A perfect model has NSE=1; NSE=0 means predicting the mean; NSE<0 means worse than mean.

For mini-batch training we use a per-batch NSE that approximates the per-basin NSE.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_CFG


# -----------------------------------------------------------------------------
# 1. NSE loss (batched)
# -----------------------------------------------------------------------------
def nse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute MSE loss for training (NSE is used only for evaluation).

    NOTE: We use MSE instead of the NSE-based loss `1 - numer/denom` for training
    because the latter is unbounded below: when predictions are far off, the loss
    becomes very negative, and minimising it drives the model to predict EVEN
    larger errors (divergence). MSE is bounded below by 0 and has stable gradients.
    NSE is still used as the evaluation metric in evaluate.py.
    """
    return ((pred - target) ** 2).mean()


def nse_loss_per_basin(
    pred: torch.Tensor, target: torch.Tensor, basin_idx: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Compute per-basin MSE loss averaged across basins in the batch.

    Uses MSE (not NSE) for stable training. See nse_loss() for rationale.

    pred, target: (batch,)
    basin_idx: (batch,) long tensor, group key per basin
    """
    losses = []
    for b in torch.unique(basin_idx):
        mask = basin_idx == b
        if mask.sum() < 2:
            continue
        p = pred[mask]
        t = target[mask]
        losses.append(((p - t) ** 2).mean())
    if not losses:
        return nse_loss(pred, target, eps)
    return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# 2. Extreme-event weighted loss
# -----------------------------------------------------------------------------
def extreme_weighted_mse(
    pred: torch.Tensor, target: torch.Tensor, q_mean: torch.Tensor,
    q_std: torch.Tensor, gamma: float = MODEL_CFG.lambda_extreme,
    quantile: float = MODEL_CFG.extreme_quantile, eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted MSE where high-flow samples get larger weight (normalised scale).

    Weight: w_t = 1 + gamma * (Q_t - Q_min) / (Q_max - Q_min)
    This emphasises flood events without ignoring base flow entirely.

    pred, target are in NORMALISED units (same scale as l_nse); q_mean/q_std
    are used only to recover raw Q for the weight, NOT for the MSE itself.
    This keeps the loss on the same scale as l_nse (O(1)) so it does not
    dominate the total loss.
    """
    raw_t = (target * q_std + q_mean).detach()   # raw Q (mm/day), detached for weighting only
    q_max = raw_t.max()
    q_min = raw_t.min()
    w = 1.0 + gamma * (raw_t - q_min) / (q_max - q_min + eps)
    return (w * (pred - target) ** 2).mean()


# -----------------------------------------------------------------------------
# 3. Mass-balance constraint (physics soft constraint)
# -----------------------------------------------------------------------------
def mass_balance_loss(
    storage_diff: torch.Tensor,    # dS/dt (mm/day)  shape (batch,)
    precip: torch.Tensor,           # P (mm/day)     shape (batch,)
    et: torch.Tensor,               # ET (mm/day)    shape (batch,)
    q_out: torch.Tensor,            # Q (mm/day)     shape (batch,)
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalise violation of the water-balance continuity equation:
        dS/dt = P - ET - Q

    Uses a SCALE-INVARIANT relative residual so the loss is O(1) regardless of
    whether P/Q are in mm/day (which can be 0-100+). The absolute-residual
    version produces losses of 10-1000+ that dominate the NSE loss and distort
    the q_raw scale (causing systematic over-estimation, PBIAS~90%).

    Loss = mean( (dS/dt - (P - ET - Q))^2 / (|P| + |ET| + |Q| + |dS/dt| + eps)^2 )
    """
    residual = storage_diff - (precip - et - q_out)
    scale = precip.abs() + et.abs() + q_out.abs() + storage_diff.abs() + eps
    return ((residual / scale) ** 2).mean()


# -----------------------------------------------------------------------------
# 4. Combined PINN-UHconv loss
# -----------------------------------------------------------------------------
class PINNUHConvLoss(nn.Module):
    """Combined loss for PINN-UHconv model.

    L = L_nse + lambda_mass * L_mass + lambda_extreme * L_extreme_extra

    where:
      L_nse = extreme-weighted NSE-style loss (or plain MSE on normalised Q)
      L_mass = water-balance constraint penalty
      L_extreme = extra high-flow quantile MSE (sharpens extreme-event prediction)
    """

    def __init__(
        self,
        lambda_mass: float = MODEL_CFG.lambda_mass,
        lambda_extreme: float = MODEL_CFG.lambda_extreme,
        extreme_quantile: float = MODEL_CFG.extreme_quantile,
        use_mass_balance: bool = True,
        use_extreme_weighting: bool = True,
    ):
        super().__init__()
        self.lambda_mass = lambda_mass
        self.lambda_extreme = lambda_extreme
        self.extreme_quantile = extreme_quantile
        self.use_mass_balance = use_mass_balance
        self.use_extreme_weighting = use_extreme_weighting

    def forward(
        self,
        pred_q_norm: torch.Tensor,          # (batch,) predicted normalised Q
        target_q_norm: torch.Tensor,        # (batch,) observed normalised Q
        q_mean: torch.Tensor,               # (batch,)
        q_std: torch.Tensor,                # (batch,)
        precip: torch.Tensor,               # (batch,) P at prediction day (mm/day, raw)
        et: Optional[torch.Tensor] = None,  # (batch,) ET (mm/day, raw)
        storage_diff: Optional[torch.Tensor] = None,  # (batch,) dS/dt predicted (mm/day)
        basin_idx: Optional[torch.Tensor] = None,
        q_raw: Optional[torch.Tensor] = None,  # (batch,) model's raw Q in mm/day (for mass balance)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (total_loss, log_dict)."""
        # Primary NSE loss (batch-wise)
        if basin_idx is not None:
            l_nse = nse_loss_per_basin(pred_q_norm, target_q_norm, basin_idx)
        else:
            l_nse = nse_loss(pred_q_norm, target_q_norm)

        logs = {"loss_nse": float(l_nse.detach().cpu())}

        total = l_nse

        # Extreme weighting: additional Huber loss on top-quantile samples (NORMALISED scale)
        if self.use_extreme_weighting and self.lambda_extreme > 0:
            # Recover raw Q only to find the high-flow threshold
            raw_t = (target_q_norm * q_std + q_mean).detach()
            q_thresh = torch.quantile(raw_t, self.extreme_quantile)
            mask = raw_t >= q_thresh
            if mask.sum() >= 2:
                # Huber (SmoothL1) loss in NORMALISED units — robust to large
                # errors on extreme events. MSE on extreme events produces
                # losses of 3-6+ that dominate the total loss and cause gradient
                # explosions; Huber linearises the penalty above |delta|>beta,
                # giving stable gradients while still emphasising high-flow events.
                l_ext = F.smooth_l1_loss(pred_q_norm[mask], target_q_norm[mask], beta=1.0)
                total = total + self.lambda_extreme * l_ext
                logs["loss_extreme"] = float(l_ext.detach().cpu())

        # Mass-balance constraint
        if self.use_mass_balance and self.lambda_mass > 0:
            if et is None:
                et = torch.zeros_like(precip)
            if storage_diff is None:
                storage_diff = torch.zeros_like(precip)
            # Use the model's raw Q if provided (physical Q in mm/day); otherwise
            # recover from normalised prediction (assumes q_norm = (q_raw-q_mean)/q_std)
            if q_raw is None:
                q_raw = pred_q_norm * q_std + q_mean
            l_mass = mass_balance_loss(storage_diff, precip, et, q_raw)
            total = total + self.lambda_mass * l_mass
            logs["loss_mass"] = float(l_mass.detach().cpu())

        logs["loss_total"] = float(total.detach().cpu())
        return total, logs
