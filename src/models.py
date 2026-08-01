"""Models for the PINN-UHconv hydrology runoff prediction project.

Models implemented:
  1. LSTM                 — basic LSTM (Kratzert et al. 2018)
  2. EA_LSTM              — Embedding-Approach LSTM (Kratzert et al. 2018)
  3. MTS_LSTM             — Multi-Temporal-Scale LSTM (Gauch et al. 2021)
  4. Transformer          — attention-based baseline
  5. Phys_LSTM            — LSTM + mass-balance loss (uses LSTM model, physics loss)
  6. UH_LSTM              — LSTM + UHconv but without mass-balance (ablation bridge)
  7. PINN_UHConv (ours)   — LSTM + UHconv + mass-balance + static modulation + extreme weighting

Key innovation: differentiable unit-hydrograph convolution (UHconv) that embeds
the linear routing physics as a 1D convolution with Gamma-distributed kernel whose
shape parameters (alpha, beta) are predicted from basin attributes.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_CFG


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def gamma_log_pdf(k: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """log of Gamma(k; alpha, beta) PDF for integer-ish k (lag indices 1..K).

    f(k) = beta^alpha / Gamma(alpha) * k^(alpha-1) * exp(-beta*k)
    log f(k) = alpha*log(beta) - log Gamma(alpha) + (alpha-1)*log(k) - beta*k

    k: (K,)  alpha,beta: (B,)  ->  returns (B, K)
    """
    k = k.unsqueeze(0)  # (1, K)
    alpha = alpha.unsqueeze(1)  # (B, 1)
    beta = beta.unsqueeze(1)
    log_k = torch.log(k.clamp(min=1e-6))
    log_pdf = alpha * torch.log(beta.clamp(min=1e-6)) \
              - torch.lgamma(alpha) \
              + (alpha - 1.0) * log_k \
              - beta * k
    return log_pdf


# -----------------------------------------------------------------------------
# Differentiable Unit-Hydrograph Convolution (UHconv)
# -----------------------------------------------------------------------------
class UHconv(nn.Module):
    """Differentiable Unit Hydrograph convolution.

    Predicts (alpha, beta) Gamma-distribution parameters from a static embedding,
    builds a normalised kernel U over lag indices 1..K, then convolves effective
    rainfall R (mm/day) with U to yield surface runoff Q_surface.

    Q_surface[t] = sum_{k=1..K} U[k] * R[t-k]
    """
    def __init__(self, static_dim: int, kernel_size: int = MODEL_CFG.uh_kernel_size,
                 hidden_size: int = MODEL_CFG.uh_hidden_size):
        super().__init__()
        self.kernel_size = kernel_size
        # MLP: static -> (alpha, beta). Use softplus to keep > 0.
        self.param_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),  # outputs raw alpha, raw beta
        )
        # Lag indices 1..K (start at 1 to avoid div-by-zero in log)
        self.register_buffer("lags", torch.arange(1, kernel_size + 1, dtype=torch.float32))

    def forward(self, eff_rain: torch.Tensor, static_emb: torch.Tensor) -> torch.Tensor:
        """Compute surface runoff via causal 1D convolution with Gamma UH kernel.

        eff_rain:    (B, T)  effective rainfall in mm/day (already positive)
        static_emb:  (B, S)  static attribute embedding
        Returns:     (B, T)  surface runoff Q_surface in mm/day

        Math: Q[t] = sum_{k=0}^{K-1} U[k] * R[t-k]  (R is zero-padded on the left)
        Implementation: unfold padded R into (B, T, K) windows, then bmm with U.
        """
        B, T = eff_rain.shape
        K = self.kernel_size
        # Predict (alpha, beta), enforce positivity via softplus
        raw = self.param_mlp(static_emb)
        # Constrain alpha >= 1 (so that peak is at finite lag) and beta > 0
        alpha = F.softplus(raw[:, 0]) + 1.0  # (B,)
        beta = F.softplus(raw[:, 1]) + 0.1   # (B,)

        # Build Gamma kernel (B, K)
        log_pdf = gamma_log_pdf(self.lags, alpha, beta)  # (B, K)
        U = torch.exp(log_pdf)
        U = U / (U.sum(dim=1, keepdim=True) + 1e-8)  # normalise so sum_k U[k] = 1

        # Causal conv via unfold + bmm.
        # Pad R on the left with K-1 zeros so that output length == T.
        R_pad = F.pad(eff_rain, (K - 1, 0))   # (B, T+K-1)
        # Build sliding windows of size K: shape (B, T, K)
        #   window[t, k] = R_pad[t + k]   for k=0..K-1
        # We want Q[t] = sum_k U[k] * R[t-k] = sum_k U[k] * R_pad[t + (K-1) - k]
        #              = sum_k U[k] * window[t][K-1-k] = sum_k U_flipped[k] * window[t][k]
        # where U_flipped = U.flip(dim=1)
        windows = R_pad.unfold(dimension=1, size=K, step=1)  # (B, T, K)
        U_flip = U.flip(dims=[1]).unsqueeze(2)               # (B, K, 1)
        Q_surface = torch.bmm(windows, U_flip).squeeze(-1)    # (B, T)
        return Q_surface


# -----------------------------------------------------------------------------
# LSTM cell with static attribute modulation (FiLM-like gating)
# -----------------------------------------------------------------------------
class StaticModulatedLSTMCell(nn.Module):
    """LSTM cell whose hidden state is gated by static attributes.

    h_t = LSTMCell(x_t, h_{t-1}) ⊙ σ(W_a * static_emb + b_a)
    """
    def __init__(self, input_size: int, hidden_size: int, static_dim: int):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.gate = nn.Linear(static_dim, hidden_size)
        # init gate bias to 0 -> sigmoid=0.5 initially (50% pass-through)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor,
                static_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_new, c_new = self.lstm_cell(x, (h, c))
        gate = torch.sigmoid(self.gate(static_emb))
        h_new = h_new * gate
        return h_new, c_new


# -----------------------------------------------------------------------------
# 1. Basic LSTM baseline
# -----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    """Basic LSTM baseline (Kratzert et al. 2018).

    Predicts Q at last time-step from a sequence of forcings.
    Static attributes are concatenated with the final hidden state.
    """
    def __init__(self, input_size: int = MODEL_CFG.input_size,
                 hidden_size: int = MODEL_CFG.hidden_size,
                 num_layers: int = MODEL_CFG.num_layers,
                 static_input_size: int = MODEL_CFG.static_input_size,
                 dropout: float = MODEL_CFG.dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        # forcing: (B, T, F), static: (B, A)
        out, (h_n, c_n) = self.lstm(forcing)
        h_last = out[:, -1, :]   # (B, H)
        # Concat static
        x = torch.cat([h_last, static], dim=-1)
        q_norm = self.head(x).squeeze(-1)
        result = {"q_norm": q_norm}
        if return_states:
            result["hidden_states"] = out  # (B, T, H)
            result["h_last"] = h_last
        return result


# -----------------------------------------------------------------------------
# 2. EA-LSTM (Embedding Approach LSTM)
# -----------------------------------------------------------------------------
class EALSTMModel(nn.Module):
    """EA-LSTM: static attributes modulate the LSTM input gate (Kratzert et al. 2018).

    The input gate i_t is computed from (x_t, static) instead of (x_t, h_{t-1}).
    Implementation: use a custom LSTM cell.
    """
    def __init__(self, input_size: int = MODEL_CFG.input_size,
                 hidden_size: int = MODEL_CFG.hidden_size,
                 static_input_size: int = MODEL_CFG.static_input_size,
                 dropout: float = MODEL_CFG.dropout):
        super().__init__()
        self.hidden_size = hidden_size
        # Static embedding -> hidden_size
        self.static_emb = nn.Linear(static_input_size, hidden_size)
        # Standard LSTM gates except input gate (which uses static)
        # i = σ(W_xi * x + W_si * static + b_i)
        self.x2i = nn.Linear(input_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        # f, g, o use standard LSTM formulation
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        B, T, _ = forcing.shape
        s_emb = self.static_emb(static)  # (B, H)
        h = torch.zeros(B, self.hidden_size, device=forcing.device, dtype=forcing.dtype)
        c = torch.zeros(B, self.hidden_size, device=forcing.device, dtype=forcing.dtype)
        hs = []
        for t in range(T):
            x_t = forcing[:, t, :]
            # Input gate uses static embedding
            i = torch.sigmoid(self.x2i(x_t) + s_emb + self.b_i)
            # f, g, o via standard cell
            gates = self.lstm_cell(x_t, (h, c))  # this returns (h_new, c_new)
            # We need access to f, g, o individually; nn.LSTMCell hides them.
            # Workaround: use full custom cell below
            # Use the EA-style update:
            f = torch.sigmoid(gates[0] * 0 + 1)  # placeholder; better to use full custom cell
            # Simpler: ignore EA exact math and apply i as a multiplicative gate on h
            h_new = gates[0]
            c_new = gates[1]
            h_new = h_new * i
            h, c = h_new, c_new
            hs.append(h)
        out = torch.stack(hs, dim=1)  # (B, T, H)
        h_last = out[:, -1, :]
        x = h_last  # static already embedded via input gate
        q_norm = self.head(x).squeeze(-1)
        result = {"q_norm": q_norm}
        if return_states:
            result["hidden_states"] = out
            result["h_last"] = h_last
        return result


class EALSTMModelClean(nn.Module):
    """Clean EA-LSTM implementation (fast cuDNN version).

    Following Kratzert et al. 2018 (Water Resources Research), the EA-LSTM uses
    static catchment attributes to modulate the input gate. The exact formulation
    requires a Python for-loop over time steps (since the input gate depends on
    static attributes but the recurrence still depends on h_{t-1}), which is
    ~100x slower than cuDNN nn.LSTM for seq_length=365.

    Fast approximation used here:
      1. Run a standard nn.LSTM (cuDNN-accelerated) over the full forcing sequence.
      2. Apply a static-attribute-dependent multiplicative gate to the output
         hidden states: h_t' = h_t ⊙ σ(W_si * static + b_i).
    This preserves the key EA-LSTM intent (static attributes condition the LSTM
    output) while being fast enough for large-scale experiments. The static
    information still flows into the model through the gate, and the final
    prediction head benefits from static conditioning.
    """
    def __init__(self, input_size: int = MODEL_CFG.input_size,
                 hidden_size: int = MODEL_CFG.hidden_size,
                 static_input_size: int = MODEL_CFG.static_input_size,
                 num_layers: int = MODEL_CFG.num_layers,
                 dropout: float = MODEL_CFG.dropout):
        super().__init__()
        self.hidden_size = hidden_size
        # Standard cuDNN LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Static -> input gate modulation (EA-style: static affects input gate)
        self.W_si = nn.Linear(static_input_size, hidden_size)
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        B, T, _ = forcing.shape
        # Standard LSTM (fast, cuDNN)
        out, _ = self.lstm(forcing)            # (B, T, H)
        # EA-style input gate modulation from static attributes
        i_gate = torch.sigmoid(self.W_si(static) + self.b_i)  # (B, H)
        out = out * i_gate.unsqueeze(1)        # (B, T, H)
        h_last = out[:, -1, :]
        q_norm = self.head(h_last).squeeze(-1)
        result = {"q_norm": q_norm}
        if return_states:
            result["hidden_states"] = out
            result["h_last"] = h_last
        return result


# -----------------------------------------------------------------------------
# 3. MTS-LSTM (Multi-Temporal Scale LSTM, simplified)
# -----------------------------------------------------------------------------
class MTS_LSTMModel(nn.Module):
    """Simplified Multi-Temporal-Scale LSTM (Gauch et al. 2021).

    Runs two LSTMs in parallel: one on the daily sequence (1-day resolution)
    and one on a downsampled sequence (weekly aggregation). Their final hidden
    states are concatenated for prediction.
    """
    def __init__(self, input_size: int = MODEL_CFG.input_size,
                 hidden_size: int = MODEL_CFG.hidden_size,
                 static_input_size: int = MODEL_CFG.static_input_size,
                 dropout: float = MODEL_CFG.dropout):
        super().__init__()
        # Daily LSTM
        self.lstm_d = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Weekly LSTM (downsample by 7)
        self.lstm_w = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size + static_input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        # Daily
        out_d, _ = self.lstm_d(forcing)
        h_d = out_d[:, -1, :]
        # Weekly downsample (take every 7th step from the end)
        T = forcing.shape[1]
        weekly = forcing[:, T % 7::7, :] if T >= 7 else forcing
        out_w, _ = self.lstm_w(weekly)
        h_w = out_w[:, -1, :]
        x = torch.cat([h_d, h_w, static], dim=-1)
        q_norm = self.head(x).squeeze(-1)
        result = {"q_norm": q_norm}
        if return_states:
            result["hidden_states"] = out_d
        return result


# -----------------------------------------------------------------------------
# 4. Transformer baseline
# -----------------------------------------------------------------------------
class TransformerModel(nn.Module):
    """Simple Transformer encoder + linear head baseline."""
    def __init__(self, input_size: int = MODEL_CFG.input_size,
                 hidden_size: int = MODEL_CFG.hidden_size,
                 static_input_size: int = MODEL_CFG.static_input_size,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = MODEL_CFG.dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        # Positional encoding (sinusoidal)
        self.pos_encoding = SinPositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4 * hidden_size,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        T = forcing.shape[1]
        x = self.input_proj(forcing)
        x = x + self.pos_encoding(T).to(x.device).unsqueeze(0)
        out = self.encoder(x)
        h_last = out[:, -1, :]
        x = torch.cat([h_last, static], dim=-1)
        q_norm = self.head(x).squeeze(-1)
        result = {"q_norm": q_norm}
        if return_states:
            result["hidden_states"] = out
        return result


class SinPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, T: int) -> torch.Tensor:
        return self.pe[:T]


# -----------------------------------------------------------------------------
# 5. PINN-UHconv (our method)
# -----------------------------------------------------------------------------
class PINNUHConvModel(nn.Module):
    """PINN + Differentiable Unit-Hydrograph Convolution.

    Architecture:
      1. Static attribute embedding MLP -> static_emb (B, S)
      2. Static-modulated LSTM encoder over forcing sequence -> hidden_states (B, T, H)
      3. Effective rainfall head: R_t = softplus(MLP(h_t, static_emb))   (B, T)
      4. UHconv: Q_surface = U(R, static_emb)                          (B, T)
      5. Baseflow head: Q_base = softplus(MLP(h_T, static_emb))         (B,)
      6. Storage head: S_t = MLAP(h_t) (B, T) (for mass-balance loss)
      7. ET head: ET_t = softplus(MLP(h_t)) (B, T) (for mass-balance loss)
      8. Total Q_norm = (Q_surface[:, -1] + Q_base - q_mean) / q_std   (predicted normalised Q at last step)

    Flags allow component-wise ablation:
      use_uhconv=False         -> Q_surface = sum(R) (no routing)
      use_mass_balance=False   -> no storage/ET heads; loss ignores mass constraint
      use_static_modulation=False -> use plain LSTM cell instead of modulated cell
      use_extreme_weighting=False -> handled by loss
    """
    def __init__(
        self,
        input_size: int = MODEL_CFG.input_size,
        hidden_size: int = MODEL_CFG.hidden_size,
        num_layers: int = MODEL_CFG.num_layers,
        static_input_size: int = MODEL_CFG.static_input_size,
        static_embed_size: int = MODEL_CFG.static_size,
        uh_kernel_size: int = MODEL_CFG.uh_kernel_size,
        uh_hidden_size: int = MODEL_CFG.uh_hidden_size,
        dropout: float = MODEL_CFG.dropout,
        use_uhconv: bool = True,
        use_mass_balance: bool = True,
        use_static_modulation: bool = True,
    ):
        super().__init__()
        self.use_uhconv = use_uhconv
        self.use_mass_balance = use_mass_balance
        self.use_static_modulation = use_static_modulation
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Static embedding MLP
        self.static_encoder = nn.Sequential(
            nn.Linear(static_input_size, static_embed_size * 2),
            nn.Tanh(),
            nn.Linear(static_embed_size * 2, static_embed_size),
            nn.Tanh(),
        )

        # LSTM encoder — always use cuDNN-accelerated nn.LSTM for speed.
        # When use_static_modulation=True we apply a static-attribute-dependent
        # multiplicative gate (FiLM-style) to the LSTM output hidden states.
        # This is a fast approximation of the per-step StaticModulatedLSTMCell
        # (which uses a Python for-loop and is ~100x slower for seq_length=365).
        # The gate depends only on static attributes (constant across time), so
        # applying it post-recurrence preserves the static-conditioning intent.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        if use_static_modulation:
            # FiLM-like gate: h_modulated = h * sigmoid(W * static_emb + b)
            self.static_gate = nn.Linear(static_embed_size, hidden_size)

        # Effective rainfall head (predict excess rainfall R_t)
        self.rain_head = nn.Sequential(
            nn.Linear(hidden_size + static_embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # UHconv module (only if used)
        if use_uhconv:
            self.uhconv = UHconv(static_dim=static_embed_size, kernel_size=uh_kernel_size,
                                  hidden_size=uh_hidden_size)

        # Baseflow head (slow component)
        self.baseflow_head = nn.Sequential(
            nn.Linear(hidden_size + static_embed_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Storage head (for mass-balance)
        if use_mass_balance:
            self.storage_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )
            self.et_head = nn.Sequential(
                nn.Linear(hidden_size + static_embed_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, forcing: torch.Tensor, static: torch.Tensor,
                return_states: bool = False) -> Dict[str, torch.Tensor]:
        """forcing: (B, T, F), static: (B, A). Returns dict with at least 'q_norm' (B,)."""
        B, T, _ = forcing.shape
        s_emb = self.static_encoder(static)  # (B, S)

        # LSTM encoder (cuDNN-accelerated, no Python for-loop)
        out, _ = self.lstm(forcing)            # (B, T, H)
        if self.use_static_modulation:
            # Apply static-attribute-dependent multiplicative gate to all hidden states.
            # gate: (B, H) -> broadcast over time -> (B, T, H)
            gate = torch.sigmoid(self.static_gate(s_emb))  # (B, H)
            hidden_states = out * gate.unsqueeze(1)         # (B, T, H)
        else:
            hidden_states = out
        h_last = hidden_states[:, -1, :]  # (B, H)

        # Effective rainfall head (B, T)
        rain_in = torch.cat([hidden_states, s_emb.unsqueeze(1).expand(-1, T, -1)], dim=-1)
        eff_rain = F.softplus(self.rain_head(rain_in).squeeze(-1))  # (B, T), >= 0

        # Surface runoff via UHconv or simple sum
        if self.use_uhconv:
            q_surface_series = self.uhconv(eff_rain, s_emb)  # (B, T)
        else:
            # Without routing: surface runoff = effective rainfall at same time-step
            q_surface_series = eff_rain
        q_surface_last = q_surface_series[:, -1]  # (B,)

        # Baseflow (slow component)
        baseflow_in = torch.cat([h_last, s_emb], dim=-1)
        q_base = F.softplus(self.baseflow_head(baseflow_in).squeeze(-1))  # (B,), >= 0

        # Total raw Q (mm/day) at last time-step — physical output of the model.
        # The training loop normalises this using per-basin (q_mean, q_std) to
        # produce q_norm for the regression loss. q_raw is used directly for the
        # mass-balance loss (physical consistency in mm/day).
        q_raw_last = q_surface_last + q_base   # (B,)

        # Storage + ET series (for mass-balance loss)
        result = {"q_norm": q_raw_last, "q_raw": q_raw_last,
                  "q_surface": q_surface_last, "q_base": q_base,
                  "eff_rain": eff_rain}
        if self.use_mass_balance:
            storage_series = self.storage_head(hidden_states).squeeze(-1)  # (B, T)
            et_in = torch.cat([hidden_states, s_emb.unsqueeze(1).expand(-1, T, -1)], dim=-1)
            et_series = F.softplus(self.et_head(et_in).squeeze(-1))  # (B, T), >= 0
            # dS/dt = S_t - S_{t-1} (per-day rate)
            ds_dt = storage_series[:, 1:] - storage_series[:, :-1]  # (B, T-1)
            # Pad to length T for convenience (first step dS/dt = 0)
            ds_dt_full = torch.cat([torch.zeros_like(ds_dt[:, :1]), ds_dt], dim=1)  # (B, T)
            result["storage"] = storage_series
            result["et"] = et_series
            result["ds_dt"] = ds_dt_full
            # Mass-balance outputs at last time-step (for loss)
            result["ds_dt_last"] = ds_dt_full[:, -1]
            result["et_last"] = et_series[:, -1]
            result["precip_last"] = forcing[:, -1, 0]   # PRCP at last day (assumes col 0 = PRCP after norm)

        if return_states:
            result["hidden_states"] = hidden_states
            result["h_last"] = h_last
            result["static_emb"] = s_emb
        return result


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "EA_LSTM": EALSTMModelClean,
    "MTS_LSTM": MTS_LSTMModel,
    "Transformer": TransformerModel,
    # Phys_LSTM uses LSTMModel; the physics loss is enabled in train loop
    "Phys_LSTM": LSTMModel,
    # UH_LSTM = PINN-UHconv with mass_balance off (ablation bridge)
    # PINN_UHConv is the full model
    # NOTE: Both map to PINNUHConvModel; the use_mass_balance flag is set in
    # build_model() based on the model name. Using the class directly (not a
    # lambda) so inspect.signature can see the constructor parameters.
    "UH_LSTM": PINNUHConvModel,
    "PINN_UHConv": PINNUHConvModel,
}


def build_model(name: str, input_size: int, static_input_size: int) -> nn.Module:
    """Factory to construct a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    cls = MODEL_REGISTRY[name]
    # Different models accept different kwargs; use inspection to pass only supported ones
    import inspect
    sig = inspect.signature(cls)
    kwargs = {
        "input_size": input_size,
        "static_input_size": static_input_size,
        "hidden_size": MODEL_CFG.hidden_size,
        "num_layers": MODEL_CFG.num_layers,
        "dropout": MODEL_CFG.dropout,
    }
    # PINN_UHConv / UH_LSTM also take UH-related kwargs
    if name in ("PINN_UHConv", "UH_LSTM"):
        kwargs.update({
            "static_embed_size": MODEL_CFG.static_size,
            "uh_kernel_size": MODEL_CFG.uh_kernel_size,
            "uh_hidden_size": MODEL_CFG.uh_hidden_size,
            "use_uhconv": MODEL_CFG.use_uhconv,
            "use_mass_balance": MODEL_CFG.use_mass_balance if name == "PINN_UHConv" else False,
            "use_static_modulation": MODEL_CFG.use_static_modulation,
        })
    # Filter to only kwargs accepted by the constructor
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    model = cls(**filtered)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test
    print("Model smoke test:")
    B, T, F_dim = 4, 30, 7
    A = 16
    forcing = torch.randn(B, T, F_dim)
    static = torch.randn(B, A)
    for name in ["LSTM", "EA_LSTM", "MTS_LSTM", "Transformer", "Phys_LSTM", "UH_LSTM", "PINN_UHConv"]:
        model = build_model(name, input_size=F_dim, static_input_size=A)
        out = model(forcing, static, return_states=True)
        n_params = count_parameters(model)
        keys = list(out.keys())
        print(f"  {name:15s} | params={n_params:>8d} | q_norm.shape={tuple(out['q_norm'].shape)} | keys={keys}")
