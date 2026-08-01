"""Global configuration for the PINN-UHconv hydrology runoff prediction project.

This module defines all paths, hyper-parameters, dataset splits and experiment settings.
Reviewer can reproduce all experiments by running scripts that import this config.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Kaggle hydromtl archive (CAMELS-US Daymet forcing + streamflow)
KAGGLE_DATA_DIR = DATA_DIR / "kaggle_hydromtl"
KAGGLE_ZIP = KAGGLE_DATA_DIR / "hydromtl.zip"
# After extraction, expected layout:
#   kaggle_hydromtl/camels/camels_us/basin_timeseries_v1p2_metForcing_obsFlow/
#       basin_dataset_public_v1p2/basin_mean_forcing/daymet/<HUC2>/<BASIN_ID>_lump_cida_forcing_leap.txt
#       basin_dataset_public_v1p2/usgs_streamflow/<BASIN_ID>_usgs_streamflow_qc.txt
#   kaggle_hydromtl/camels/camels_us/basin_set_full_res/HCDN_nhru_final_671.shp
CAMELS_ROOT = KAGGLE_DATA_DIR / "camels" / "camels_us"
FORCING_ROOT = CAMELS_ROOT / "basin_timeseries_v1p2_metForcing_obsFlow" / "basin_dataset_public_v1p2"
DAYMET_DIR = FORCING_ROOT / "basin_mean_forcing" / "daymet"
STREAMFLOW_DIR = FORCING_ROOT / "usgs_streamflow"

# Basin attributes (may need separate download if hydromtl lacks them)
ATTRIBUTES_DIR = CAMELS_ROOT  # where camels_topo.txt etc. live if present

# Output sub-dirs
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Dataset configuration
# -----------------------------------------------------------------------------
@dataclass
class DataConfig:
    # Date range (CAMELS-US Daymet: 1980-2014)
    train_start: str = "1980-10-01"
    train_end: str = "1995-09-30"        # 15 years train
    val_start: str = "1995-10-01"
    val_end: str = "2000-09-30"          # 5 years val
    test_start: str = "2000-10-01"
    test_end: str = "2010-09-30"         # 10 years test

    # Forcing variables (Daymet: 5 variables, will add 1 day-of-year sin/cos)
    forcing_vars: Tuple[str, ...] = ("PRCP", "TMEAN", "TMAX", "TMIN", "DAYL")

    # Static attributes (subset of CAMELS attributes that influence hydrology)
    static_attrs: Tuple[str, ...] = (
        "area_gages2",       # drainage area
        "elev_mean",         # mean elevation
        "slope_mean",        # mean slope
        "forest_frac",       # forest fraction
        "lai_max",           # max leaf area index
        "lai_diff",          # lai difference
        "dom_soil_type",     # dominant soil (categorical -> embedded)
        "soil_depth_pelletier",  # soil depth
        "soil_porosity",     # soil porosity
        "soil_conductivity", # saturated hydraulic conductivity
        "max_water_content", # max water content
        "geol_1st_class",    # dominant geology (categorical)
        "geol_porostiy",     # subsurface porosity
        "aridity",           # climatic aridity index
        "high_prec_freq",    # frequency of high precip days
        "low_prec_freq",     # frequency of low precip days
    )

    # Sequence length (days). 180 = look back ~6 months (balance of speed/context)
    seq_length: int = 180
    forecast_horizon: int = 1   # 1-day-ahead prediction
    batch_size: int = 256

    # Number of basins to use (671 in CAMELS, but for speed we sample a subset first)
    # Set to None for all basins; use a subset for rapid iteration
    n_basins: Optional[int] = 100   # 100 basins for quick experiments; set None for all 671

    # Train/val/test basin split (held-out basins for regionalisation experiment)
    test_basin_fraction: float = 0.15   # 15% basins held out for ungaged test

    # Normalisation
    normalize_forcing: str = "per-basin"  # 'per-basin' or 'global'
    normalize_static: str = "global"


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str = "PINN_UHConv"

    # LSTM encoder
    input_size: int = 7        # 5 forcing + 2 (sin/cos day-of-year)
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.3

    # Static attribute embedding (for EA-LSTM / static modulation)
    static_size: int = 16      # embedded dim
    static_input_size: int = 16   # actual count of static attrs used (filled at runtime)

    # UHconv (differentiable unit hydrograph convolution)
    uh_kernel_size: int = 60       # max travel time (days)
    uh_hidden_size: int = 64       # hidden for alpha/beta prediction MLP

    # Mass-balance loss
    lambda_mass: float = 0.01      # weight of mass-balance constraint (small for stability)
    lambda_extreme: float = 0.5    # weight of extreme-event loss (consistent with all cached baselines; clamping in train.py prevents instability)
    extreme_quantile: float = 0.95 # 95th percentile -> "extreme flood"

    # Effective-rainfall head (predict excess rainfall R from P, S, etc.)
    use_uhconv: bool = True
    use_mass_balance: bool = True
    use_static_modulation: bool = True
    use_extreme_weighting: bool = True


# -----------------------------------------------------------------------------
# Training configuration
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    seed: int = 42
    n_seeds: int = 5                 # 5 random seeds for statistical analysis
    seeds: Tuple[int, ...] = (42, 2024, 7, 123, 999)

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_nse"

    # Mixed precision for speed
    use_amp: bool = True

    # Device
    device: str = "cuda"             # 'cuda' or 'cpu'

    # Number of worker threads for DataLoader
    num_workers: int = 0             # Windows: 0 = main process (safer)

    # Output
    results_dir: Path = RESULTS_DIR
    log_dir: Path = LOG_DIR

    # Validation
    val_every_n_epochs: int = 1


# -----------------------------------------------------------------------------
# Experiment configuration (which baselines / ablations to run)
# -----------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # Baselines to compare (≥5 per user rules)
    baselines: Tuple[str, ...] = (
        "LSTM",            # Kratzert et al. 2018 (basic LSTM)
        "EA_LSTM",         # Kratzert et al. 2018 (embedding-approach LSTM)
        "MTS_LSTM",        # gauch et al. 2021 (multi-temporal scale)
        "Transformer",     # attention-based baseline
        "Phys_LSTM",       # LSTM + mass-balance loss (physics-guided)
        "UH_LSTM",         # LSTM + UHconv but without mass-balance (ablation bridge)
    )
    our_method: str = "PINN_UHConv"

    # Ablation: component-level (each removes one core component from PINN_UHConv)
    ablation_components: Tuple[str, ...] = (
        "full",                    # full PINN_UHConv
        "no_uhconv",               # remove UHconv module
        "no_mass_balance",         # remove mass-balance loss
        "no_static_modulation",    # remove static attribute modulation
        "no_extreme_weighting",    # remove extreme-event loss weighting
    )

    # Hyper-parameter sensitivity sweep
    sensitivity_params: dict = field(default_factory=lambda: {
        "lambda_mass": [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],
        "uh_kernel_size": [10, 20, 30, 60, 90, 120],
        "hidden_size": [32, 64, 128, 256],
        "lambda_extreme": [0.0, 0.1, 0.3, 0.5, 1.0, 2.0],
        "seq_length": [60, 90, 180, 365, 540],
    })

    # Robustness analysis
    robustness_noise_levels: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)  # Gaussian noise std on forcing
    robustness_missing_rates: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30) # random missing forcing
    robustness_unseen_basin: bool = True    # held-out basin regionalisation


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# Default config singletons
DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
EXP_CFG = ExperimentConfig()
