"""CAMELS-US data loader for the PINN-UHconv hydrology project.

This module parses the raw CAMELS-US files (Daymet forcing + USGS streamflow +
basin attributes) into PyTorch-ready tensors.

CAMELS-US file format conventions:
  - Forcing: <BASIN_ID>_lump_cida_forcing_leap.txt
      First 4 lines are header, then tab-separated columns:
      Year, Mnth, Day, Hr, PRCP(mm/day), TMEAN(degC), TMAX, TMIN, DAYL(s), PET(mm/day)
  - Streamflow: <BASIN_ID>_usgs_streamflow_qc.txt
      Space-separated columns: Year Mnth Day Q(cfs) QC_flag
      1 cfs = 0.028316846592 m^3/s; we convert to mm/day using basin area.
  - Attributes: camels_<category>.txt (topo/clim/hydro/soil/vege/geol/name)
      Tab-separated, basin ID in first column.

References:
  Newman et al. 2015 HESS - dataset description
  Addor et al. 2017 HESS - catchment attributes
"""
from __future__ import annotations

import io
import os
import re
import sys
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_CFG, MODEL_CFG, TRAIN_CFG,
    KAGGLE_DATA_DIR, KAGGLE_ZIP, CAMELS_ROOT, DAYMET_DIR, STREAMFLOW_DIR,
    ATTRIBUTES_DIR, CACHE_DIR,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CFS_TO_CMS = 0.028316846592  # 1 cubic foot/sec -> m^3/s
MM_PER_M3_PER_KM2_PER_DAY = 86400.0 / 1000.0  # m^3/s -> mm/day given area in km^2

FORCING_COLUMNS = ["Year", "Mnth", "Day", "Hr", "dayl(s)", "prcp(mm/day)",
                   "srad(W/m2)", "swe(mm)", "tmax(C)", "tmin(C)", "vp(Pa)"]
# Variables actually used as model inputs (subset; we'll compute TMEAN from tmax/tmin)
FORCING_USE = ["prcp(mm/day)", "tmax(C)", "tmin(C)", "dayl(s)", "srad(W/m2)", "swe(mm)", "vp(Pa)"]

STREAMFLOW_COLUMNS = ["Year", "Mnth", "Day", "Q_cfs", "QC_flag"]

# HUC2 directories in Daymet: 01-18 (we glob all subdirs)
HUC2_DIRS = [f"{i:02d}" for i in range(1, 19)]


# -----------------------------------------------------------------------------
# 1. Extraction
# -----------------------------------------------------------------------------
def extract_kaggle_zip(force: bool = False) -> None:
    """Extract hydromtl.zip if not yet extracted."""
    if not KAGGLE_ZIP.exists():
        raise FileNotFoundError(
            f"Kaggle zip not found at {KAGGLE_ZIP}. "
            f"Run src/kaggle_download_hydromtl.py first."
        )

    # Check if already extracted (look for camels/camels_us dir)
    if CAMELS_ROOT.exists() and not force:
        n_forcing = sum(1 for _ in DAYMET_DIR.rglob("*_lump_cida_forcing_leap.txt")) if DAYMET_DIR.exists() else 0
        if n_forcing > 100:
            print(f"[extract] Already extracted: {n_forcing} forcing files in {DAYMET_DIR}")
            return
        print(f"[extract] Partial extraction ({n_forcing} forcing files), re-extracting...")

    print(f"[extract] Extracting {KAGGLE_ZIP} -> {KAGGLE_DATA_DIR} ...")
    with zipfile.ZipFile(KAGGLE_ZIP, "r") as zf:
        # Get member list to estimate progress
        members = zf.namelist()
        print(f"[extract] {len(members)} members in zip")
        # Extract all (zipfile handles nested dirs)
        zf.extractall(KAGGLE_DATA_DIR)
    print(f"[extract] Done. Verifying...")
    n_forcing = sum(1 for _ in DAYMET_DIR.rglob("*_lump_cida_forcing_leap.txt")) if DAYMET_DIR.exists() else 0
    print(f"[extract] {n_forcing} forcing files found after extraction")


# -----------------------------------------------------------------------------
# 2. Basin discovery
# -----------------------------------------------------------------------------
def discover_basins() -> List[str]:
    """Find all basin IDs that have both Daymet forcing and streamflow files."""
    forcing_basins = {}
    for forcing_path in DAYMET_DIR.rglob("*_lump_cida_forcing_leap.txt"):
        basin_id = forcing_path.stem.split("_")[0]
        forcing_basins[basin_id] = forcing_path
    print(f"[discover] {len(forcing_basins)} basins with Daymet forcing")

    streamflow_basins = {}
    if STREAMFLOW_DIR.exists():
        # CAMELS-US filenames: <BASIN_ID>_streamflow_qc.txt (NOT _usgs_streamflow_qc.txt)
        # Files are in HUC2 subdirs (e.g. usgs_streamflow/01/01013500_streamflow_qc.txt)
        for sf_path in STREAMFLOW_DIR.rglob("*_streamflow_qc.txt"):
            basin_id = sf_path.stem.split("_")[0]
            streamflow_basins[basin_id] = sf_path
    print(f"[discover] {len(streamflow_basins)} basins with USGS streamflow")

    common = sorted(set(forcing_basins) & set(streamflow_basins))
    print(f"[discover] {len(common)} basins with both forcing + streamflow")
    return common


# -----------------------------------------------------------------------------
# 3. File parsing
# -----------------------------------------------------------------------------
def parse_forcing(path: Path) -> pd.DataFrame:
    """Parse a CAMELS Daymet forcing file.

    File format (CAMELS-US v1.2, Daymet product):
      Line 1: lat (float)
      Line 2: elevation (m, float)
      Line 3: area (m^2? float)
      Line 4: column header: 'Year Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)'
      Line 5+: data rows, whitespace-separated (Year/Mnth/Day/Hr space-separated, rest tab-separated)
    """
    df = pd.read_csv(
        path, sep=r"\s+", skiprows=4, header=0, names=FORCING_COLUMNS,
        engine="python",
    )
    # Build date column (rename to expected names for pd.to_datetime)
    df["date"] = pd.to_datetime(
        df[["Year", "Mnth", "Day"]].astype(int).rename(
            columns={"Mnth": "month", "Day": "day"}
        )
    )
    # Compute TMEAN = (TMAX + TMIN) / 2 for convenience
    df["tmean(C)"] = (df["tmax(C)"] + df["tmin(C)"]) / 2.0
    return df


def parse_streamflow(path: Path, basin_area_km2: Optional[float] = None) -> pd.DataFrame:
    """Parse a USGS streamflow QC file. Convert cfs -> mm/day if area given."""
    df = pd.read_csv(
        path, sep=r"\s+", header=None, names=STREAMFLOW_COLUMNS,
        engine="python",
    )
    df["date"] = pd.to_datetime(
        df[["Year", "Mnth", "Day"]].astype(int).rename(
            columns={"Mnth": "month", "Day": "day"}
        )
    )
    # Replace -999 with NaN
    df["Q_cfs"] = df["Q_cfs"].replace(-999, np.nan)
    if basin_area_km2 is not None and basin_area_km2 > 0:
        # cfs -> m^3/s -> mm/day
        df["Q_mm_day"] = df["Q_cfs"] * CFS_TO_CMS * MM_PER_M3_PER_KM2_PER_DAY / basin_area_km2
    return df


# -----------------------------------------------------------------------------
# 4. Basin attributes loading
# -----------------------------------------------------------------------------
ATTRIBUTE_FILES = {
    "topo": "camels_topo.txt",
    "clim": "camels_clim.txt",
    "hydro": "camels_hydro.txt",
    "soil": "camels_soil.txt",
    "vege": "camels_vege.txt",
    "geol": "camels_geol.txt",
    "name": "camels_name.txt",
}


def load_all_attributes() -> pd.DataFrame:
    """Load all CAMELS attribute files and merge into one DataFrame indexed by basin ID."""
    frames = []
    for cat, fname in ATTRIBUTE_FILES.items():
        path = ATTRIBUTES_DIR / fname
        if not path.exists():
            print(f"[attrs] Missing {path}, skipping {cat}")
            continue
        df = pd.read_csv(path, sep=";", header=0, dtype={"gauge_id": str})
        df = df.rename(columns={df.columns[0]: "basin_id"})
        df = df.set_index("basin_id")
        # Prefix columns with category to avoid collisions
        df = df.rename(columns={c: f"{cat}.{c}" for c in df.columns})
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    attrs = pd.concat(frames, axis=1)
    print(f"[attrs] Loaded {attrs.shape[1]} attributes for {len(attrs)} basins")
    return attrs


# -----------------------------------------------------------------------------
# 5. Per-basin combined time series
# -----------------------------------------------------------------------------
def load_basin_timeseries(
    basin_id: str,
    forcing_path: Path,
    streamflow_path: Path,
    basin_area_km2: Optional[float] = None,
) -> pd.DataFrame:
    """Load forcing + streamflow for a single basin, aligned by date."""
    forcing = parse_forcing(forcing_path)
    streamflow = parse_streamflow(streamflow_path, basin_area_km2=basin_area_km2)
    # Merge on date
    merged = pd.merge(forcing, streamflow[["date", "Q_cfs", "Q_mm_day"]], on="date", how="inner")
    merged = merged.set_index("date").sort_index()
    return merged


# -----------------------------------------------------------------------------
# 6. Train/val/test split
# -----------------------------------------------------------------------------
def split_basins(
    basin_ids: List[str], test_fraction: float = 0.15, seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split basins into train/val/test by region (held-out basins for ungaged test)."""
    rng = np.random.RandomState(seed)
    n = len(basin_ids)
    indices = rng.permutation(n)
    n_test = int(n * test_fraction)
    n_val = int(n * 0.15)  # 15% val
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    return (
        [basin_ids[i] for i in train_idx],
        [basin_ids[i] for i in val_idx],
        [basin_ids[i] for i in test_idx],
    )


def temporal_split(
    df: pd.DataFrame, train_start: str, train_end: str,
    val_start: str, val_end: str, test_start: str, test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a per-basin time series by date."""
    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]
    return train, val, test


# -----------------------------------------------------------------------------
# 7. Preprocessing pipeline (builds tensors for training)
# -----------------------------------------------------------------------------
def build_per_basin_statistics(
    basin_data: Dict[str, pd.DataFrame], attrs: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute per-basin mean/std for forcing variables + streamflow.
    Uses only training-period data (per-basin normalisation, standard in CAMELS literature).
    """
    stats = {}
    for bid, df in basin_data.items():
        train_df = df.loc[DATA_CFG.train_start:DATA_CFG.train_end]
        if len(train_df) == 0:
            continue
        s = {}
        for col in FORCING_USE:
            s[f"{col}_mean"] = float(train_df[col].mean())
            s[f"{col}_std"] = float(train_df[col].std() + 1e-8)
        s["Q_mm_day_mean"] = float(train_df["Q_mm_day"].mean())
        s["Q_mm_day_std"] = float(train_df["Q_mm_day"].std() + 1e-8)
        stats[bid] = s
    return stats


def normalize_static_attributes(attrs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-normalise numeric static attributes and label-encode categorical columns.

    Categorical columns (object dtype or low-cardinality int) are converted to
    integer codes so the entire DataFrame is numeric (PyTorch-ready).
    Returns (normalized_df, mean, std).
    """
    norm = attrs.copy()
    numeric_cols = []
    categorical_cols = []
    for c in attrs.columns:
        if attrs[c].dtype.kind in "iufc":
            # Numeric: treat as categorical only if it's int with low cardinality
            nunique = attrs[c].nunique()
            if nunique <= 20 and attrs[c].dtype.kind == "i":
                categorical_cols.append(c)
            else:
                numeric_cols.append(c)
        else:
            # Object/string dtype -> categorical
            categorical_cols.append(c)
    # Label-encode categoricals
    for c in categorical_cols:
        norm[c] = pd.Categorical(norm[c]).codes.astype(np.float32)
        # Replace -1 (NaN) with 0
        norm[c] = norm[c].replace(-1, 0)
    # Z-normalise numeric columns
    mean = norm[numeric_cols].mean()
    std = norm[numeric_cols].std().replace(0, 1.0)
    norm[numeric_cols] = (norm[numeric_cols] - mean) / std
    # Fill any remaining NaN with 0
    norm = norm.fillna(0)
    print(f"[norm-static] {len(numeric_cols)} numeric + {len(categorical_cols)} categorical (label-encoded)")
    return norm, mean, std


# -----------------------------------------------------------------------------
# 8. PyTorch Dataset
# -----------------------------------------------------------------------------
class CAMELSDataset:
    """In-memory dataset of sliding-window sequences for a set of basins.
    Pre-loads all basins into memory (CAMELS is small enough: ~1.5M sample-days total).
    """

    def __init__(
        self,
        basin_ids: List[str],
        period: str,                # 'train', 'val', or 'test'
        basin_data: Dict[str, pd.DataFrame],
        basin_stats: Dict[str, Dict[str, float]],
        static_attrs: pd.DataFrame,
        seq_length: int = DATA_CFG.seq_length,
        stride: int = 1,
    ):
        self.basin_ids = basin_ids
        self.period = period
        self.seq_length = seq_length
        self.stride = stride
        self.basin_data = basin_data
        self.basin_stats = basin_stats
        self.static_attrs = static_attrs

        # Build index of (basin_id, start_date_idx) tuples
        self.index: List[Tuple[str, int]] = []
        if period == "train":
            start, end = DATA_CFG.train_start, DATA_CFG.train_end
        elif period == "val":
            start, end = DATA_CFG.val_start, DATA_CFG.val_end
        elif period == "test":
            start, end = DATA_CFG.test_start, DATA_CFG.test_end
        else:
            raise ValueError(f"Unknown period: {period}")

        for bid in basin_ids:
            df = basin_data[bid].loc[start:end]
            n = len(df)
            # Each sequence has length seq_length, predicting last day's Q.
            # Skip sequences where the target (last day's Q) is NaN (missing data).
            q_vals = df["Q_mm_day"].values
            for i in range(0, n - seq_length, stride):
                if not np.isnan(q_vals[i + seq_length - 1]):
                    self.index.append((bid, i))
        print(f"[dataset] {period}: {len(self.index)} sequences across {len(basin_ids)} basins")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        bid, start_i = self.index[idx]
        df = self.basin_data[bid]
        stats = self.basin_stats[bid]

        # Slice forcing sequence + target
        window = df.iloc[start_i:start_i + self.seq_length]
        # Fill any NaN in forcing with 0 (Daymet rarely has gaps, but safeguard)
        window_filled = window.fillna(0.0)
        # Normalise forcing
        forcing = np.zeros((self.seq_length, len(FORCING_USE) + 2), dtype=np.float32)  # +2 for sin/cos doy
        for j, col in enumerate(FORCING_USE):
            m, s = stats[f"{col}_mean"], stats[f"{col}_std"]
            forcing[:, j] = ((window_filled[col].values - m) / s).astype(np.float32)
        # Day-of-year sin/cos
        doy = window_filled.index.dayofyear.values.astype(np.float32)
        forcing[:, -2] = np.sin(2 * np.pi * doy / 365.0)
        forcing[:, -1] = np.cos(2 * np.pi * doy / 365.0)

        # Target Q (normalised)
        q_mean = stats["Q_mm_day_mean"]
        q_std = stats["Q_mm_day_std"]
        q_raw = window_filled["Q_mm_day"].values[-1].astype(np.float32)   # predict last day
        q_norm = (q_raw - q_mean) / (q_std + 1e-8)

        # Raw PRCP at the prediction day (last day of the window) for mass-balance loss.
        prcp_raw = window_filled["prcp(mm/day)"].values[-1].astype(np.float32)
        prcp_series_raw = window_filled["prcp(mm/day)"].values.astype(np.float32)

        # Static attributes (numeric normalised, categorical raw)
        if bid in self.static_attrs.index:
            static = self.static_attrs.loc[bid].values.astype(np.float32)
        else:
            static = np.zeros(self.static_attrs.shape[1], dtype=np.float32)

        # Also provide raw Q for computing NSE on un-normalised scale
        return {
            "forcing": forcing,                  # (seq_length, n_forcing+2)
            "target_norm": q_norm,               # scalar
            "target_raw": q_raw,                 # scalar (mm/day)
            "q_mean": np.float32(q_mean),
            "q_std": np.float32(q_std),
            "static": static,                    # (n_static,)
            "basin_id": bid,
            "date_str": window.index[-1].strftime("%Y-%m-%d"),
            "precip_raw": prcp_raw,              # scalar (mm/day) - PRCP at prediction day
            "precip_series_raw": prcp_series_raw,  # (seq_length,) - raw PRCP for whole window
        }


# -----------------------------------------------------------------------------
# 9. End-to-end pipeline: load -> cache -> Dataset
# -----------------------------------------------------------------------------
def load_full_pipeline(
    n_basins: Optional[int] = None,
    cache: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, Dict[str, Dict[str, float]], List[str]]:
    """Top-level convenience: extract zip, discover basins, load all time series + attrs.
    Returns (basin_data, static_attrs, basin_stats, basin_ids).
    """
    cache_path = CACHE_DIR / "full_pipeline.npz"
    cache_meta_path = CACHE_DIR / "full_pipeline_meta.json"

    if cache and cache_path.exists() and cache_meta_path.exists():
        print(f"[pipeline] Loading from cache: {cache_path}")
        try:
            return _load_from_cache(cache_path, cache_meta_path)
        except Exception as e:
            print(f"[pipeline] Cache load failed ({e}), rebuilding...")

    # Extract
    extract_kaggle_zip()

    # Discover basins
    all_basins = discover_basins()
    if not all_basins:
        raise RuntimeError("No basins discovered. Check data extraction.")

    # Subsample if requested
    if n_basins is not None and n_basins < len(all_basins):
        rng = np.random.RandomState(42)
        all_basins = [all_basins[i] for i in rng.choice(len(all_basins), n_basins, replace=False)]
        print(f"[pipeline] Subsampled to {len(all_basins)} basins")

    # Load attributes (need area for cfs->mm/day conversion)
    attrs = load_all_attributes()

    # Load all basin time series
    basin_data: Dict[str, pd.DataFrame] = {}
    for i, bid in enumerate(all_basins):
        # Find files (basin ID may have leading 0s, 8 digits)
        forcing_files = list(DAYMET_DIR.rglob(f"{bid}_lump_cida_forcing_leap.txt"))
        sf_files = list(STREAMFLOW_DIR.rglob(f"{bid}_streamflow_qc.txt")) if STREAMFLOW_DIR.exists() else []
        if not forcing_files or not sf_files:
            continue
        # Get basin area for unit conversion
        if not attrs.empty and bid in attrs.index and "topo.area_gages2" in attrs.columns:
            area = float(attrs.loc[bid, "topo.area_gages2"])
        else:
            area = None
        try:
            df = load_basin_timeseries(bid, forcing_files[0], sf_files[0], basin_area_km2=area)
            if len(df) > 365 * 5:  # need at least 5 years of data
                basin_data[bid] = df
        except Exception as e:
            print(f"[pipeline] Failed basin {bid}: {e}")
        if (i + 1) % 100 == 0:
            print(f"[pipeline] Loaded {i+1}/{len(all_basins)} basins")

    print(f"[pipeline] Successfully loaded {len(basin_data)} basins with full data")

    # Build basin statistics
    basin_stats = build_per_basin_statistics(basin_data, attrs)

    # Normalize static attrs
    if not attrs.empty:
        attrs_norm, _, _ = normalize_static_attributes(attrs)
    else:
        attrs_norm = pd.DataFrame()

    # Save cache
    if cache:
        _save_to_cache(cache_path, cache_meta_path, basin_data, attrs_norm, basin_stats, list(basin_data.keys()))

    return basin_data, attrs_norm, basin_stats, list(basin_data.keys())


def _save_to_cache(path, meta_path, basin_data, attrs_norm, basin_stats, basin_ids):
    """Save pipeline outputs to npz + json."""
    print(f"[cache] Saving to {path} ...")
    # Save basin_data as dict of arrays
    np.savez_compressed(
        path,
        basin_ids=np.array(basin_ids, dtype=object),
        **{f"bd_{bid}_data": df[FORCING_USE + ["Q_mm_day"]].values for bid, df in basin_data.items()},
        **{f"bd_{bid}_dates": df.index.values.astype("datetime64[s]").astype(np.int64) for bid, df in basin_data.items()},
        attrs_norm=attrs_norm.values if not attrs_norm.empty else np.array([]),
        attrs_cols=np.array(attrs_norm.columns.tolist(), dtype=object) if not attrs_norm.empty else np.array([]),
        attrs_index=np.array(attrs_norm.index.tolist(), dtype=object) if not attrs_norm.empty else np.array([]),
    )
    with open(meta_path, "w") as f:
        json.dump({"basin_ids": basin_ids, "basin_stats": basin_stats}, f)
    print(f"[cache] Done")


def _load_from_cache(path, meta_path):
    print(f"[cache] Loading from {path} ...")
    data = np.load(path, allow_pickle=True)
    with open(meta_path) as f:
        meta = json.load(f)
    basin_ids = list(meta["basin_ids"])
    basin_stats = meta["basin_stats"]
    basin_data = {}
    for bid in basin_ids:
        arr = data[f"bd_{bid}_data"]
        dates = pd.to_datetime(data[f"bd_{bid}_dates"], unit="s")
        df = pd.DataFrame(arr, index=dates, columns=FORCING_USE + ["Q_mm_day"])
        basin_data[bid] = df
    # attrs
    if data["attrs_norm"].size > 0:
        attrs_norm = pd.DataFrame(
            data["attrs_norm"],
            index=list(data["attrs_index"]),
            columns=list(data["attrs_cols"]),
        )
    else:
        attrs_norm = pd.DataFrame()
    return basin_data, attrs_norm, basin_stats, basin_ids


# -----------------------------------------------------------------------------
# Command-line interface for testing
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("CAMELS data loader test")
    print("=" * 80)
    basin_data, attrs, stats, bids = load_full_pipeline(n_basins=5)
    print()
    print(f"# basins loaded: {len(basin_data)}")
    print(f"# attrs shape:   {attrs.shape}")
    for bid in bids[:3]:
        df = basin_data[bid]
        print(f"Basin {bid}: {len(df)} days, {df.columns.tolist()}, Q range [{df['Q_mm_day'].min():.2f}, {df['Q_mm_day'].max():.2f}] mm/day")
    # Build a small train dataset
    train_basins, val_basins, test_basins = split_basins(bids)
    ds = CAMELSDataset(train_basins, "train", basin_data, stats, attrs)
    print(f"# train sequences: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample forcing shape: {sample['forcing'].shape}")
        print(f"Sample target_norm: {sample['target_norm']:.4f}")
        print(f"Sample static shape: {sample['static'].shape}")
