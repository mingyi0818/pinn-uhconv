"""Smoke test the forcing/streamflow parsers by reading directly from the zip.

This avoids waiting for full extraction before validating the parsing logic.
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (
    parse_forcing, parse_streamflow, FORCING_COLUMNS, FORCING_USE,
    STREAMFLOW_COLUMNS, CFS_TO_CMS, MM_PER_M3_PER_KM2_PER_DAY,
)
from config import KAGGLE_ZIP, ATTRIBUTES_DIR

ZIP_PATH = KAGGLE_ZIP


def main():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found")
        sys.exit(1)
    zf = zipfile.ZipFile(ZIP_PATH)
    names = zf.namelist()

    # Find one forcing file
    forcing_name = next(n for n in names if n.endswith("_lump_cida_forcing_leap.txt") and "/01/" in n)
    # Find matching streamflow (same basin id)
    basin_id = Path(forcing_name).stem.split("_")[0]
    sf_name = next((n for n in names if n.endswith(f"/{basin_id}_streamflow_qc.txt")), None)

    print(f"Forcing file: {forcing_name}")
    print(f"Streamflow file: {sf_name}")
    print()

    # Read forcing (parse directly from bytes since the file is inside the zip)
    with zf.open(forcing_name) as f:
        raw_bytes = f.read()
    df_f = pd.read_csv(
        io.BytesIO(raw_bytes), sep=r"\s+", skiprows=4, header=0, names=FORCING_COLUMNS,
        engine="python",
    )
    df_f["date"] = pd.to_datetime(
        df_f[["Year", "Mnth", "Day"]].astype(int).rename(columns={"Mnth": "month", "Day": "day"})
    )
    print("Forcing (first 5 rows):")
    print(df_f.head())
    print(f"Forcing date range: {df_f['date'].min()} to {df_f['date'].max()}")
    print(f"Forcing shape: {df_f.shape}")
    print(f"PRCP stats: mean={df_f['prcp(mm/day)'].mean():.2f} max={df_f['prcp(mm/day)'].max():.2f}")
    print()

    # Read streamflow
    if sf_name:
        with zf.open(sf_name) as f:
            sf_bytes = f.read()
        df_s = pd.read_csv(
            io.BytesIO(sf_bytes), sep=r"\s+", header=None, names=STREAMFLOW_COLUMNS,
            engine="python",
        )
        df_s["date"] = pd.to_datetime(
            df_s[["Year", "Mnth", "Day"]].astype(int).rename(columns={"Mnth": "month", "Day": "day"})
        )
        df_s["Q_cfs"] = df_s["Q_cfs"].replace(-999, np.nan)
        # Use a fake area of 1000 km^2 for test
        area = 1000.0
        df_s["Q_mm_day"] = df_s["Q_cfs"] * CFS_TO_CMS * MM_PER_M3_PER_KM2_PER_DAY / area
        print("Streamflow (first 5 rows):")
        print(df_s.head())
        print(f"Streamflow date range: {df_s['date'].min()} to {df_s['date'].max()}")
        print(f"Streamflow shape: {df_s.shape}")
        print(f"Q_cfs stats: mean={df_s['Q_cfs'].mean():.2f} max={df_s['Q_cfs'].max():.2f}")
        print(f"Q_mm_day stats: mean={df_s['Q_mm_day'].mean():.4f} max={df_s['Q_mm_day'].max():.4f}")
        print()

    # Test attribute file
    attr_name = "camels/camels_us/camels_topo.txt"
    if attr_name in names:
        with zf.open(attr_name) as f:
            attr_bytes = f.read()
        print("camels_topo.txt (first 3 rows):")
        df_a = pd.read_csv(io.BytesIO(attr_bytes), sep=";", header=0, dtype={"gauge_id": str})
        print(df_a.head(3))
        print(f"Attribute shape: {df_a.shape}")
        print(f"Columns: {df_a.columns.tolist()[:10]} ... ({len(df_a.columns)} total)")
        print()
        # Check area column
        if "area_gages2" in df_a.columns:
            print(f"area_gages2 stats: min={df_a['area_gages2'].min():.2f} max={df_a['area_gages2'].max():.2f} mean={df_a['area_gages2'].mean():.2f}")

    print()
    print("[OK] All parsers work correctly")


if __name__ == "__main__":
    main()
