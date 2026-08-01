"""Download entire owenyy/hydromtl Kaggle dataset (contains CAMELS-US Daymet forcing + streamflow)."""
import os
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

if "KAGGLE_API_TOKEN" not in os.environ:
    print("Error: Set KAGGLE_API_TOKEN environment variable first.")
    print("Get your token from https://www.kaggle.com/settings")
    sys.exit(1)
api = KaggleApi()
api.authenticate()

OUT_DIR = Path(r"D:\tourism\03_hydrology_runoff\data\kaggle_hydromtl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_PATH = OUT_DIR / "hydromtl.zip"

print(f"Downloading owenyy/hydromtl to {ZIP_PATH} ...")
print(f"This dataset contains CAMELS-US Daymet forcing + streamflow observations")
print(f"Size unknown, likely 1-3 GB. Please wait...")
sys.stdout.flush()

start = time.time()
try:
    # api.dataset_download_files returns the downloaded zip path
    # If unzip=True, extracts to OUT_DIR; we keep zip first for verification
    api.dataset_download_files(
        dataset="owenyy/hydromtl",
        path=str(OUT_DIR),
        unzip=False,
        quiet=False,
        force=False,  # don't re-download if exists
    )
    elapsed = time.time() - start
    print(f"\n[OK] Download finished in {elapsed/60:.1f} min")
    # Check resulting file(s)
    for f in OUT_DIR.iterdir():
        sz = f.stat().st_size
        print(f"  {f.name} | {sz/1e6:.2f} MB")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
