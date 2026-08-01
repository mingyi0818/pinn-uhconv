"""Download CAMELS-US dataset from Zenodo (record 15529996).

Strategy:
- Download all small attribute files immediately (synchronously)
- Download the large basin_timeseries zip with resume support + progress
- Skip model output files (SAC-SMA, not needed for our PINN model)

Zenodo download URL pattern:
  https://zenodo.org/records/{RECORD_ID}/files/{FILENAME}?download=1
"""
import urllib.request
import os
import sys
import time
import hashlib
from pathlib import Path

# NCAR GDEX official source (no anti-scraping, more reliable than Zenodo)
# Direct link pattern: https://gdex.ucar.edu/dataset/camels/file/{FILENAME}
DATA_DIR = Path(r"D:\tourism\03_hydrology_runoff\data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://gdex.ucar.edu/dataset/camels/file"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}

# (filename, expected_md5, expected_size_bytes)
FILES = [
    # Small attribute files (synchronous download)
    ("readme.txt", "b37d64950e9d4c5c10a8b4ef82bc6219", 1_740),
    ("camels_attributes_v2.0.pdf", "77a6c084c798a31fbd05594ee58a90c7", 91_500),
    ("camels_attributes_v2.0.xlsx", "714c68bd5bb3314ca39b14f9467bd609", 16_300),
    ("camels_name.txt", "c96491b32c4df55a31bead7ceca7d64b", 30_400),
    ("camels_topo.txt", "0f6267838c40b1507b64582433bc0b8e", 38_700),
    ("camels_clim.txt", "67f22592f3fb72c57df81358ce68458b", 100_700),
    ("camels_hydro.txt", "55ebdeb36c42ee7acdb998229c3edb3a", 122_800),
    ("camels_soil.txt", "8edb46a363a20b466a4b7105ba633767", 109_100),
    ("camels_vege.txt", "f40e843defc1e654a800be9fe5fd5090", 108_000),
    ("camels_geol.txt", "f5ce5de53eb1ea2532cda7e3b4813993", 71_600),
    # Basin shapefile (small)
    ("basin_set_full_res.zip", "958fe520f6c4062dbddbbb67cfc28985", 45_200_000),
    # Large file: time series (3.4 GB) - download with progress + resume
    ("basin_timeseries_v1p2_metForcing_obsFlow.zip", "8e9a466710e8270b58f01d332a87184f", 3_400_000_000),
]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_small(fname: str, expected_md5: str, expected_size: int) -> bool:
    """Download small file synchronously with md5 verification."""
    out = DATA_DIR / fname
    if out.exists() and out.stat().st_size > 0:
        # Verify
        actual_md5 = md5_file(out)
        if actual_md5 == expected_md5:
            print(f"[OK] {fname} already exists ({out.stat().st_size:,} B), md5 verified")
            return True
        else:
            print(f"[WARN] {fname} exists but md5 mismatch ({actual_md5} vs {expected_md5}), re-download")

    url = f"{BASE_URL}/{urllib.parse.quote(fname)}"
    print(f"[DL] {fname} <- {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(out, "wb") as f:
            f.write(data)
        actual_md5 = md5_file(out)
        if actual_md5 == expected_md5:
            print(f"[OK] {fname} downloaded ({len(data):,} B), md5 verified")
            return True
        else:
            print(f"[FAIL] {fname} md5 mismatch: {actual_md5} vs {expected_md5}")
            return False
    except Exception as e:
        print(f"[ERR] {fname}: {e}")
        return False


def download_large_resumable(fname: str, expected_md5: str, expected_size: int) -> bool:
    """Download large file with HTTP Range resume support + progress every 30s."""
    import urllib.parse
    out = DATA_DIR / fname
    tmp = out.with_suffix(out.suffix + ".part")

    # If final file exists & verified
    if out.exists():
        actual_md5 = md5_file(out)
        if actual_md5 == expected_md5:
            print(f"[OK] {fname} already complete & verified ({out.stat().st_size:,} B)")
            return True
        print(f"[WARN] {fname} exists but md5 mismatch, will overwrite")

    # Resume from .part file
    existing = tmp.stat().st_size if tmp.exists() else 0
    if existing > 0:
        print(f"[RESUME] {fname}: {existing:,} B already in .part")

    url = f"{BASE_URL}/{urllib.parse.quote(fname)}"
    headers = dict(HEADERS)
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0)) + existing
            mode = "ab" if existing > 0 and resp.status == 206 else "wb"
            if mode == "wb":
                existing = 0
                total = int(resp.headers.get("Content-Length", 0))
            print(f"[START] {fname}: resume_from={existing:,} total~={total:,} B ({total/1e9:.2f} GB)")

            downloaded = existing
            last_report = time.time()
            last_bytes = downloaded
            with open(tmp, mode) as f:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_report >= 30.0:
                        speed = (downloaded - last_bytes) / (now - last_report) / 1e6
                        pct = 100.0 * downloaded / total if total else 0
                        eta_s = (total - downloaded) / (downloaded - last_bytes) * (now - last_report) if downloaded > last_bytes else 0
                        print(f"[PROG] {fname}: {downloaded/1e9:.3f}/{total/1e9:.3f} GB ({pct:.1f}%) | {speed:.2f} MB/s | ETA {eta_s/60:.1f} min", flush=True)
                        last_report = now
                        last_bytes = downloaded
            print(f"[DONE] {fname}: downloaded {downloaded:,} B")
    except Exception as e:
        print(f"[ERR] {fname}: {e}")
        return False

    # Verify md5
    print(f"[VERIFY] computing md5 of {tmp} ...")
    actual_md5 = md5_file(tmp)
    if actual_md5 == expected_md5:
        tmp.replace(out)
        print(f"[OK] {fname} md5 verified, renamed to final")
        return True
    else:
        print(f"[FAIL] {fname} md5 mismatch: {actual_md5} vs {expected_md5}")
        print("       .part file kept for inspection; will retry on next run")
        return False


def main():
    import urllib.parse  # noqa
    only_large = "--large-only" in sys.argv
    skip_large = "--skip-large" in sys.argv

    results = {}
    if not only_large:
        print("=" * 80)
        print("Phase 1: Download small attribute files")
        print("=" * 80)
        for fname, md5, size in FILES[:-1]:  # all except last large
            ok = download_small(fname, md5, size)
            results[fname] = ok

    if not skip_large:
        print()
        print("=" * 80)
        print("Phase 2: Download large time series file (3.4 GB) with resume")
        print("=" * 80)
        fname, md5, size = FILES[-1]
        ok = download_large_resumable(fname, md5, size)
        results[fname] = ok

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for fname, ok in results.items():
        print(f"  {'[OK] ' if ok else '[FAIL]'} {fname}")
    n_fail = sum(1 for ok in results.values() if not ok)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
