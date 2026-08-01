"""Search and inspect CAMELS datasets on Kaggle (API 2.x compatible)."""
import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

if "KAGGLE_API_TOKEN" not in os.environ:
    print("Error: Set KAGGLE_API_TOKEN environment variable first.")
    print("Get your token from https://www.kaggle.com/settings")
    sys.exit(1)
api = KaggleApi()
api.authenticate()

# Inspect owenyy/hydromtl dataset
print("=" * 80)
print("Inspecting owenyy/hydromtl dataset files...")
print("=" * 80)
try:
    resp = api.dataset_list_files("owenyy/hydromtl")
    # Inspect object attributes
    print(f"Type: {type(resp)}")
    print(f"Attrs: {[a for a in dir(resp) if not a.startswith('_')]}")
    # Try .files attribute
    if hasattr(resp, "files"):
        for f in resp.files:
            print(f"  {f.name:<60s} | size={getattr(f,'size','?')}")
    elif hasattr(resp, "dataset_files"):
        for f in resp.dataset_files:
            print(f"  {f.name:<60s} | size={getattr(f,'size','?')}")
    else:
        print(f"resp: {resp}")
except Exception as e:
    import traceback
    traceback.print_exc()

# List datasets matching "camels"
print()
print("=" * 80)
print("Searching Kaggle datasets for 'camels'...")
print("=" * 80)
try:
    results = api.dataset_list(search="camels")
    print(f"Type: {type(results)}")
    print(f"Count: {len(results)}")
    # Inspect first item attrs
    if len(results) > 0:
        d0 = results[0]
        print(f"First item attrs: {[a for a in dir(d0) if not a.startswith('_')]}")
    for i, d in enumerate(results[:15]):
        # try common attrs
        ref = getattr(d, "ref", None) or getattr(d, "id", None) or getattr(d, "slug", None)
        title = getattr(d, "title", "?")
        size = getattr(d, "total_bytes", "?") or getattr(d, "size", "?")
        print(f"  {i+1:2d}. ref={ref} | size={size} | {title}")
except Exception as e:
    import traceback
    traceback.print_exc()
