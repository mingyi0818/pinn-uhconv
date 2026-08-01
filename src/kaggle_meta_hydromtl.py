"""Quick metadata fetch for owenyy/hydromtl."""
import os, json
from kaggle.api.kaggle_api_extended import KaggleApi

if "KAGGLE_API_TOKEN" not in os.environ:
    print("Error: Set KAGGLE_API_TOKEN environment variable first.")
    print("Get your token from https://www.kaggle.com/settings")
    sys.exit(1)
api = KaggleApi()
api.authenticate()

# Try metadata view
print("Metadata for owenyy/hydromtl:")
print("=" * 80)
try:
    # Try datasets_view-like call
    md = api.dataset_view("owenyy/hydromtl")
    print(f"Type: {type(md)}")
    print(f"Attrs: {[a for a in dir(md) if not a.startswith('_')]}")
    print(f"Total bytes: {getattr(md, 'total_bytes', 'N/A')}")
    print(f"Title: {getattr(md, 'title', 'N/A')}")
    print(f"Description: {(getattr(md, 'description', '') or '')[:300]}")
    print(f"Last updated: {getattr(md, 'last_updated', 'N/A')}")
    print(f"Download count: {getattr(md, 'download_count', 'N/A')}")
except Exception as e:
    print(f"dataset_view ERR: {e}")
    # Fallback: search list which has total_bytes
    print("\nFallback: search results containing total_bytes:")
    results = api.dataset_list(search="hydromtl")
    for d in results[:5]:
        print(f"  ref={d.ref} | total_bytes={getattr(d,'total_bytes','?')} | title={d.title}")
