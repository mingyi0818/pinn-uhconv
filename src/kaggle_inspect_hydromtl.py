"""Get full file listing and total size of owenyy/hydromtl Kaggle dataset."""
import os
from kaggle.api.kaggle_api_extended import KaggleApi

if "KAGGLE_API_TOKEN" not in os.environ:
    print("Error: Set KAGGLE_API_TOKEN environment variable first.")
    print("Get your token from https://www.kaggle.com/settings")
    sys.exit(1)
api = KaggleApi()
api.authenticate()

# Use underlying kaggle client to do paginated file listing
all_files = []
page_token = None
while True:
    if page_token:
        resp = api.dataset_list_files("owenyy/hydromtl", page_token=page_token)
    else:
        resp = api.dataset_list_files("owenyy/hydromtl")
    files_page = resp.files if hasattr(resp, "files") else []
    all_files.extend(files_page)
    page_token = resp.next_page_token if hasattr(resp, "next_page_token") else None
    print(f"Fetched {len(files_page)} files (total so far: {len(all_files)})", flush=True)
    if not page_token:
        break
    if len(all_files) > 50000:
        print("Stopping at 50k files")
        break

print()
print(f"Total files: {len(all_files)}")

# Total size
total = 0
sizes_known = 0
for f in all_files:
    sz = getattr(f, "size", None)
    if isinstance(sz, int):
        total += sz
        sizes_known += 1
print(f"Files with size info: {sizes_known}/{len(all_files)}")
print(f"Total size: {total/1e9:.2f} GB" if total else "Total size: unknown")

# Group by top-level dir
from collections import Counter
top_dirs = Counter()
for f in all_files:
    parts = f.name.split("/")
    if len(parts) >= 4:
        # e.g. camels/camels_us/<category>/<...>
        key = "/".join(parts[:3])
    elif len(parts) >= 2:
        key = "/".join(parts[:2])
    else:
        key = parts[0]
    top_dirs[key] += 1

print()
print("Top-level directory breakdown:")
for k, v in sorted(top_dirs.items(), key=lambda x: -x[1]):
    print(f"  {k:<70s} : {v} files")

# Sample some attribute files (search for camels_*.txt at any level)
print()
print("Attribute-like files (camels_*.txt, *.xlsx, README, readme):")
attr_files = [f for f in all_files if "camels_" in f.name.lower() or "readme" in f.name.lower() or f.name.lower().endswith(".xlsx")]
for f in attr_files[:30]:
    print(f"  {f.name} | size={getattr(f,'size','?')}")
