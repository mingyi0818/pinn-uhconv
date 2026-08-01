"""Probe Zenodo CAMELS-US record file list and sizes via HTML scraping (API 403 fallback)."""
import urllib.request
import re
import sys

RECORD_ID = "15529996"
url = f"https://zenodo.org/record/{RECORD_ID}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
req = urllib.request.Request(url, headers=headers)
try:
    resp = urllib.request.urlopen(req, timeout=120)
    html = resp.read().decode("utf-8", errors="ignore")
except Exception as e:
    print(f"ERROR fetching Zenodo HTML: {e}", file=sys.stderr)
    sys.exit(1)

# Search for filename + size patterns in HTML
# Zenodo HTML usually contains data like: <a href="/records/15529996/files/FILENAME?download=1">
file_link_pattern = re.compile(r'/records/[\d]+/files/([^"?]+)(?:\?download=1)?"')
sizes_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(kB|MB|GB)')

links = set(file_link_pattern.findall(html))
print(f"Found {len(links)} file links in HTML:")
for i, fn in enumerate(sorted(links)):
    print(f"  {i+1}. {fn}")

# Also dump any size info nearby
print("\nSize mentions in HTML (first 30):")
sizes = sizes_pattern.findall(html)
for s, unit in sizes[:30]:
    print(f"  {s} {unit}")
