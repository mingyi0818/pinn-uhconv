"""Quick test of the full data loading pipeline with 5 basins."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_full_pipeline, split_basins, CAMELSDataset, FORCING_USE

print("=" * 80)
print("Testing full data loading pipeline (n_basins=5)")
print("=" * 80)

basin_data, attrs, stats, bids = load_full_pipeline(n_basins=5, cache=False)

print()
print(f"Loaded {len(basin_data)} basins")
print(f"Attrs shape: {attrs.shape}")
print(f"Forcing variables used: {FORCING_USE}")
print(f"Basin IDs: {bids}")

print()
print("Per-basin summary:")
for bid, df in basin_data.items():
    q = df["Q_mm_day"]
    print(f"  {bid}: {len(df)} days | date {df.index.min().date()} to {df.index.max().date()} | "
          f"Q mean={q.mean():.3f} max={q.max():.3f} std={q.std():.3f} mm/day")

# Test split + dataset
print()
print("Testing split + dataset...")
train_b, val_b, test_b = split_basins(bids, seed=42)
print(f"  Split: {len(train_b)} train / {len(val_b)} val / {len(test_b)} test")

ds = CAMELSDataset(train_b, "train", basin_data, stats, attrs, seq_length=365)
print(f"  Train dataset: {len(ds)} sequences")
if len(ds) > 0:
    sample = ds[0]
    print(f"  Sample forcing shape: {sample['forcing'].shape}")
    print(f"  Sample target_norm: {sample['target_norm']:.4f}")
    print(f"  Sample target_raw (mm/day): {sample['target_raw']:.4f}")
    print(f"  Sample static shape: {sample['static'].shape}")
    print(f"  Sample basin_id: {sample['basin_id']}")

print()
print("[OK] Pipeline works end-to-end")
