"""Quick validation that the losses.py scale fix works.

Runs PINN_UHConv and LSTM for 5 epochs on 10 basins and prints NSE/PBIAS.
Compares against the buggy 2-epoch baseline (NSE=0.09, PBIAS=93.7%).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import train_one_model
from config import RESULTS_DIR


def main():
    save_dir = RESULTS_DIR / "smoke_test_fix"
    save_dir.mkdir(parents=True, exist_ok=True)

    models = ["LSTM", "PINN_UHConv"]
    n_basins = 10
    epochs = 5
    seed = 42

    print("=" * 80)
    print(f"SMOKE TEST (losses.py fix validation): {models}, {epochs} epochs, {n_basins} basins, seed={seed}")
    print("=" * 80)

    results = {}
    for model_name in models:
        t0 = time.time()
        result = train_one_model(
            model_name=model_name,
            seed=seed,
            epochs=epochs,
            n_basins=n_basins,
            batch_size=256,
            device="cuda",
            cache_data=True,
            save_dir=save_dir,
            quiet=False,
        )
        elapsed = time.time() - t0
        tm = result["test_metrics"]
        vm = result["val_metrics"]
        print()
        print(f"--- {model_name} ({elapsed:.0f}s) ---")
        print(f"  best_epoch={result['best_epoch']}  best_val_NSE_med={result['best_val_nse_median']:.4f}")
        print(f"  test NSE_med={vm.get('NSE_median', float('nan')):.4f} -> {tm.get('NSE_median', float('nan')):.4f}")
        print(f"  test NSE_ext_med={tm.get('NSE_extreme_median', float('nan')):.4f}")
        print(f"  test KGE_med={tm.get('KGE_median', float('nan')):.4f}")
        print(f"  test PBIAS_med={tm.get('PBIAS_median', float('nan')):.2f}%  (target <30%)")
        print(f"  test Beta_NSE_med={tm.get('Beta_NSE_median', float('nan')):.4f}  (target ~1.0)")
        print(f"  test Alpha_NSE_med={tm.get('Alpha_NSE_median', float('nan')):.4f}")
        print(f"  test FHV_med={tm.get('FHV_median', float('nan')):.2f}")
        print(f"  test RMSE_med={tm.get('RMSE_median', float('nan')):.4f}")
        # Print loss history
        print(f"  loss history:")
        for h in result["history"]:
            print(f"    epoch {h['epoch']}: loss={h['train_loss']:.4f} nse={h['train_nse_loss']:.4f} mass={h['train_mass_loss']:.4f} ext={h['train_extreme_loss']:.4f} | val_NSE_med={h['val_nse_median']:.4f}")
        results[model_name] = {
            "test_NSE_median": tm.get("NSE_median"),
            "test_NSE_extreme_median": tm.get("NSE_extreme_median"),
            "test_KGE_median": tm.get("KGE_median"),
            "test_PBIAS_median": tm.get("PBIAS_median"),
            "test_Beta_NSE_median": tm.get("Beta_NSE_median"),
            "test_Alpha_NSE_median": tm.get("Alpha_NSE_median"),
            "test_FHV_median": tm.get("FHV_median"),
            "best_val_NSE_median": result["best_val_nse_median"],
            "best_epoch": result["best_epoch"],
        }

    # Summary
    print()
    print("=" * 80)
    print("SMOKE TEST SUMMARY (losses.py fix validation)")
    print("=" * 80)
    print(f"{'Model':<15} {'NSE_med':>10} {'NSE_ext':>10} {'KGE_med':>10} {'PBIAS%':>10} {'Beta':>8} {'Alpha':>8}")
    for m, r in results.items():
        print(f"{m:<15} {r['test_NSE_median']:>10.4f} {r['test_NSE_extreme_median']:>10.4f} {r['test_KGE_median']:>10.4f} {r['test_PBIAS_median']:>10.2f} {r['test_Beta_NSE_median']:>8.4f} {r['test_Alpha_NSE_median']:>8.4f}")
    print()
    print("Buggy baseline (2 epochs): PINN_UHConv NSE_med=0.0896 PBIAS=93.69% Beta=1.94")
    print("Fix target: NSE_med > 0.30, PBIAS < 30%, Beta ~ 1.0")

    # Save summary
    out = save_dir / "smoke_test_fix_summary.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
