"""Run improved PINN-UHConv experiments with lambda_mass=1.0 (from sensitivity analysis).

Sensitivity analysis showed lambda_mass=1.0 gives NSE=0.5116 (vs base 0.4686)
on seed=42. This script runs 5 seeds with the improved config to get a
statistically valid comparison against UH-LSTM.

Config: lambda_mass=1.0, epochs=15, all other params same as baseline.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, TRAIN_CFG
from train import train_one_model

SEEDS = [42, 2024, 7, 123, 999]
LAMBDA_MASS = 1.0
EPOCHS = 15
SAVE_DIR = RESULTS_DIR / "experiment5_improved"

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"Improved PINN-UHConv: lambda_mass={LAMBDA_MASS}, epochs={EPOCHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Save dir: {SAVE_DIR}")
    print("=" * 80)

    all_results = []
    start = time.time()

    for i, seed in enumerate(SEEDS):
        print(f"\n[{i+1}/{len(SEEDS)}] seed={seed}...")
        t0 = time.time()
        result = train_one_model(
            model_name="PINN_UHConv",
            seed=seed,
            epochs=EPOCHS,
            n_basins=100,
            lambda_mass=LAMBDA_MASS,
            lambda_extreme=0.5,
            use_mass_balance=True,
            use_extreme_weighting=True,
            use_uhconv=True,
            use_static_modulation=True,
            uh_kernel_size=60,
            hidden_size=128,
            seq_length=180,
            save_dir=SAVE_DIR,
            run_name=f"PINN_UHConv_improved_seed{seed}",
            quiet=False,
        )
        elapsed = time.time() - t0
        test = result["test_metrics"]
        print(f"  Done in {elapsed/60:.1f} min")
        print(f"  Test NSE_median={test['NSE_median']:.4f} KGE_median={test['KGE_median']:.4f}")
        all_results.append(result)

    total = time.time() - start
    print(f"\n{'=' * 80}")
    print(f"All {len(SEEDS)} seeds finished in {total/60:.1f} min")

    # Summary
    nse_vals = [r["test_metrics"]["NSE_median"] for r in all_results]
    kge_vals = [r["test_metrics"]["KGE_median"] for r in all_results]
    import numpy as np
    summary = {
        "config": {
            "model": "PINN_UHConv",
            "lambda_mass": LAMBDA_MASS,
            "lambda_extreme": 0.5,
            "epochs": EPOCHS,
            "seeds": SEEDS,
            "n_basins": 100,
            "seq_length": 180,
            "hidden_size": 128,
            "uh_kernel_size": 60,
        },
        "NSE_median": {
            "mean": float(np.mean(nse_vals)),
            "std": float(np.std(nse_vals, ddof=1)),
            "min": float(np.min(nse_vals)),
            "max": float(np.max(nse_vals)),
            "per_seed": [float(v) for v in nse_vals],
        },
        "KGE_median": {
            "mean": float(np.mean(kge_vals)),
            "std": float(np.std(kge_vals, ddof=1)),
            "min": float(np.min(kge_vals)),
            "max": float(np.max(kge_vals)),
            "per_seed": [float(v) for v in kge_vals],
        },
        "total_time_sec": total,
    }

    # Compare with UH-LSTM (from experiment1)
    uh_lstm_nse_mean = 0.5187848143400267  # from experiment1_baseline_comparison_results.json
    summary["comparison"] = {
        "UH_LSTM_NSE_median_mean": uh_lstm_nse_mean,
        "PINN_UHConv_improved_NSE_median_mean": summary["NSE_median"]["mean"],
        "difference": summary["NSE_median"]["mean"] - uh_lstm_nse_mean,
        "PINN_beats_UH_LSTM": summary["NSE_median"]["mean"] > uh_lstm_nse_mean,
    }

    summary_file = SAVE_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    print(f"\nNSE_median: mean={summary['NSE_median']['mean']:.4f} std={summary['NSE_median']['std']:.4f}")
    print(f"UH_LSTM NSE_median mean: {uh_lstm_nse_mean:.4f}")
    print(f"Difference: {summary['comparison']['difference']:+.4f}")
    print(f"PINN-UHConv beats UH-LSTM: {summary['comparison']['PINN_beats_UH_LSTM']}")

if __name__ == "__main__":
    main()
