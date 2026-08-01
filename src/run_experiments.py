"""Run all comparison experiments: train all baselines + our method across 5 seeds.

This is the main entry point for Experiment 1 (baseline comparison).
Each model is trained with 5 random seeds; results are aggregated with
mean ± std and saved to results/experiment1_baseline_comparison_results.json.

Usage:
  python run_experiments.py --n_basins 100 --epochs 30 --seeds 42 2024 7 123 999
  python run_experiments.py --quick   # quick smoke test: 10 basins, 3 epochs, 1 seed
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXP_CFG, TRAIN_CFG, RESULTS_DIR, set_seed
from train import train_one_model


# -----------------------------------------------------------------------------
# Main experiment runner
# -----------------------------------------------------------------------------
def run_baseline_comparison(
    models: List[str],
    seeds: List[int],
    n_basins: int,
    epochs: int,
    batch_size: int,
    seq_length: int,
    device: str = "cuda",
    cache_data: bool = True,
    skip_existing: bool = True,
) -> Dict:
    """Train each (model, seed) combination and aggregate results.

    If skip_existing=True, a (model, seed) run whose result file already exists
    and contains no error is loaded from disk instead of being retrained. This
    allows resuming an interrupted experiment without re-running completed models.
    """
    all_results = []
    total = len(models) * len(seeds)
    i = 0
    start_time = time.time()
    save_dir = RESULTS_DIR / "experiment1"

    for model_name in models:
        for seed in seeds:
            i += 1
            print()
            print("=" * 100)
            print(f"[{i}/{total}] Training {model_name} | seed={seed}")
            print("=" * 100)

            # Skip already-completed runs (resume support)
            out_file = save_dir / f"train_{model_name}_seed{seed}.json"
            if skip_existing and out_file.exists():
                try:
                    with open(out_file) as f:
                        cached = json.load(f)
                    # Treat as complete only if it has test_metrics and no error
                    if cached.get("test_metrics") and not cached.get("error"):
                        n_params = cached.get("n_params", "?")
                        best_ep = cached.get("best_epoch", "?")
                        best_nse = cached.get("best_val_nse_median", "?")
                        print(f"[SKIP] {model_name} seed={seed} already trained "
                              f"(params={n_params}, best_epoch={best_ep}, val_NSE_med={best_nse}). "
                              f"Loading cached result.")
                        all_results.append(cached)
                        continue
                    else:
                        print(f"[REDO] {model_name} seed={seed}: cached result has error/no metrics, retraining.")
                except Exception as e:
                    print(f"[REDO] {model_name} seed={seed}: failed to load cache ({e}), retraining.")

            try:
                result = train_one_model(
                    model_name=model_name,
                    seed=seed,
                    epochs=epochs,
                    n_basins=n_basins,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    device=device,
                    cache_data=cache_data,
                    save_dir=save_dir,
                )
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] {model_name} seed={seed} failed: {e}")
                all_results.append({
                    "model_name": model_name, "seed": seed, "error": str(e),
                    "test_metrics": {},
                })

    elapsed = time.time() - start_time
    print()
    print("=" * 100)
    print(f"All training finished in {elapsed/60:.1f} min")
    print("=" * 100)

    # Aggregate per model
    summary = aggregate_results(all_results)
    summary["config"] = {
        "models": models, "seeds": seeds, "n_basins": n_basins,
        "epochs": epochs, "batch_size": batch_size, "seq_length": seq_length,
        "total_time_sec": elapsed,
    }
    summary["per_run_results"] = [
        {k: v for k, v in r.items() if k not in ("history",)}
        for r in all_results
    ]

    # Save
    out_file = RESULTS_DIR / "experiment1_baseline_comparison_results.json"
    with open(out_file, "w") as f:
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return str(o)
        json.dump(summary, f, indent=2, default=_default)
    print(f"\nSummary saved to {out_file}")

    # Print final table
    print_summary_table(summary)
    return summary


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate per-seed results into mean ± std per model."""
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        if "test_metrics" in r and r["test_metrics"]:
            by_model[r["model_name"]].append(r)

    agg = {}
    metrics_to_aggregate = [
        "NSE_median", "NSE_extreme_median", "KGE_median", "Pearson_r_median",
        "Alpha_NSE_median", "Beta_NSE_median", "RMSE_median",
        "FHV_median", "FLV_median", "Peak_Error_median",
        "NSE_mean", "NSE_extreme_mean",
    ]
    for model, runs in by_model.items():
        m_agg = {"n_runs": len(runs), "n_params": runs[0].get("n_params", 0) if runs else 0}
        for m in metrics_to_aggregate:
            vals = [r["test_metrics"].get(m, float("nan")) for r in runs if "test_metrics" in r]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                m_agg[f"{m}_mean"] = float(np.mean(vals))
                m_agg[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                m_agg[f"{m}_min"] = float(np.min(vals))
                m_agg[f"{m}_max"] = float(np.max(vals))
            else:
                m_agg[f"{m}_mean"] = float("nan")
                m_agg[f"{m}_std"] = float("nan")
        agg[model] = m_agg
    return {"per_model": agg}


def print_summary_table(summary: Dict) -> None:
    """Print a Markdown-style table of NSE / NSE_extreme / KGE per model."""
    print()
    print("## Baseline Comparison Summary (mean ± std across seeds)")
    print()
    print(f"| Model | NSE | NSE_extreme | KGE | RMSE | FHV | n_params |")
    print(f"|-------|-----|-------------|-----|------|-----|----------|")
    for model, m in summary["per_model"].items():
        nse = f"{m.get('NSE_median_mean', float('nan')):.4f}±{m.get('NSE_median_std', 0):.4f}"
        nse_ext = f"{m.get('NSE_extreme_median_mean', float('nan')):.4f}±{m.get('NSE_extreme_median_std', 0):.4f}"
        kge = f"{m.get('KGE_median_mean', float('nan')):.4f}±{m.get('KGE_median_std', 0):.4f}"
        rmse = f"{m.get('RMSE_median_mean', float('nan')):.4f}"
        fhv = f"{m.get('FHV_median_mean', float('nan')):.2f}"
        npar = m.get("n_params", 0)
        print(f"| {model} | {nse} | {nse_ext} | {kge} | {rmse} | {fhv} | {npar:,} |")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison experiment")
    parser.add_argument("--models", nargs="+", default=list(EXP_CFG.baselines) + [EXP_CFG.our_method])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(TRAIN_CFG.seeds))
    parser.add_argument("--n_basins", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--device", type=str, default=TRAIN_CFG.device)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_skip", action="store_true", help="Re-train all models even if cached results exist")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test: 10 basins, 3 epochs, 1 seed")
    args = parser.parse_args()

    if args.quick:
        args.n_basins = 10
        args.epochs = 3
        args.seeds = [42]
        args.models = ["LSTM", "PINN_UHConv"]   # subset for smoke test
        print("[QUICK] Quick mode: 10 basins, 3 epochs, 1 seed, 2 models")

    summary = run_baseline_comparison(
        models=args.models,
        seeds=args.seeds,
        n_basins=args.n_basins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
        cache_data=not args.no_cache,
        skip_existing=not args.no_skip,
    )
    return summary


if __name__ == "__main__":
    main()
