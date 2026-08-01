"""Run ablation experiments: train PINN_UHConv with each core component removed.

Ablation configs (from config.py EXP_CFG.ablation_components):
  - full:                  complete PINN_UHConv
  - no_uhconv:             remove UHconv routing (Q_surface = eff_rain at same step)
  - no_mass_balance:       remove mass-balance loss
  - no_static_modulation:  remove static attribute FiLM gate
  - no_extreme_weighting:  remove extreme-event loss weighting

Each config is trained with 3 random seeds (42, 2024, 7) for statistical analysis.
Results saved to results/experiment2_ablation/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EXP_CFG, TRAIN_CFG, RESULTS_DIR, set_seed
from train import train_one_model


# Ablation config -> (use_uhconv, use_mass_balance, use_static_modulation, use_extreme_weighting)
ABLATION_CONFIGS = {
    "full":                  (True,  True,  True,  True),
    "no_uhconv":             (False, True,  True,  True),
    "no_mass_balance":       (True,  False, True,  True),
    "no_static_modulation":  (True,  True,  False, True),
    "no_extreme_weighting":  (True,  True,  True,  False),
}


def run_ablation(
    configs: List[str],
    seeds: List[int],
    n_basins: int,
    epochs: int,
    batch_size: int,
    seq_length: int,
    device: str = "cuda",
) -> Dict:
    """Train each ablation config across seeds and aggregate."""
    all_results = []
    total = len(configs) * len(seeds)
    i = 0
    start_time = time.time()
    save_dir = RESULTS_DIR / "experiment2_ablation"
    save_dir.mkdir(parents=True, exist_ok=True)

    for cfg_name in configs:
        use_uh, use_mass, use_static, use_ext = ABLATION_CONFIGS[cfg_name]
        for seed in seeds:
            i += 1
            print()
            print("=" * 100)
            print(f"[{i}/{total}] Ablation: {cfg_name} | seed={seed}")
            print(f"  use_uhconv={use_uh} use_mass_balance={use_mass} use_static_mod={use_static} use_extreme={use_ext}")
            print("=" * 100)
            try:
                result = train_one_model(
                    model_name="PINN_UHConv",
                    seed=seed,
                    epochs=epochs,
                    n_basins=n_basins,
                    batch_size=batch_size,
                    device=device,
                    cache_data=True,
                    save_dir=save_dir,
                    seq_length=seq_length,
                    use_uhconv=use_uh,
                    use_mass_balance=use_mass,
                    use_static_modulation=use_static,
                    use_extreme_weighting=use_ext,
                    quiet=False,
                    run_name=f"{cfg_name}_seed{seed}",
                )
                result["ablation_config"] = cfg_name
                all_results.append(result)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] ablation {cfg_name} seed={seed} failed: {e}")
                all_results.append({
                    "ablation_config": cfg_name, "seed": seed, "error": str(e),
                    "test_metrics": {},
                })

    elapsed = time.time() - start_time
    print()
    print("=" * 100)
    print(f"All ablation training finished in {elapsed/60:.1f} min")
    print("=" * 100)

    # Aggregate per config
    by_cfg = defaultdict(list)
    for r in all_results:
        by_cfg[r["ablation_config"]].append(r)

    agg = {}
    metrics_to_agg = [
        "NSE_median", "NSE_extreme_median", "KGE_median", "Pearson_r_median",
        "Alpha_NSE_median", "Beta_NSE_median", "RMSE_median",
        "FHV_median", "FLV_median", "Peak_Error_median", "PBIAS_median",
    ]
    for cfg, runs in by_cfg.items():
        m_agg = {"n_runs": len(runs), "n_params": runs[0].get("n_params", 0) if runs else 0}
        for m in metrics_to_agg:
            vals = [r["test_metrics"].get(m, float("nan")) for r in runs if "test_metrics" in r and r["test_metrics"]]
            vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if vals:
                m_agg[f"{m}_mean"] = float(np.mean(vals))
                m_agg[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                m_agg[f"{m}_min"] = float(np.min(vals))
                m_agg[f"{m}_max"] = float(np.max(vals))
            else:
                m_agg[f"{m}_mean"] = float("nan")
                m_agg[f"{m}_std"] = float("nan")
        agg[cfg] = m_agg

    summary = {
        "per_config": agg,
        "config": {
            "configs": configs, "seeds": seeds, "n_basins": n_basins,
            "epochs": epochs, "batch_size": batch_size, "seq_length": seq_length,
            "total_time_sec": elapsed,
        },
        "per_run_results": [
            {k: v for k, v in r.items() if k not in ("history",)}
            for r in all_results
        ],
    }

    out_file = RESULTS_DIR / "experiment2_ablation_results.json"
    with open(out_file, "w") as f:
        def _default(o):
            if isinstance(o, (np.floating, np.integer)):
                return float(o) if isinstance(o, np.floating) else int(o)
            return str(o)
        json.dump(summary, f, indent=2, default=_default)
    print(f"\nSummary saved to {out_file}")

    # Print table
    print()
    print("## Ablation Summary (mean ± std across seeds)")
    print()
    print(f"| Config | NSE | NSE_extreme | KGE | PBIAS | FHV | n_params |")
    print(f"|--------|-----|-------------|-----|-------|-----|----------|")
    for cfg, m in agg.items():
        nse = f"{m.get('NSE_median_mean', float('nan')):.4f}±{m.get('NSE_median_std', 0):.4f}"
        nse_ext = f"{m.get('NSE_extreme_median_mean', float('nan')):.4f}±{m.get('NSE_extreme_median_std', 0):.4f}"
        kge = f"{m.get('KGE_median_mean', float('nan')):.4f}±{m.get('KGE_median_std', 0):.4f}"
        pbias = f"{m.get('PBIAS_median_mean', float('nan')):.2f}"
        fhv = f"{m.get('FHV_median_mean', float('nan')):.2f}"
        npar = m.get("n_params", 0)
        print(f"| {cfg} | {nse} | {nse_ext} | {kge} | {pbias} | {fhv} | {npar:,} |")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 7])
    parser.add_argument("--n_basins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_ablation(
        configs=args.configs,
        seeds=args.seeds,
        n_basins=args.n_basins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
