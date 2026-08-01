"""Run parameter sensitivity analysis.

For each parameter, sweep its value range while keeping others at default.
Compute the Elasticity coefficient to quantify sensitivity:
  E = (ΔMetric / Metric_base) / (ΔParam / Param_base)
  - |E| > 0.5: high sensitivity
  - 0.2 <= |E| <= 0.5: medium
  - |E| < 0.2: low

Parameters swept (from config.py EXP_CFG.sensitivity_params):
  - lambda_mass:      [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
  - uh_kernel_size:   [10, 20, 30, 60, 90, 120]
  - hidden_size:      [32, 64, 128, 256]
  - lambda_extreme:   [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
  - seq_length:       [60, 90, 180, 365, 540]

Each value trained with 1 seed (42) for tractability.
Results saved to results/experiment3_sensitivity/.
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
from config import MODEL_CFG, DATA_CFG, RESULTS_DIR
from train import train_one_model


# Default values (baseline)
DEFAULTS = {
    "lambda_mass": MODEL_CFG.lambda_mass,        # 0.01
    "uh_kernel_size": MODEL_CFG.uh_kernel_size,  # 60
    "hidden_size": MODEL_CFG.hidden_size,        # 128
    "lambda_extreme": MODEL_CFG.lambda_extreme,  # 0.5
    "seq_length": DATA_CFG.seq_length,           # 180
}

SWEEPS = {
    "lambda_mass":     [0.0, 0.01, 0.1, 1.0],
    "uh_kernel_size":  [20, 60, 120],
    "hidden_size":     [64, 128, 256],
    "lambda_extreme":  [0.0, 0.5, 2.0],
    "seq_length":      [90, 180, 365],
}


def compute_elasticity(param_values, metric_values, base_param, base_metric):
    """Compute elasticity coefficient for each param value relative to baseline.

    E = (Δmetric/metric) / (Δparam/param)
    Returns list of (param_value, metric_value, elasticity, sensitivity_level).
    """
    results = []
    for pv, mv in zip(param_values, metric_values):
        if pv == base_param or base_param == 0 or base_metric == 0:
            results.append((pv, mv, 0.0, "baseline"))
            continue
        d_param = (pv - base_param) / base_param
        d_metric = (mv - base_metric) / (abs(base_metric) + 1e-8)
        if abs(d_param) < 1e-8:
            e = 0.0
        else:
            e = d_metric / d_param
        level = "high" if abs(e) > 0.5 else ("medium" if abs(e) > 0.2 else "low")
        results.append((pv, mv, e, level))
    return results


def run_sensitivity(
    params: List[str],
    seed: int,
    n_basins: int,
    epochs: int,
    batch_size: int,
    seq_length: int = 180,
    device: str = "cuda",
) -> Dict:
    """Sweep each parameter and compute elasticity."""
    save_dir = RESULTS_DIR / "experiment3_sensitivity"
    save_dir.mkdir(parents=True, exist_ok=True)
    DEFAULTS["seq_length"] = seq_length

    all_results = {}
    total_runs = sum(len(SWEEPS[p]) for p in params)
    i = 0
    start_time = time.time()

    for param_name in params:
        values = SWEEPS[param_name]
        base_val = DEFAULTS[param_name]
        print()
        print("=" * 100)
        print(f"Sensitivity sweep: {param_name} (base={base_val}, values={values})")
        print("=" * 100)

        per_param = []
        for val in values:
            i += 1
            print(f"\n[{i}/{total_runs}] {param_name}={val}")
            # Build kwargs for train_one_model
            kwargs = dict(
                model_name="PINN_UHConv",
                seed=seed,
                epochs=epochs,
                n_basins=n_basins,
                batch_size=batch_size,
                device=device,
                cache_data=True,
                save_dir=save_dir,
                quiet=False,
                run_name=f"{param_name}_{val}_seed{seed}",
            )
            # Set this param and keep others at default
            kwargs["lambda_mass"] = DEFAULTS["lambda_mass"]
            kwargs["lambda_extreme"] = DEFAULTS["lambda_extreme"]
            kwargs["uh_kernel_size"] = DEFAULTS["uh_kernel_size"]
            kwargs["hidden_size"] = DEFAULTS["hidden_size"]
            kwargs["seq_length"] = DEFAULTS["seq_length"]
            # Override the swept param
            kwargs[param_name] = val

            try:
                result = train_one_model(**kwargs)
                tm = result["test_metrics"]
                per_param.append({
                    "param_value": val,
                    "NSE_median": tm.get("NSE_median", float("nan")),
                    "NSE_extreme_median": tm.get("NSE_extreme_median", float("nan")),
                    "KGE_median": tm.get("KGE_median", float("nan")),
                    "RMSE_median": tm.get("RMSE_median", float("nan")),
                    "PBIAS_median": tm.get("PBIAS_median", float("nan")),
                    "FHV_median": tm.get("FHV_median", float("nan")),
                    "best_epoch": result["best_epoch"],
                    "train_time_sec": result["train_time_sec"],
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                per_param.append({"param_value": val, "error": str(e)})

        # Compute elasticity on NSE
        valid = [p for p in per_param if "NSE_median" in p and not np.isnan(p["NSE_median"])]
        max_e = 0.0
        max_level = "low"
        if valid:
            base_entry = next((p for p in valid if p["param_value"] == base_val), valid[0])
            base_metric = base_entry["NSE_median"]
            elas = compute_elasticity(
                [p["param_value"] for p in valid],
                [p["NSE_median"] for p in valid],
                base_val, base_metric,
            )
            for p, (pv, mv, e, lvl) in zip(valid, elas):
                p["elasticity"] = e
                p["sensitivity_level"] = lvl
            max_e = max((abs(s.get("elasticity", 0)) for s in valid), default=0.0)
            max_level = "high" if max_e > 0.5 else ("medium" if max_e > 0.2 else "low")

        all_results[param_name] = {
            "base_value": base_val,
            "base_NSE_median": base_metric if valid else float("nan"),
            "max_abs_elasticity": float(max_e),
            "sensitivity_level": max_level,
            "sweep": per_param,
        }

    elapsed = time.time() - start_time
    print()
    print("=" * 100)
    print(f"All sensitivity sweeps finished in {elapsed/60:.1f} min")
    print("=" * 100)

    summary = {
        "per_param": all_results,
        "config": {
            "params": params, "seed": seed, "n_basins": n_basins,
            "epochs": epochs, "batch_size": batch_size,
            "defaults": DEFAULTS,
            "total_time_sec": elapsed,
        },
    }

    out_file = RESULTS_DIR / "experiment3_sensitivity_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {out_file}")

    # Print sensitivity table
    print()
    print("## Parameter Sensitivity Summary (elasticity on NSE_median)")
    print()
    print(f"| Parameter | Base | Best | BestNSE | MaxE | Level |")
    print(f"|-----------|------|------|---------|------|-------|")
    for param_name, pr in all_results.items():
        sweep = pr["sweep"]
        valid = [s for s in sweep if "NSE_median" in s and not np.isnan(s.get("NSE_median", float("nan")))]
        if not valid:
            print(f"| {param_name} | {pr['base_value']} | N/A | N/A | N/A | N/A |")
            continue
        best = max(valid, key=lambda x: x["NSE_median"])
        max_e = max((abs(s.get("elasticity", 0)) for s in valid), default=0)
        max_level = next((s.get("sensitivity_level", "low") for s in valid if abs(s.get("elasticity", 0)) == max_e), "low")
        print(f"| {param_name} | {pr['base_value']} | {best['param_value']} | {best['NSE_median']:.4f} | {max_e:.3f} | {max_level} |")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run parameter sensitivity analysis")
    parser.add_argument("--params", nargs="+", default=list(SWEEPS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_basins", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_length", type=int, default=180)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    run_sensitivity(
        params=args.params,
        seed=args.seed,
        n_basins=args.n_basins,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
