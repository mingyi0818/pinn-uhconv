"""Statistical analysis: paired t-test, 95% CI, effect size, elasticity verification.

Reads results from experiment1 (baseline comparison), experiment2 (ablation),
experiment3 (sensitivity), and experiment4 (robustness), then computes:
  - Paired t-test (PINN_UHConv vs each baseline, per-seed NSE)
  - 95% confidence intervals (t-distribution)
  - Cohen's d effect size
  - One-way ANOVA across ablation configs
  - Pearson/Spearman correlation for sensitivity
  - Elasticity verification

All statistical tests report: method, dof, statistic, p-value.
Results saved to results/statistics/statistical_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, EXP_CFG


STAT_DIR = RESULTS_DIR / "statistics"
STAT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict:
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def extract_per_seed_metric(per_run_results: List[Dict], model_name: str, metric: str) -> np.ndarray:
    """Extract a metric array for one model across seeds."""
    vals = []
    for r in per_run_results:
        if r.get("model_name") == model_name and "test_metrics" in r and r["test_metrics"]:
            v = r["test_metrics"].get(metric)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
    return np.array(vals)


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Dict:
    """Paired t-test: a vs b. Returns method, dof, t, p, mean_diff, ci."""
    n = min(len(a), len(b))
    if n < 2:
        return {"method": "paired_t_test", "error": "insufficient samples", "n": n}
    a, b = a[:n], b[:n]
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se = std_diff / np.sqrt(n)
    t_stat = float(mean_diff / se) if se > 0 else 0.0
    dof = n - 1
    p_val = float(2 * stats.t.sf(abs(t_stat), dof))
    # 95% CI for mean difference
    t_crit = float(stats.t.ppf(0.975, dof))
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se
    # Cohen's d (paired): mean_diff / std_diff
    cohens_d = float(mean_diff / std_diff) if std_diff > 0 else 0.0
    return {
        "method": "paired_t_test (two-sided)",
        "n": int(n),
        "dof": int(dof),
        "t_statistic": t_stat,
        "p_value": p_val,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": float(se),
        "ci_95": {"level": 0.95, "lower": float(ci_low), "upper": float(ci_high)},
        "cohens_d_paired": cohens_d,
        "effect_size_label": _effect_size_label(abs(cohens_d)),
        "significant_at_0.05": bool(p_val < 0.05),
    }


def _effect_size_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def confidence_interval(values: np.ndarray, level: float = 0.95) -> Dict:
    """t-distribution CI for the mean of values."""
    n = len(values)
    if n < 2:
        return {"level": level, "mean": float(np.mean(values)) if n else None,
                "lower": None, "upper": None, "n": n}
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = std / np.sqrt(n)
    dof = n - 1
    t_crit = float(stats.t.ppf(1 - (1 - level) / 2, dof))
    return {
        "level": level,
        "mean": mean,
        "std": std,
        "se": se,
        "dof": int(dof),
        "lower": float(mean - t_crit * se),
        "upper": float(mean + t_crit * se),
        "n": int(n),
    }


def one_way_anova(groups: Dict[str, np.ndarray]) -> Dict:
    """One-way ANOVA across named groups."""
    valid = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(valid) < 2:
        return {"method": "one_way_anova", "error": "need >=2 groups with n>=2", "n_groups": len(valid)}
    vals = list(valid.values())
    f_stat, p_val = stats.f_oneway(*vals)
    # df: between = k-1, within = N-k
    k = len(valid)
    N = sum(len(v) for v in valid.values())
    return {
        "method": "one_way_anova",
        "k_groups": int(k),
        "N_total": int(N),
        "df_between": int(k - 1),
        "df_within": int(N - k),
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "significant_at_0.05": bool(p_val < 0.05),
    }


def bonferroni_correct(p_values: List[float], alpha: float = 0.05) -> Dict:
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    corrected_alpha = alpha / m
    rejected = [p < corrected_alpha for p in p_values]
    return {
        "method": "bonferroni_correction",
        "n_comparisons": int(m),
        "corrected_alpha": float(corrected_alpha),
        "original_alpha": float(alpha),
        "rejected": rejected,
        "corrected_p_values": [float(min(p * m, 1.0)) for p in p_values],
    }


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> Dict:
    """Pearson and Spearman correlation."""
    out = {}
    if len(x) >= 3:
        r_p, p_p = stats.pearsonr(x, y)
        out["pearson"] = {"r": float(r_p), "p_value": float(p_p), "n": int(len(x))}
        r_s, p_s = stats.spearmanr(x, y)
        out["spearman"] = {"rho": float(r_s), "p_value": float(p_s), "n": int(len(x))}
    return out


def analyze_experiment1() -> Dict:
    """Baseline comparison: paired t-tests, CI, effect sizes."""
    data = load_json(RESULTS_DIR / "experiment1_baseline_comparison_results.json")
    if not data:
        return {"error": "experiment1 results not found"}
    per_run = data.get("per_run_results", [])
    our = EXP_CFG.our_method
    baselines = list(EXP_CFG.baselines)

    metrics = ["NSE_median", "NSE_extreme_median", "KGE_median", "PBIAS_median", "FHV_median"]
    result = {"experiment": "experiment1_baseline_comparison", "our_method": our}

    # Per-model CI
    ci_table = {}
    for model in baselines + [our]:
        ci_table[model] = {}
        for m in metrics:
            vals = extract_per_seed_metric(per_run, model, m)
            ci_table[model][m] = {
                "ci": confidence_interval(vals),
                "values": vals.tolist(),
            }
    result["confidence_intervals"] = ci_table

    # Paired t-test: our vs each baseline
    t_tests = {}
    p_values_for_bonferroni = []
    for m in metrics:
        t_tests[m] = {}
        our_vals = extract_per_seed_metric(per_run, our, m)
        for bl in baselines:
            bl_vals = extract_per_seed_metric(per_run, bl, m)
            tt = paired_t_test(our_vals, bl_vals)
            t_tests[m][bl] = tt
            if "p_value" in tt:
                p_values_for_bonferroni.append(tt["p_value"])
    result["paired_t_tests"] = t_tests

    # Bonferroni correction across all comparisons
    if p_values_for_bonferroni:
        result["bonferroni_correction"] = bonferroni_correct(p_values_for_bonferroni)

    return result


def analyze_experiment2() -> Dict:
    """Ablation: one-way ANOVA + paired t-test (full vs each ablated)."""
    data = load_json(RESULTS_DIR / "experiment2_ablation_results.json")
    if not data:
        return {"error": "experiment2 results not found"}
    per_run = data.get("per_run_results", [])
    configs = list(EXP_CFG.ablation_components)

    metrics = ["NSE_median", "NSE_extreme_median", "KGE_median", "PBIAS_median"]
    result = {"experiment": "experiment2_ablation", "configs": configs}

    # Extract per-config metric arrays
    per_config_vals = {}
    for m in metrics:
        per_config_vals[m] = {}
        for cfg in configs:
            vals = []
            for r in per_run:
                if r.get("ablation_config") == cfg and "test_metrics" in r and r["test_metrics"]:
                    v = r["test_metrics"].get(m)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(float(v))
            per_config_vals[m][cfg] = np.array(vals)

    # ANOVA across configs
    anova_table = {}
    for m in metrics:
        groups = per_config_vals[m]
        anova_table[m] = one_way_anova(groups)
        # CI per config
        anova_table[m]["ci_per_config"] = {
            cfg: confidence_interval(v) for cfg, v in groups.items()
        }
    result["anova"] = anova_table

    # Paired t-test: full vs each ablated
    t_tests = {}
    p_values_for_bonf = []
    for m in metrics:
        t_tests[m] = {}
        full_vals = per_config_vals[m].get("full", np.array([]))
        for cfg in configs:
            if cfg == "full":
                continue
            ablated_vals = per_config_vals[m].get(cfg, np.array([]))
            tt = paired_t_test(full_vals, ablated_vals)
            t_tests[m][cfg] = tt
            if "p_value" in tt:
                p_values_for_bonf.append(tt["p_value"])
    result["paired_t_tests_full_vs_ablated"] = t_tests
    if p_values_for_bonf:
        result["bonferroni_correction"] = bonferroni_correct(p_values_for_bonf)

    return result


def analyze_experiment3() -> Dict:
    """Sensitivity: elasticity verification + Pearson/Spearman correlation."""
    data = load_json(RESULTS_DIR / "experiment3_sensitivity_results.json")
    if not data:
        return {"error": "experiment3 results not found"}
    per_param = data.get("per_param", {})
    result = {"experiment": "experiment3_sensitivity", "per_param": {}}

    for param, pr in per_param.items():
        sweep = pr.get("sweep", [])
        valid = [s for s in sweep if "NSE_median" in s and not np.isnan(s.get("NSE_median", float("nan")))]
        if len(valid) < 2:
            result["per_param"][param] = {"error": "insufficient valid points"}
            continue

        param_vals = np.array([float(s["param_value"]) for s in valid])
        nse_vals = np.array([float(s["NSE_median"]) for s in valid])

        # Correlation between param value and NSE
        corr = pearson_spearman(param_vals, nse_vals)

        # Find best
        best_idx = int(np.argmax(nse_vals))
        best = valid[best_idx]

        # Max elasticity already computed in run_sensitivity
        max_e = max((abs(s.get("elasticity", 0)) for s in valid), default=0.0)
        max_level = "high" if max_e > 0.5 else ("medium" if max_e > 0.2 else "low")

        result["per_param"][param] = {
            "base_value": pr.get("base_value"),
            "base_NSE": pr.get("base_NSE_median"),
            "best_value": best["param_value"],
            "best_NSE": float(best["NSE_median"]),
            "max_abs_elasticity": float(max_e),
            "sensitivity_level": max_level,
            "correlation": corr,
            "sweep": valid,
        }

    return result


def analyze_experiment4() -> Dict:
    """Robustness: trend analysis + correlation (noise/missing rate vs NSE)."""
    data = load_json(RESULTS_DIR / "experiment4_robustness_results.json")
    if not data:
        return {"error": "experiment4 results not found"}
    result = {"experiment": "experiment4_robustness"}

    # Noise robustness
    noise = data.get("noise_robustness", {})
    if noise:
        rates = []
        nses = []
        for key, m in sorted(noise.items()):
            rate = float(key.replace("noise_", ""))
            rates.append(rate)
            nses.append(float(m.get("NSE_median", float("nan"))))
        result["noise"] = {
            "rates": rates,
            "NSE_median": nses,
            "correlation": pearson_spearman(np.array(rates), np.array(nses)),
            "degradation_at_max": float(nses[0] - nses[-1]) if nses else None,
        }

    # Missing data
    missing = data.get("missing_robustness", {})
    if missing:
        rates = []
        nses = []
        for key, m in sorted(missing.items()):
            rate = float(key.replace("missing_", ""))
            rates.append(rate)
            nses.append(float(m.get("NSE_median", float("nan"))))
        result["missing"] = {
            "rates": rates,
            "NSE_median": nses,
            "correlation": pearson_spearman(np.array(rates), np.array(nses)),
            "degradation_at_max": float(nses[0] - nses[-1]) if nses else None,
        }

    # Unseen basin
    unseen = data.get("unseen_basin_robustness", {})
    if unseen:
        nses = []
        for key, m in unseen.items():
            nses.append(float(m.get("NSE_median", float("nan"))))
        result["unseen_basin"] = {
            "NSE_values": nses,
            "mean_NSE": float(np.mean(nses)) if nses else None,
            "std_NSE": float(np.std(nses, ddof=1)) if len(nses) > 1 else 0.0,
            "ci": confidence_interval(np.array(nses)) if nses else None,
        }

    return result


def main():
    print("=" * 80)
    print("Statistical Analysis")
    print("=" * 80)

    all_results = {}

    print("\n[1/4] Analyzing experiment1 (baseline comparison)...")
    all_results["experiment1"] = analyze_experiment1()

    print("[2/4] Analyzing experiment2 (ablation)...")
    all_results["experiment2"] = analyze_experiment2()

    print("[3/4] Analyzing experiment3 (sensitivity)...")
    all_results["experiment3"] = analyze_experiment3()

    print("[4/4] Analyzing experiment4 (robustness)...")
    all_results["experiment4"] = analyze_experiment4()

    out_file = STAT_DIR / "statistical_analysis.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nStatistical analysis saved to {out_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    exp1 = all_results.get("experiment1", {})
    if "paired_t_tests" in exp1:
        print("\n## Paired t-tests (PINN_UHConv vs baselines) on NSE_median:")
        tts = exp1["paired_t_tests"].get("NSE_median", {})
        for bl, tt in tts.items():
            if "p_value" in tt:
                sig = "***" if tt["p_value"] < 0.001 else ("**" if tt["p_value"] < 0.01 else ("*" if tt["p_value"] < 0.05 else "ns"))
                print(f"  vs {bl:20s}: t={tt['t_statistic']:.3f} p={tt['p_value']:.4f} d={tt['cohens_d_paired']:.3f} ({tt['effect_size_label']}) {sig}")

    exp2 = all_results.get("experiment2", {})
    if "anova" in exp2:
        print("\n## Ablation ANOVA on NSE_median:")
        an = exp2["anova"].get("NSE_median", {})
        if "f_statistic" in an:
            print(f"  F({an['df_between']},{an['df_within']})={an['f_statistic']:.3f} p={an['p_value']:.4f} sig={an['significant_at_0.05']}")

    exp3 = all_results.get("experiment3", {})
    if "per_param" in exp3:
        print("\n## Sensitivity (max |elasticity| per parameter):")
        for p, pr in exp3["per_param"].items():
            if "max_abs_elasticity" in pr:
                print(f"  {p:20s}: max|E|={pr['max_abs_elasticity']:.3f} ({pr['sensitivity_level']}) best={pr['best_value']} (NSE={pr['best_NSE']:.4f})")

    return all_results


if __name__ == "__main__":
    main()
