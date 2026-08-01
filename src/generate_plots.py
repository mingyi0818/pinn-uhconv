"""Generate all paper figures (>=300 dpi PNG).

Figures:
  Figure 1: Algorithm architecture diagram (PINN_UHConv model structure)
  Figure 2: Baseline comparison (grouped bar chart: NSE / NSE_extreme / KGE)
  Figure 3: Ablation results (bar chart of 5 configs)
  Figure 4: Parameter sensitivity (line plots, 5 subplots)
  Figure 5: Robustness analysis (noise / missing / unseen-basin)

Reads from results/experiment1..4. Saves to plots/ directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, PLOTS_DIR, EXP_CFG

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colour palette
COLOR_OUR = "#d62728"       # red for our method
COLOR_BASELINE = "#1f77b4"  # blue for baselines
COLOR_ABLATION = "#2ca02c"  # green for ablation full
COLOR_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# Figure 1: Architecture diagram
# =============================================================================
def plot_architecture() -> Path:
    """Draw the PINN_UHConv architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Architecture of PINN-UHConv: Physics-Informed Neural Network with "
                 "Differentiable Unit Hydrograph Convolution", fontsize=12, pad=10)

    def box(x, y, w, h, text, color="#4a90d9", fontsize=9, textcolor="white"):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, color=textcolor, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # Input layer
    box(0.2, 4.5, 1.5, 1.0, "Forcing\n(P, T, DayL)\n[T×7]", "#5b9bd5")
    box(0.2, 2.8, 1.5, 1.0, "Static\nAttributes\n[59]", "#5b9bd5")

    # Static encoder
    box(2.2, 2.8, 1.5, 1.0, "Static\nEncoder\n(MLP)", "#70ad47")

    # LSTM encoder
    box(2.2, 4.5, 1.5, 1.0, "LSTM\nEncoder\n(h=128)", "#70ad47")

    # FiLM modulation
    box(4.2, 3.5, 1.5, 1.2, "FiLM\nStatic\nGate\n(γ, β)", "#ffc000", textcolor="black")

    # Rain head
    box(6.2, 4.5, 1.5, 1.0, "Rain Head\n(softplus)\nR_eff", "#ed7d31")

    # Baseflow head
    box(6.2, 2.5, 1.5, 1.0, "Baseflow\nHead\n(softplus)\nQ_b", "#ed7d31")

    # UHconv
    box(8.2, 4.0, 1.8, 1.5, "UHconv\n(Gamma kernel\ncausal conv)\nQ_surface", "#c00000")

    # Sum
    box(10.5, 3.2, 1.2, 1.2, "Q_total\n= Q_s\n+ Q_b", "#7030a0")

    # Losses (bottom)
    box(3.0, 0.5, 2.0, 0.8, "NSE Loss\n(MSE norm)", "#a5a5a5", textcolor="black")
    box(5.5, 0.5, 2.0, 0.8, "Mass Balance\nLoss (relative)", "#a5a5a5", textcolor="black")
    box(8.0, 0.5, 2.0, 0.8, "Extreme-Event\nLoss (weighted)", "#a5a5a5", textcolor="black")

    # Arrows (data flow)
    arrow(1.7, 5.0, 2.2, 5.0)    # forcing -> LSTM
    arrow(1.7, 3.3, 2.2, 3.3)    # static -> encoder
    arrow(3.7, 3.3, 4.2, 4.0)    # static_enc -> FiLM
    arrow(3.7, 5.0, 4.2, 4.5)    # LSTM -> FiLM (top)
    arrow(5.7, 4.5, 6.2, 4.9)    # FiLM -> rain_head
    arrow(5.7, 4.0, 6.2, 3.0)    # FiLM -> baseflow_head
    arrow(7.7, 4.9, 8.2, 4.9)    # rain_head -> UHconv
    arrow(7.7, 3.0, 10.5, 3.5)   # baseflow -> sum
    arrow(10.0, 4.5, 10.5, 4.2)  # UHconv -> sum

    # Loss arrows (dashed)
    for (x1, y1, x2, y2) in [(10.5, 3.2, 4.0, 1.3), (10.5, 3.2, 6.5, 1.3), (10.5, 3.2, 9.0, 1.3)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="gray", ls="--"))

    ax.text(6.0, 0.2, "L = L_NSE + λ_mass · L_mass + λ_ext · L_extreme",
            ha="center", fontsize=10, style="italic", color="gray")

    out = PLOTS_DIR / "figure1_architecture.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


# =============================================================================
# Figure 2: Baseline comparison
# =============================================================================
def plot_baseline_comparison() -> Optional[Path]:
    """Grouped bar chart: NSE / NSE_extreme / KGE for all models."""
    data = load_json(RESULTS_DIR / "experiment1_baseline_comparison_results.json")
    if not data:
        print("  [SKIP] experiment1 results not found")
        return None

    per_model = data.get("per_model", {})
    models = list(EXP_CFG.baselines) + [EXP_CFG.our_method]
    metrics = ["NSE_median", "NSE_extreme_median", "KGE_median"]
    metric_labels = ["NSE", "NSE_extreme", "KGE"]

    # Extract mean ± std
    means = {m: [] for m in metrics}
    stds = {m: [] for m in metrics}
    for model in models:
        pm = per_model.get(model, {})
        for m in metrics:
            means[m].append(pm.get(f"{m}_mean", 0))
            stds[m].append(pm.get(f"{m}_std", 0))

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means[m], width, yerr=stds[m], label=label,
                      color=COLOR_PALETTE[i], capsize=3, edgecolor="black", linewidth=0.5)
        # Highlight our method
        if i == 0:
            bars[-1].set_color(COLOR_OUR)
            bars[-1].set_edgecolor("black")

    ax.set_ylabel("Score")
    ax.set_title("Baseline Comparison: NSE / NSE_extreme / KGE (mean ± std, 5 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(loc="upper left")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim(bottom=min(0, min(means["NSE_median"]) - 0.1))

    # Annotate best
    for i, model in enumerate(models):
        if model == EXP_CFG.our_method:
            ax.annotate("Ours", xy=(i + offset, means["NSE_median"][i] + stds["NSE_median"][i]),
                        ha="center", va="bottom", fontsize=8, color=COLOR_OUR, fontweight="bold")

    out = PLOTS_DIR / "figure2_baseline_comparison.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


# =============================================================================
# Figure 3: Ablation results
# =============================================================================
def plot_ablation() -> Optional[Path]:
    """Bar chart of ablation configs."""
    data = load_json(RESULTS_DIR / "experiment2_ablation_results.json")
    if not data:
        print("  [SKIP] experiment2 results not found")
        return None

    per_config = data.get("per_config", {})
    configs = list(EXP_CFG.ablation_components)
    config_labels = ["Full", "w/o UHconv", "w/o Mass-Bal", "w/o Static", "w/o Extreme"]
    metrics = ["NSE_median", "NSE_extreme_median", "KGE_median"]
    metric_labels = ["NSE", "NSE_extreme", "KGE"]

    means = {m: [] for m in metrics}
    stds = {m: [] for m in metrics}
    for cfg in configs:
        pc = per_config.get(cfg, {})
        for m in metrics:
            means[m].append(pc.get(f"{m}_mean", 0))
            stds[m].append(pc.get(f"{m}_std", 0))

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (m, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means[m], width, yerr=stds[m], label=label,
                      color=COLOR_PALETTE[i], capsize=3, edgecolor="black", linewidth=0.5)
        if i == 0:
            bars[0].set_color(COLOR_ABLATION)
            bars[0].set_hatch("//")
            bars[0].set_edgecolor("black")

    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Component Contributions (mean ± std, 3 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=15, ha="right")
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotate full as baseline
    ax.annotate("Full model", xy=(0, means["NSE_median"][0] + stds["NSE_median"][0] + 0.02),
                ha="center", fontsize=8, color=COLOR_ABLATION, fontweight="bold")

    out = PLOTS_DIR / "figure3_ablation.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


# =============================================================================
# Figure 4: Parameter sensitivity
# =============================================================================
def plot_sensitivity() -> Optional[Path]:
    """Line plots for each parameter sweep."""
    data = load_json(RESULTS_DIR / "experiment3_sensitivity_results.json")
    if not data:
        print("  [SKIP] experiment3 results not found")
        return None

    per_param = data.get("per_param", {})
    params = list(per_param.keys())
    if not params:
        print("  [SKIP] no sensitivity params")
        return None

    n = len(params)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, param in enumerate(params):
        ax = axes[idx // ncols][idx % ncols]
        pr = per_param[param]
        sweep = pr.get("sweep", [])
        valid = [s for s in sweep if "NSE_median" in s and not np.isnan(s.get("NSE_median", float("nan")))]
        if not valid:
            ax.set_title(f"{param} (no data)")
            continue

        pvals = [s["param_value"] for s in valid]
        nses = [s["NSE_median"] for s in valid]
        base_val = pr.get("base_value")

        ax.plot(pvals, nses, "o-", color=COLOR_BASELINE, linewidth=2, markersize=6)
        # Mark baseline
        if base_val in pvals:
            bi = pvals.index(base_val)
            ax.plot(base_val, nses[bi], "s", color=COLOR_OUR, markersize=10, zorder=5, label="Default")
        # Mark best
        best_i = int(np.argmax(nses))
        ax.plot(pvals[best_i], nses[best_i], "*", color="gold", markersize=14, zorder=6, label="Best")

        ax.set_xlabel(param)
        ax.set_ylabel("NSE (median)")
        level = pr.get("sensitivity_level", "?")
        max_e = pr.get("max_abs_elasticity", 0)
        ax.set_title(f"{param}\n(max|E|={max_e:.2f}, {level})")
        ax.legend(fontsize=7)

    # Hide unused
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Parameter Sensitivity Analysis (Elasticity on NSE)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "figure4_sensitivity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


# =============================================================================
# Figure 5: Robustness analysis
# =============================================================================
def plot_robustness() -> Optional[Path]:
    """Robustness: noise / missing / unseen-basin."""
    data = load_json(RESULTS_DIR / "experiment4_robustness_results.json")
    if not data:
        print("  [SKIP] experiment4 results not found")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Noise
    ax = axes[0]
    noise = data.get("noise_robustness", {})
    if noise:
        rates, nses, kges = [], [], []
        for key in sorted(noise.keys()):
            r = float(key.replace("noise_", ""))
            rates.append(r)
            nses.append(noise[key].get("NSE_median", 0))
            kges.append(noise[key].get("KGE_median", 0))
        ax.plot(rates, nses, "o-", color=COLOR_OUR, linewidth=2, label="NSE")
        ax.plot(rates, kges, "s--", color=COLOR_BASELINE, linewidth=2, label="KGE")
    ax.set_xlabel("Noise std (fraction of normalised forcing)")
    ax.set_ylabel("Score")
    ax.set_title("(a) Noise Robustness")
    ax.legend()

    # Missing
    ax = axes[1]
    missing = data.get("missing_robustness", {})
    if missing:
        rates, nses, kges = [], [], []
        for key in sorted(missing.keys()):
            r = float(key.replace("missing_", ""))
            rates.append(r)
            nses.append(missing[key].get("NSE_median", 0))
            kges.append(missing[key].get("KGE_median", 0))
        ax.plot(rates, nses, "o-", color=COLOR_OUR, linewidth=2, label="NSE")
        ax.plot(rates, kges, "s--", color=COLOR_BASELINE, linewidth=2, label="KGE")
    ax.set_xlabel("Missing rate (fraction of time-steps zeroed)")
    ax.set_ylabel("Score")
    ax.set_title("(b) Missing-Data Robustness")
    ax.legend()

    # Unseen basin
    ax = axes[2]
    unseen = data.get("unseen_basin_robustness", {})
    if unseen:
        labels = list(unseen.keys())
        nses = [unseen[k].get("NSE_median", 0) for k in labels]
        kges = [unseen[k].get("KGE_median", 0) for k in labels]
        x = np.arange(len(labels))
        ax.bar(x - 0.15, nses, 0.3, color=COLOR_OUR, label="NSE", edgecolor="black", linewidth=0.5)
        ax.bar(x + 0.15, kges, 0.3, color=COLOR_BASELINE, label="KGE", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace("unseen_", "") for k in labels], rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("(c) Unseen-Basin Generalisation")
    ax.legend()

    fig.suptitle("Robustness Analysis of PINN-UHConv", fontsize=13, y=1.02)
    plt.tight_layout()
    out = PLOTS_DIR / "figure5_robustness.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


# =============================================================================
# Figure 6 (optional): Training curves
# =============================================================================
def plot_training_curves() -> Optional[Path]:
    """Training loss curves for PINN_UHConv vs LSTM (seed 42)."""
    exp1_dir = RESULTS_DIR / "experiment1"
    if not exp1_dir.exists():
        print("  [SKIP] experiment1 dir not found")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    found = False
    for model in ["LSTM", "PINN_UHConv"]:
        f = exp1_dir / f"train_{model}_seed42.json"
        if not f.exists():
            continue
        with open(f, "r") as fh:
            d = json.load(fh)
        hist = d.get("history", [])
        if not hist:
            continue
        epochs = [h.get("epoch", i + 1) for i, h in enumerate(hist)]
        train_loss = [h.get("train_loss", float("nan")) for h in hist]
        val_nse = [h.get("val_nse_median", float("nan")) for h in hist]
        color = COLOR_OUR if model == "PINN_UHConv" else COLOR_BASELINE
        ax.plot(epochs, train_loss, "o-", color=color, linewidth=2, label=f"{model} (train loss)")
        ax2 = ax.twinx()
        ax2.plot(epochs, val_nse, "s--", color=color, linewidth=1.5, alpha=0.6, label=f"{model} (val NSE)")
        found = True

    if not found:
        print("  [SKIP] no training history found")
        plt.close()
        return None

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax2.set_ylabel("Validation NSE (median)")
    ax.set_title("Training Curves: PINN-UHConv vs LSTM (seed=42)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    out = PLOTS_DIR / "figure6_training_curves.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  Saved {out}")
    return out


def main():
    print("=" * 80)
    print("Generating Figures")
    print("=" * 80)

    print("\n[Figure 1] Architecture diagram...")
    plot_architecture()

    print("\n[Figure 2] Baseline comparison...")
    plot_baseline_comparison()

    print("\n[Figure 3] Ablation results...")
    plot_ablation()

    print("\n[Figure 4] Parameter sensitivity...")
    plot_sensitivity()

    print("\n[Figure 5] Robustness analysis...")
    plot_robustness()

    print("\n[Figure 6] Training curves (optional)...")
    plot_training_curves()

    print(f"\nAll figures saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
