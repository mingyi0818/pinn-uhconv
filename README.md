# PINN-UHConv: Physics-Informed Neural Network with Differentiable Unit-Hydrograph Convolution for Rainfall–Runoff Modeling

A physics-informed deep learning model that embeds a **differentiable unit-hydrograph convolution (UHconv)** inside an LSTM encoder for rainfall–runoff prediction. UHconv parameterises a Gamma-distributed routing kernel whose shape parameters (α, β) are predicted from catchment attributes, ensuring basin-specific, learnable, and mass-conserving flood routing.

## Results

Running the commands above regenerates all metrics, the ablation study, and the manuscript figures locally under `results/` (which is **not** stored in this repository). Numerical results are intentionally **not** pre-published here to avoid disclosing unpublished findings; reviewers reproduce them by running the code.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CAMELS-US data (requires Kaggle token)
#    Get your token from https://www.kaggle.com/settings
#    On Windows:  set KAGGLE_API_TOKEN=<your_token>
#    On Linux:    export KAGGLE_API_TOKEN=<your_token>
cd src
python kaggle_download_hydromtl.py

# 3. Run main comparison
python run_experiments.py --models PINN_UHConv --seeds 42 --n_basins 100 --epochs 15

# 4. Full reproduction (see reproduce.md for details)
python run_experiments.py    # Experiment 1
python run_ablation.py       # Experiment 2
python run_sensitivity.py --n_basins 100 --epochs 15 --seed 42 --batch_size 256 --seq_length 180  # Experiment 3
python run_robustness.py     # Experiment 4
python run_statistics.py     # Statistical analysis
python generate_plots.py     # Generate figures
```

See **[reproduce.md](reproduce.md)** for complete reproduction instructions.

## Repository Structure

```
03_hydrology_runoff/
├── src/
│   ├── config.py              # All hyperparameters and paths
│   ├── data_loader.py         # CAMELS-US loading, preprocessing, normalisation
│   ├── models.py              # PINN-UHConv and 6 baseline models
│   ├── losses.py              # NSE loss + mass-balance + extreme-event weighting
│   ├── train.py               # Training loop (train_one_model)
│   ├── evaluate.py            # Evaluation metrics (NSE, KGE, FHV, etc.)
│   ├── run_experiments.py     # Experiment 1: Main comparison
│   ├── run_ablation.py        # Experiment 2: Component ablation
│   ├── run_sensitivity.py     # Experiment 3: Parameter sensitivity
│   ├── run_robustness.py      # Experiment 4: Robustness analysis
│   ├── run_statistics.py      # Statistical tests (t-test, ANOVA, Cohen's d)
│   └── generate_plots.py      # Publication figures (300 dpi)
├── results/
│   ├── experiment1/           # 35 per-run JSON files (7 models × 5 seeds)
│   ├── experiment2_ablation/  # 15 per-run JSON files (5 configs × 3 seeds)
│   ├── experiment3_sensitivity/ # 16 per-run JSON files
│   ├── statistics/            # Statistical analysis JSON
│   └── *.json                 # Aggregated result files
├── plots/                     # Generated figures (PNG, 300 dpi)
├── paper/
│   └── manuscript.md          # Full manuscript
├── requirements.txt
├── reproduce.md               # Detailed reproduction guide
└── README.md
```

## Model Architecture

PINN-UHConv combines four components:

1. **FiLM-modulated LSTM encoder** — static catchment attributes modulate LSTM gate activations via a learned affine transform.
2. **Differentiable Unit-Hydrograph Convolution (UHconv)** — a Gamma-distributed routing kernel parameterised by (α, β) predicted from basin attributes. Proven causal and mass-conserving (Theorem 1).
3. **Mass-balance constraint** — a scale-invariant loss term coupling storage, evapotranspiration, and discharge (Proposition 2).
4. **Extreme-event weighted loss** — smooth bounded weighting amplifies flood-peak contribution without destabilising low-flow gradients.

## Dataset

**CAMELS-US** (Newman et al., 2015; Addor et al., 2017):
- 100 basins from the HCDN network
- Daymet forcing: PRCP, TMEAN, TMAX, TMIN, DAYL (+ sin/cos day-of-year)
- 16 static attributes (topography, soil, geology, climate, vegetation)
- Period: 1980–2010 (15 train / 5 val / 10 test)
- Split: 70 train / 15 val / 15 test basins (held-out basins for regionalisation)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

```bibtex
@article{zeng2026pinnuhconv,
  title={Physics-Informed Neural Network with Differentiable Unit-Hydrograph Convolution for Rainfall--Runoff Modeling},
  author={Zeng, Jingyuan and Zeng, Ming and Guo, Jianghong and Jiang, Chuanxian and Feng, Yafen},
  journal={Journal of Hydrology},
  year={2026}
}
```
