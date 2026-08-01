# Reproduction Guide — PINN-UHConv

This document describes how to reproduce all experiments reported in the paper.

## 1. Environment

### Hardware
- GPU: NVIDIA RTX 2000 Pro (16 GB VRAM)
- CPU: Intel Xeon W7-2595X (24 cores, 2.5–4.8 GHz)
- RAM: 48 GB DDR5 RDIMM
- OS: Windows 11 Professional

### Software
- Python 3.13
- CUDA 13.2
- PyTorch 2.12.0

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Data

The experiments use **CAMELS-US** (Catchment Attributes and Meteorology for Large-sample Studies, US subset).

### Option A: Kaggle download (recommended)
```bash
# Set Kaggle credentials (replace with your own token from https://www.kaggle.com/settings)
set KAGGLE_API_TOKEN=<your_kaggle_token>

# Download and extract
cd src
python kaggle_download_hydromtl.py
```

Expected data layout after extraction:
```
data/kaggle_hydromtl/camels/camels_us/
├── basin_set_full_res/          # basin shapefile
│   └── HCDN_nhru_final_671.shp
├── camels_topo.txt              # basin attributes
├── camels_clim.txt
├── camels_soil.txt
├── camels_geol.txt
├── camels_vege.txt
└── basin_timeseries_v1p2_metForcing_obsFlow/
    └── basin_dataset_public_v1p2/
        ├── basin_mean_forcing/daymet/<HUC2>/<BASIN_ID>_lump_cida_forcing_leap.txt
        └── usgs_streamflow/<BASIN_ID>_usgs_streamflow_qc.txt
```

### Option B: Direct download from NCAR
Download from https://ral.ucar.edu/solutions/products/camels and place files in the same layout.

## 3. Preprocessing

The data preprocessing (loading, normalisation, basin splitting) is cached on first run:
```bash
cd src
python -c "from data_loader import load_full_pipeline; load_full_pipeline(n_basins=100, cache=True)"
```
This creates `results/cache/full_pipeline.npz` (~200 MB) used by all subsequent experiments.

## 4. Experiments

All commands below are run from the `src/` directory. Each experiment saves results to `results/`.

### Experiment 1: Main Comparison (7 models × 5 seeds)

```bash
python run_experiments.py --models LSTM EA_LSTM MTS_LSTM Transformer Phys_LSTM UH_LSTM PINN_UHConv --seeds 42 2024 7 123 999 --n_basins 100 --epochs 15 --batch_size 256 --seq_length 180
```

**Output:**
- `results/experiment1/train_<MODEL>_seed<SEED>.json` — 35 per-run files
- `results/experiment1_baseline_comparison_results.json` — aggregated comparison

**Expected runtime:** ~2 hours (35 runs × ~3.5 min/run)

### Experiment 2: Ablation (5 configs × 3 seeds)

```bash
python run_ablation.py --configs full no_uhconv no_mass_balance no_static_modulation no_extreme_weighting --seeds 42 2024 7 --n_basins 100 --epochs 15 --batch_size 256 --seq_length 180
```

**Output:**
- `results/experiment2_ablation/<config>_seed<seed>.json` — 15 per-run files
- `results/experiment2_ablation_results.json` — aggregated ablation summary

**Expected runtime:** ~7.5 hours (15 runs × ~30 min/run)

### Experiment 3: Sensitivity (5 parameters × 3-4 values)

```bash
python run_sensitivity.py --n_basins 100 --epochs 15 --seed 42 --batch_size 256 --seq_length 180
```

**Parameters swept:**
| Parameter | Values |
|-----------|--------|
| `lambda_mass` | 0.0, 0.01, 0.1, 1.0 |
| `uh_kernel_size` | 20, 60, 120 |
| `hidden_size` | 64, 128, 256 |
| `lambda_extreme` | 0.0, 0.5, 2.0 |
| `seq_length` | 90, 180, 365 |

**Output:**
- `results/experiment3_sensitivity/<param>_<value>_seed42.json` — 16 per-run files
- `results/experiment3_sensitivity_results.json` — aggregated elasticity analysis

**Expected runtime:** ~10 hours (16 runs × ~36 min/run)

### Experiment 4: Robustness

```bash
python run_robustness.py --n_basins 100 --epochs 15 --seed 42 --batch_size 256 --seq_length 180
```

**Analysis:**
- Noise robustness: Gaussian noise std = {0.0, 0.05, 0.10, 0.20, 0.30}
- Missing-data robustness: missing rate = {0.0, 0.05, 0.10, 0.20, 0.30}
- Unseen-basin transfer: evaluate on 4 different test splits

**Output:**
- `results/experiment4_robustness_results.json`

**Expected runtime:** ~1 hour (1 training run + 14 evaluation passes)

### Statistics & Plots

```bash
# Compute statistical tests (paired t-test, ANOVA, Cohen's d, elasticity)
python run_statistics.py

# Generate publication-quality figures (300 dpi PNG)
python generate_plots.py
```

**Output:**
- `results/statistics/statistical_analysis.json`
- `plots/figure1_architecture.png`
- `plots/figure2_comparison.png`
- `plots/figure3_ablation.png`
- `plots/figure4_sensitivity.png`
- `plots/figure5_robustness.png` (optional)

## 5. Data-to-Paper Traceability

Every numerical value in the paper can be traced to a specific file:

| Paper section | Data source |
|---------------|------------|
| Table 1 (Main comparison) | `results/experiment1_baseline_comparison_results.json` → `per_model.<MODEL>.NSE_median_mean` etc. |
| Table 2 (Statistical tests) | `results/statistics/statistical_analysis.json` → `paired_t_test` |
| Table 3 (Ablation) | `results/experiment2_ablation_results.json` → `per_config.<CONFIG>` |
| Tables 4–5 (Sensitivity/elasticity) | `results/experiment3_sensitivity_results.json` → `per_param.<P>.sweep`, `elasticity` |
| Tables 6–8 (Robustness) | `results/experiment4_robustness_results.json` |
| Table 9 (Computational performance) | `results/experiment1/train_<MODEL>_seed<SEED>.json` → `train_time_sec`, `n_params` |
| Tables 10–11 (Case study) | `results/case_study_results.json` |
| Figures 1–6 | `plots/figure*.png` |

## 6. Hyperparameters

All hyperparameters are defined in `src/config.py`:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-5 |
| Batch size | 256 |
| Sequence length | 180 days |
| LSTM hidden size | 128 |
| LSTM layers | 1 |
| Dropout | 0.3 |
| Gradient clipping | 1.0 |
| UH kernel size | 60 |
| λ_mass | 0.01 |
| λ_extreme | 0.5 |
| Extreme quantile | 0.95 |
| Epochs | 15 |
| Early stopping patience | 5 |
| Seeds | {42, 2024, 7, 123, 999} |
| n_basins | 100 (70 train / 15 val / 15 test) |
