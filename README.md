# Uncertainty-Aware Perovskite Stability Prediction (PACT-Final)

Code and data accompanying the manuscript:

> **"Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance"**
> Xumingyong Feng — *Computational Materials Science* (submitted)

---

## Overview

This repository implements the **PACT-Final** framework for predicting the formation energy (E_f) and thermodynamic stability (energy above the convex hull, E_hull) of ABO₃ perovskite oxides. The framework integrates four components into a single nested cross-validation pipeline:

1. **Physics-informed point predictor** — KernelRidge baseline on 14 physics descriptors + GBDT stacking residual (LightGBM / XGBoost / HistGBT) on the full 110-dimensional descriptor.
2. **Conditional conformal intervals** — Conformalized Quantile Regression (CQR) with a finite-sample, distribution-free 80% marginal coverage guarantee.
3. **Multi-criteria applicability domain** — ensemble σ, k-NN distance, and PCA leverage, each targeting a distinct failure mode.
4. **Symbolic-regression interpretability cross-check** — 50 independent equations (10 seeds × 5 folds), aggregated as feature-consensus frequency.

A 73-element leave-one-element-out (LOEO) evaluation quantifies extrapolation behavior.

## Key results

| Target | R² | MAE (eV/atom) | CQR PICP | CQR ECE | ECE reduction vs. standard conformal |
|---|---|---|---|---|---|
| Formation energy | 0.914 | 0.186 | 0.802 | 0.034 | 60% |
| Energy above hull | 0.800 | 0.171 | 0.807 | 0.049 | 43% |

Trusted-region R² (σ < median) reaches 0.945 (formation) and 0.873 (hull). LOEO mean per-element R² = 0.739 across 73 elements, with no element yielding a negative R².

## Requirements

- Python 3.10+
- Key dependencies: numpy, pandas, scipy, scikit-learn (≥1.5), lightgbm, xgboost, pymatgen, matminer, optuna, gplearn, shap, matplotlib

```bash
pip install -r requirements.txt
```

## Data

The primary dataset is **wolverton_oxides** (4,914 ABO₃ perovskites), available via matminer:

```python
from matminer.datasets import load_dataset
df = load_dataset("wolverton_oxides")
```

Processed feature matrices and all out-of-fold predictions are included for full reproducibility:

- `data/processed/perovskite_features.csv` — 110-dimensional descriptor matrix (96 Magpie + 14 physics)
- `data/processed/perovskite_clean.csv` — cleaned raw data with targets
- `results/metrics/pact_final_oof_*.csv` — per-sample out-of-fold predictions, CQR intervals, AD flags
- `results/metrics/pact_final_results.csv` — summary metrics (R², MAE, PICP, ECE)
- `results/discovery/candidate_validation.csv` — nine candidate materials with three-tier validation

A permanent archived snapshot will be deposited on Zenodo upon acceptance.

## Reproducing the results

### Main results (Table 1, Table 2, Figs. 2, 4–8)

```bash
python src/pact_final.py
```

This runs the unified pipeline (5-fold CV with inner 3-fold stacking, ~15 min on a laptop) and writes the summary metrics and per-sample out-of-fold predictions to `results/metrics/`.

### Other components

| Command | Produces |
|---|---|
| `python src/sr_ensemble.py` | Symbolic regression consensus (Fig. 8b, 50 equations) |
| `python src/loeo_sr_full.py` | 73-element LOEO extrapolation (Fig. 7) |
| `python src/make_figures.py` | Publication figures (Figs. 4–8) |
| `python src/make_figure_cqr_heteroscedasticity.py` | Fig. 2 (CQR heteroscedasticity) |
| `python src/make_figures_schematic.py` | Fig. 1 (workflow) + graphical abstract |

## Repository structure

```
src/
  pact_final.py              Unified pipeline — main results (R², CQR, AD)
  features.py                Feature engineering (Magpie + physics descriptors)
  utils.py                   Shared utilities (PHYS_FEATURES, load_features)
  conformal.py               Split conformal prediction
  cqr_conformal.py           Conformalized Quantile Regression (CQR)
  ad_methods.py              Applicability domain (σ / kNN / leverage)
  stats_eval.py              Statistical evaluation (CI, ECE, Wilcoxon)
  sr_ensemble.py             Symbolic regression ensemble (consensus)
  loeo_sr_full.py            Leave-one-element-out extrapolation
  gbdt_stacking.py           GBDT stacking
  optuna_search_v2.py        Hyperparameter optimization
  make_figures.py            Figure generation (Figs. 4–8)
  make_figure_cqr_heteroscedasticity.py   Fig. 2
  make_figures_schematic.py  Fig. 1 + graphical abstract
data/
  raw/                       Raw datasets (wolverton_oxides, matbench_perovskites)
  processed/                 Cleaned feature matrices
results/
  metrics/                   All experimental results (CSV)
  discovery/                 Candidate materials + validation
paper/                       Manuscript, figures, cover letter, highlights
docs/                        Development log and audit notes
requirements.txt
LICENSE
```

## License

MIT License. See [`LICENSE`](LICENSE).

## Citation

If you use this code or data, please cite:

```bibtex
@article{feng2026perovskite,
  title={Uncertainty-Aware Prediction of Perovskite Oxide Stability with
         Conditional Conformal Intervals and Applicability-Domain Guidance},
  author={Feng, Xumingyong},
  journal={Computational Materials Science},
  year={2026},
  note={Submitted}
}
```

## Contact

**Xumingyong Feng**
Weiyang College, Tsinghua University, Beijing 100084, China
Email: fxmy23@mails.tsinghua.edu.cn
