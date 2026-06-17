# Uncertainty-Aware Perovskite Stability Prediction (PACT-Final)

Code and data accompanying the manuscript:
**"Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance"** by Xumingyong Feng, submitted to *Computational Materials Science*.

## Overview

This repository implements the **PACT-Final** framework for predicting the formation energy and thermodynamic stability (energy above the convex hull) of ABO3 perovskite oxides, with:

- **Physics-informed point prediction** (KernelRidge baseline + GBDT stacking residual)
- **Conditional conformal intervals** (Conformalized Quantile Regression, CQR) with finite-sample coverage guarantees
- **Multi-method applicability domain** (ensemble sigma, k-NN distance, PCA leverage)
- **Symbolic regression consensus** for interpretability (gplearn, 50 equations)
- **Leave-one-element-out** extrapolation assessment (68 elements)

## Requirements

- Python 3.10+
- Key dependencies: numpy, pandas, scipy, scikit-learn (>=1.5), lightgbm, xgboost, pymatgen, matminer, optuna, gplearn, shap, matplotlib

```bash
pip install -r requirements.txt
```

## Data

The primary dataset is **wolverton_oxides** (4914 ABO3 perovskites), available via matminer:

```python
from matminer.datasets import load_dataset
df = load_dataset("wolverton_oxides")
```

Processed feature matrices (`data/processed/perovskite_features.csv`) and all out-of-fold predictions (`results/metrics/`) are included in this repository for full reproducibility. A permanent archived snapshot is deposited on Zenodo: [DOI to be inserted upon acceptance].

## Quick start

Reproduce all main results with a single command:

```bash
python src/pact_final.py
```

This runs the unified pipeline (5-fold nested CV, approximately 15 min on a laptop) and writes:
- `results/metrics/pact_final_results.csv` - summary metrics (R2, MAE, PICP, ECE)
- `results/metrics/pact_final_oof_*.csv` - per-sample out-of-fold predictions (for figures)

## Repository structure

```
src/
  pact_final.py            Unified pipeline (main results)
  features.py              Feature engineering (Magpie + physics descriptors)
  utils.py                 Shared utilities (PHYS_FEATURES, load_features)
  conformal.py             Split conformal prediction
  cqr_conformal.py         Conformalized Quantile Regression
  ad_methods.py            Applicability domain (sigma/kNN/leverage)
  stats_eval.py            Statistical evaluation (CI, ECE, Wilcoxon)
  sr_ensemble.py           Symbolic regression ensemble (consensus)
  optuna_search_v2.py      Hyperparameter optimization
  gbdt_stacking.py         GBDT stacking
  make_figures.py          Figure generation (F2-F6)
  make_figures_schematic.py Schematic figures (F1, graphical abstract)
data/processed/            Processed features
results/metrics/           All experimental results (CSV)
results/discovery/         Candidate materials + validation
paper/                     Manuscript, figures, tables, cover letter
docs/                      Full development log (V1-V9)
requirements.txt
LICENSE
```

## Key results

| Target | R2 | MAE (eV/atom) | CQR PICP | CQR ECE | ECE improvement |
|---|---|---|---|---|---|
| Formation energy | 0.914 | 0.186 | 0.802 | 0.034 | 60 percent |
| Energy above hull | 0.800 | 0.171 | 0.807 | 0.049 | 43 percent |

## License

MIT License. See `LICENSE`.

## Citation

If you use this code or data, please cite:

    Feng, X. (2026). Uncertainty-Aware Prediction of Perovskite Oxide Stability with
    Conditional Conformal Intervals and Applicability-Domain Guidance.
    Computational Materials Science (submitted).

## Contact

Xumingyong Feng - Weiyang College, Tsinghua University
fxmy23@mails.tsinghua.edu.cn

