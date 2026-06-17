# Tables for Manuscript

## Table 1: Model comparison for formation energy and hull energy prediction (5-fold CV, bootstrap 95% CI)

| Model | Formation E R² | Formation MAE | Hull E R² | Hull MAE |
|---|---|---|---|---|
| Pure LightGBM | 0.912 | 0.193 | 0.793 | 0.179 |
| GBDT stacking (LGB+XGB+HGB) | 0.918 | 0.181 | 0.813 | 0.169 |
| **PACT-Final (KRR + stacking)** | **0.914 [0.907, 0.922]** | **0.186** | **0.800 [0.781, 0.816]** | **0.171** |
| Ref. [MDPI 2025] (different dataset) | 0.928 | — | — | — |

Note: PACT-Final uses KRR physics baseline + stacking residual. The GBDT stacking row uses stacking directly on y (no physics baseline). Ref. values are on a Materials-Project-derived dataset; direct comparison is limited by reference-frame differences.

## Table 2: Conditional conformal (CQR) vs. standard split conformal (fair LightGBM-family baseline)

| Target | Method | PICP | MPIW (eV/atom) | ECE | ECE improvement |
|---|---|---|---|---|---|
| Formation energy | Standard conformal | 0.810 | 0.611 | 0.085 | — |
| Formation energy | **CQR** | **0.802** | 0.823 | **0.034** | **−60%** |
| Hull energy | Standard conformal | 0.815 | 0.566 | 0.086 | — |
| Hull energy | **CQR** | **0.807** | 0.719 | **0.049** | **−43%** |

PICP: prediction interval coverage probability (nominal 0.80). MPIW: mean prediction interval width. ECE: expected calibration error (10-σ-bins). Both methods satisfy PICP ≥ 0.80 (marginal coverage guarantee).

## Table 3: Candidate stable perovskites (predicted E_hull < 0.05 eV/atom within trusted AD)

| Formula | A site | B site | Pred. E_hull | σ | In matbench | Goldschmidt zone | LOEO robust | Validation grade |
|---|---|---|---|---|---|---|---|---|
| LaTiO₃ | La | Ti | 0.005 | 0.022 | Yes | Yes | No | Moderate |
| PrCoO₃ | Pr | Co | 0.006 | 0.024 | No | Yes | No | Weak |
| GdCoO₃ | Gd | Co | 0.016 | 0.027 | No | Yes | No | Weak |
| YbCoO₃ | Yb | Co | 0.017 | 0.028 | No | Yes | No | Weak |
| DyCoO₃ | Dy | Co | 0.019 | 0.029 | No | Yes | No | Weak |
| ErFeO₃ | Er | Fe | 0.019 | 0.030 | No | Yes | No | Weak |
| YCoO₃ | Y | Co | 0.025 | 0.031 | Yes | Yes | No | Moderate |
| SrNpO₃ | Sr | Np | 0.025 | 0.035 | No | No | No | Weak |
| LaAlO₃ | La | Al | 0.033 | 0.037 | Yes | Yes | No | Weak |

Validation grade combines database cross-reference (matbench presence), Goldschmidt synthesizability (t ∈ [0.8, 1.05], μ ∈ [0.4, 0.9]), and leave-element-out robustness. No candidate achieved "strong" grade (would require DFT or experimental verification).

## Supplementary Table S1: Applicability-domain comparison (formation energy)

| AD method | Trusted R² | Untrusted R² | R² gap | Agreement with σ |
|---|---|---|---|---|
| Ensemble σ | 0.945 | 0.886 | +0.059 | 100% |
| k-NN distance (PCA-20) | 0.915 | 0.867 | +0.048 | 53% |
| PCA leverage | 0.916 | 0.822 | +0.094 | 51% |

## Supplementary Table S2: Optuna-selected hyperparameters (formation energy)

| Hyperparameter | Value |
|---|---|
| n_estimators | 450 |
| num_leaves | 39 |
| learning_rate | 0.078 |
| max_depth | 10 |
| min_child_samples | 17 |
| subsample | 0.902 |
| colsample_bytree | 0.728 |
| reg_alpha | 0.489 |
| reg_lambda | 0.278 |

Note: Hyperparameters selected on full dataset via TPE (25 trials, 3-fold inner CV). Nested CV estimates ~0.005 R² optimistic bias.
