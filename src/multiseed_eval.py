"""
================================================================
多 seed 统计严谨性评估 (Statistical Rigor)
================================================================
文献要求: 所有核心实验报 mean±std (≥5 seeds)。
本模块对基准模型 + 分类指标做多 seed 评估, 输出置信区间。

输出:
  results/metrics/multiseed_baseline.csv  基准模型 mean±std
  results/metrics/multiseed_classification.csv  分类指标 mean±std

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
)

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols
from src.classification_eval import compute_daf, STABLE_THRESH

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 2024]


def run_multiseed_baseline(df, target, n_seeds=N_SEEDS):
    """对 LightGBM 基准做多 seed 5折CV, 返回每 seed 的指标。"""
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values

    results = []
    for i, seed in enumerate(SEEDS[:n_seeds]):
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=seed, n_jobs=1, verbose=-1,
            )),
        ])
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        results.append({
            "seed": seed,
            "R2": float(r2_score(y, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
            "MAE": float(mean_absolute_error(y, y_pred)),
        })
        print(f"    seed={seed}: R²={results[-1]['R2']:.4f}", flush=True)
    return pd.DataFrame(results)


def run_multiseed_classification(df, target="energy_above_hull", n_seeds=N_SEEDS):
    """对分类指标做多 seed 评估。"""
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y_reg = df[target].values
    y_cls = (y_reg < STABLE_THRESH).astype(int)

    results = []
    for i, seed in enumerate(SEEDS[:n_seeds]):
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        n = len(y_reg)
        oof_pred = np.empty(n)
        for tr, te in cv.split(X):
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lgbm", LGBMRegressor(
                    n_estimators=200, num_leaves=31, learning_rate=0.1,
                    random_state=seed, n_jobs=1, verbose=-1,
                )),
            ])
            model.fit(X[tr], y_reg[tr])
            oof_pred[te] = model.predict(X[te])

        pred_stable = (oof_pred < STABLE_THRESH).astype(int)
        results.append({
            "seed": seed,
            "Precision": float(precision_score(y_cls, pred_stable, zero_division=0)),
            "Recall": float(recall_score(y_cls, pred_stable, zero_division=0)),
            "F1": float(f1_score(y_cls, pred_stable, zero_division=0)),
            "AUC_ROC": float(roc_auc_score(y_cls, -oof_pred)),
            "DAF_top10%": compute_daf(y_reg, oof_pred, top_frac=0.10),
            "DAF_top5%": compute_daf(y_reg, oof_pred, top_frac=0.05),
        })
        print(f"    seed={seed}: F1={results[-1]['F1']:.3f} DAF={results[-1]['DAF_top10%']:.2f}", flush=True)
    return pd.DataFrame(results)


def summarize(df_results, name):
    """输出 mean±std 摘要。"""
    print(f"\n  === {name} (mean±std, N={len(df_results)}) ===", flush=True)
    metrics = [c for c in df_results.columns if c not in ("seed", "target")]
    for m in metrics:
        vals = df_results[m].dropna()
        print(f"    {m:15s}: {vals.mean():.4f} ± {vals.std():.4f}  "
              f"[{vals.min():.4f}, {vals.max():.4f}]", flush=True)


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print(f"  多 seed 统计严谨性评估 (N_SEEDS={N_SEEDS})", flush=True)
    print("=" * 60, flush=True)

    # --- 基准模型多 seed ---
    all_base = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### 基准模型: {target} ###", flush=True)
        df_res = run_multiseed_baseline(df, target)
        df_res["target"] = target
        all_base.append(df_res)
        summarize(df_res, f"Baseline {target}")

    df_base = pd.concat(all_base, ignore_index=True)
    df_base.to_csv(METRICS_DIR / "multiseed_baseline.csv", index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] multiseed_baseline.csv", flush=True)

    # --- 分类指标多 seed ---
    print(f"\n### 分类指标: energy_above_hull ###", flush=True)
    df_cls = run_multiseed_classification(df)
    summarize(df_cls, "Classification")
    df_cls.to_csv(METRICS_DIR / "multiseed_classification.csv", index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] multiseed_classification.csv", flush=True)


if __name__ == "__main__":
    main()
