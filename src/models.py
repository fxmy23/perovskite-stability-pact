"""
================================================================
单任务基准建模模块 (Baseline Models)
================================================================
论文的"方法基准线": 五种回归模型在形成能 E_f / 凸包能 E_hull 上的对比。

这是后续 PGML、多任务、不确定性模块的对照基准 (baseline)。

★ 重要: 所有模型 n_jobs=1, 避免 Windows multiprocessing 问题
   (见 troubleshooting_log.md F-01)。scikit-learn 默认 threading 不 spawn,
   但 XGBoost/LightGBM 的 n_jobs 会真正多进程, 全部显式设 1。

评估协议:
  - 5 折交叉验证 (KFold, shuffle=True)
  - 指标: RMSE / MAE / R²

模型清单: RF / GBDT / SVR / XGBoost / LightGBM

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def get_models() -> dict:
    """
    返回基准回归模型。★ 全部 n_jobs=1 避免 spawn。

    性能优化(基于单折计时):
      - RF: n_estimators=100, max_features=sqrt (单折~19s)
      - 删除 sklearn GBDT (太慢, 被 LightGBM 完全替代, 快30倍)
      - SVR: 保留 (单折0.6s, 但对特征缩放敏感)
      - XGBoost/LightGBM: 现代 boosting (单折<5s)
    """
    # P0-1 修复: 加 SimpleImputer 到 Pipeline, 在 CV 折内 fit impute, 避免泄露
    from sklearn.impute import SimpleImputer
    models = {
        "RF": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=100, max_features="sqrt",
                random_state=42, n_jobs=1,
            )),
        ]),
        "SVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, gamma="scale")),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("xgb", XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=1, verbosity=0,
            )),
        ])
    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )),
        ])
    return models


def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_benchmark(
    df: pd.DataFrame,
    target: str,
    feat_prefixes: tuple = ("magpie_", "phys_"),  # P0-1: 排除struct避免泄露
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    对所有模型做 5 折交叉验证。
    返回 (指标DataFrame, 每模型逐样本预测字典)。
    """
    feat_cols = [c for c in df.columns if c.startswith(feat_prefixes)]
    X = df[feat_cols].values
    y = df[target].values
    print(f"[MODEL] 目标: {target} | 样本 {len(y)} | 特征 {X.shape[1]} 维")

    models = get_models()
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"  >> {name} ...", end=" ", flush=True)
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        m = evaluate(y, y_pred)
        predictions[name] = y_pred
        m["model"] = name
        results.append(m)
        print(f"RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R²={m['R2']:.4f}", flush=True)

    df_metrics = pd.DataFrame(results)
    df_metrics["target"] = target
    df_metrics["n_samples"] = len(y)
    df_metrics["n_features"] = X.shape[1]
    return df_metrics, predictions


def main():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils import load_features
    df = load_features()
    if df is None or len(df) == 0:
        raise SystemExit("[ERROR] 特征数据为空, 先运行 features.py")

    print("=" * 60)
    print("  单任务基准建模 (五模型对比, 5折CV)")
    print("=" * 60)

    all_metrics = []
    all_preds = {}
    targets = {
        "formation_energy_per_atom": "形成能 Ef",
        "energy_above_hull": "凸包能 Ehull",
    }

    for target_col, target_cn in targets.items():
        print(f"\n### 目标: {target_cn} ###")
        df_m, preds = run_benchmark(df, target=target_col)
        all_metrics.append(df_m)
        all_preds[target_col] = preds

    df_all = pd.concat(all_metrics, ignore_index=True)
    out_path = METRICS_DIR / "baseline_metrics.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 基准指标: {out_path}")

    # 逐样本预测持久化 (供 parity plot)
    for target_col, preds in all_preds.items():
        pred_df = pd.DataFrame(preds)
        pred_df.insert(0, "y_true", df[target_col].values)
        pred_path = METRICS_DIR / f"baseline_preds_{target_col}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("[SAVE] 逐样本预测已保存 (供 parity plot)")

    print("\n" + "=" * 60)
    print("  基准性能汇总")
    print("=" * 60)
    print(df_all[["target", "model", "RMSE", "MAE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
