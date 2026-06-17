"""
================================================================
Stacking 集成 (Stack Generalization)
================================================================
论文第三阶段: 借鉴 Nature Comm 2024 的 stack generalization 策略。
多模型融合提升泛化性能。

策略:
  基模型: LightGBM + RandomForest + Ridge (异质模型, 误差不相关)
  元模型: Ridge (简单线性融合, 避免过拟合)
  评估: 5折CV, OOF预测做 stacking

依赖: scikit-learn, lightgbm, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def get_base_models(seed=42):
    """返回异质基模型字典。"""
    models = {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100, max_features="sqrt",
                random_state=seed, n_jobs=1)),
        ]),
    }
    if HAS_LGBM:
        models["lgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=seed, n_jobs=1, verbose=-1)),
        ])
    return models


def run_stacking(df, target, n_splits=5, seed=42):
    """
    Stacking 集成评估。
    ★ 修复: 嵌套 CV — 外层评估时, 内层重新生成 OOF 训练元模型。
    避免"用同一组KFold既生成OOF又评估"的泄露。
    """
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    base_model_names = list(get_base_models(seed).keys())
    n = len(y)

    print(f"[STACK] 目标: {target}, 基模型: {base_model_names}", flush=True)
    print(f"  (嵌套CV: 外层{n_splits}折评估 / 内层{n_splits}折生成OOF)", flush=True)

    # ---- 外层: 评估 stacking ----
    stacking_pred = np.empty(n)
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        X_tr_outer, X_te_outer = X[train_idx], X[test_idx]
        y_tr_outer = y[train_idx]

        # 内层: 在外层训练集上生成 OOF (训练元模型用)
        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        meta_X_inner = np.zeros((len(train_idx), len(base_model_names)))
        for j, name in enumerate(base_model_names):
            oof_inner = np.empty(len(train_idx))
            for itr, ite in inner_cv.split(X_tr_outer):
                model = get_base_models(seed)[name]
                model.fit(X_tr_outer[itr], y_tr_outer[itr])
                oof_inner[ite] = model.predict(X_tr_outer[ite])
            meta_X_inner[:, j] = oof_inner

        # 训练元模型
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X_inner, y_tr_outer)

        # 在外层训练集上 fit 基模型 (全量), 预测外层测试集
        meta_X_test = np.zeros((len(test_idx), len(base_model_names)))
        for j, name in enumerate(base_model_names):
            model = get_base_models(seed)[name]
            model.fit(X_tr_outer, y_tr_outer)
            meta_X_test[:, j] = model.predict(X_te_outer)

        stacking_pred[test_idx] = meta.predict(meta_X_test)

    r2_stack = r2_score(y, stacking_pred)
    mae_stack = mean_absolute_error(y, stacking_pred)

    # 基模型单独评估 (用简单 CV 做基线对比)
    from sklearn.model_selection import cross_val_predict
    cv_simple = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base_metrics = {}
    for name in base_model_names:
        model = get_base_models(seed)[name]
        oof = cross_val_predict(model, X, y, cv=cv_simple, n_jobs=1)
        base_metrics[name] = {"R2": r2_score(y, oof), "MAE": mean_absolute_error(y, oof)}
        print(f"  基模型 {name}: R²={base_metrics[name]['R2']:.4f}", flush=True)
    print(f"  Stacking: R²={r2_stack:.4f} MAE={mae_stack:.4f}", flush=True)

    # 简单平均融合 (对比)
    from sklearn.model_selection import cross_val_predict as _cvp
    oof_simple = {}
    for name in base_model_names:
        model = get_base_models(seed)[name]
        oof_simple[name] = _cvp(model, X, y, cv=cv_simple, n_jobs=1)
    avg_pred = np.mean([oof_simple[name] for name in base_model_names], axis=0)
    r2_avg = r2_score(y, avg_pred)
    print(f"  简单平均: R²={r2_avg:.4f}", flush=True)

    return {
        "target": target,
        **{f"{name}_R2": base_metrics[name]["R2"] for name in base_model_names},
        "stacking_R2": r2_stack,
        "stacking_MAE": mae_stack,
        "avg_R2": r2_avg,
    }


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  Stacking 集成 (Stack Generalization)", flush=True)
    print("=" * 60, flush=True)

    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        res = run_stacking(df, target)
        all_res.append(res)

    df_out = pd.DataFrame(all_res)
    out_path = METRICS_DIR / "stacking_results.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  Stacking 汇总", flush=True)
    print("=" * 60, flush=True)
    for r in all_res:
        print(f"\n  {r['target']}:", flush=True)
        for k, v in r.items():
            if k != "target" and "R2" in k:
                print(f"    {k}: {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
