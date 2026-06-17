"""
================================================================
优化: GBDT Stacking (LightGBM + XGBoost + CatBoost) — 提R²
================================================================
文献: PMC 2023 多GBDT集成稳定提升。
方法: 3个GBDT的OOF stacking + Ridge元学习器 (嵌套CV无泄露)。

★ pre-debug: 验证3个基模型独立R² + 多样性(预测相关性<0.99)
★ post-debug: stacking R² > 最佳单模型 + 无泄露验证

依赖: lightgbm, xgboost, catboost, scikit-learn
作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Optuna最佳超参 (形成能)
LGB_PARAMS = dict(n_estimators=450, num_leaves=39, learning_rate=0.078,
                  max_depth=10, min_child_samples=17, subsample=0.902,
                  colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278)


def get_base_models(seed=42):
    """返回3个GBDT基模型工厂 (不同算法保证多样性)。

    ★ CatBoost在4914样本上太慢(37s/fit), 改用HistGradientBoosting(快)。
    多样性来源: LightGBM(leaf-wise) vs XGBoost(level-wise) vs HistGBT(sklearn直方图)。
    """
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "LightGBM": lambda: LGBMRegressor(**LGB_PARAMS, random_state=seed,
                                          n_jobs=1, verbose=-1, subsample_freq=1),
        "XGBoost": lambda: XGBRegressor(n_estimators=450, max_depth=8,
                                        learning_rate=0.07, subsample=0.8,
                                        colsample_bytree=0.7, reg_alpha=0.5,
                                        reg_lambda=1.0, random_state=seed,
                                        n_jobs=1, verbosity=0),
        "HistGBT": lambda: HistGradientBoostingRegressor(
            max_iter=450, max_depth=8, learning_rate=0.07,
            l2_regularization=1.0, random_state=seed),
    }


def stacking_cv(X, y, target_name, n_splits=5, seed=42):
    """嵌套CV stacking: 外层评估, 内层生成OOF基模型预测。"""
    factories = get_base_models(seed)
    model_names = list(factories.keys())
    outer_cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(y)

    # 单模型OOF + stacking OOF
    oof_single = {name: np.full(n, np.nan) for name in model_names}
    oof_stack = np.full(n, np.nan)

    for fold, (tr, te) in enumerate(outer_cv.split(X)):
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))

        # 内层CV生成基模型OOF (用于训练元学习器)
        inner_cv = KFold(3, shuffle=True, random_state=seed+fold)
        meta_tr = np.zeros((len(tr), len(model_names)))
        meta_te = np.zeros((len(te), len(model_names)))
        for j, name in enumerate(model_names):
            # 内层OOF
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in inner_cv.split(Xtr):
                m = factories[name]().fit(Xtr[itr], y[tr][itr])
                inner_oof[ite] = m.predict(Xtr[ite])
            meta_tr[:, j] = inner_oof
            # 全训练折fit后预测测试折
            m_full = factories[name]().fit(Xtr, y[tr])
            meta_te[:, j] = m_full.predict(Xte)
            oof_single[name][te] = meta_te[:, j]

        # 元学习器 (Ridge, 在内层OOF上训练)
        meta_lr = Ridge(alpha=1.0).fit(meta_tr, y[tr])
        oof_stack[te] = meta_lr.predict(meta_te)
        print(f"  fold{fold+1} done", flush=True)

    return oof_single, oof_stack


def pre_debug(X, y):
    """验证3基模型独立R² + 多样性。"""
    print("\n[pre-debug] 基模型独立性能 + 多样性:", flush=True)
    factories = get_base_models(42)
    cv = KFold(5, shuffle=True, random_state=42)
    n = len(y)
    preds = {name: np.full(n, np.nan) for name in factories}
    for tr, te in cv.split(X):
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        for name, fac in factories.items():
            m = fac().fit(Xtr, y[tr])
            preds[name][te] = m.predict(Xte)
    for name in factories:
        print(f"  {name:10s} R²={r2_score(y, preds[name]):.4f}", flush=True)
    # 多样性: 预测间相关性 (<0.99=有多样性, stacking才有效)
    names = list(factories.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            r = np.corrcoef(preds[names[i]], preds[names[j]])[0,1]
            print(f"  corr({names[i]},{names[j]})={r:.3f} {'✓多样' if r<0.99 else '⚠冗余'}", flush=True)
    return preds


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    X = df[feat].values

    print("=" * 64, flush=True)
    print("  GBDT Stacking (LightGBM+XGBoost+HistGBT)", flush=True)
    print("=" * 64, flush=True)

    results = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        y = df[target].values
        print(f"\n### {target} ###", flush=True)
        # pre-debug (仅形成能, 快速)
        if target == "formation_energy_per_atom":
            pre_debug(X, y)
        # stacking
        oof_single, oof_stack = stacking_cv(X, y, target)
        # post-debug
        print(f"\n[结果] {target}:", flush=True)
        r2_best_single = max(r2_score(y, oof_single[n]) for n in oof_single)
        r2_stack = r2_score(y, oof_stack)
        for name in oof_single:
            print(f"  {name:10s} R²={r2_score(y, oof_single[name]):.4f}", flush=True)
        print(f"  Stacking   R²={r2_stack:.4f}", flush=True)
        print(f"  最佳单模型 R²={r2_best_single:.4f}", flush=True)
        print(f"  Stacking提升: {r2_stack-r2_best_single:+.4f}", flush=True)
        if r2_stack > r2_best_single:
            print(f"  ✓ stacking优于最佳单模型", flush=True)
        else:
            print(f"  ⚠ stacking未超过最佳单模型", flush=True)
        results.append({"target": target,
                        **{f"r2_{n}": r2_score(y, oof_single[n]) for n in oof_single},
                        "r2_stacking": r2_stack,
                        "improvement_vs_best": r2_stack - r2_best_single})

    pd.DataFrame(results).to_csv(METRICS_DIR / "stacking_results.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] stacking_results.csv", flush=True)


if __name__ == "__main__":
    main()
