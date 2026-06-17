"""
================================================================
优化③: SHAP 特征选择 (降噪提精度)
================================================================
文献支撑: MDPI Materials 2025 选9个关键特征反而 R²=0.928 (高于全特征)。
原理: 110维特征中很多是噪声 (Magpie 的 mode/dev 等对形成能无意义),
  SHAP 重要性可识别真正有用的特征子集, 降低噪声 → 提精度。

方法 (无泄露):
  1. 5折CV, 每折内:
     a. 训练折 fit LightGBM (Optuna 最佳超参)
     b. SHAP 算特征重要性
     c. 选 top-K 特征 (K 扫描 20/40/60/80)
     d. 用 top-K 在训练折 refit, 测试折 predict
  2. 报各 K 的 OOF R², 找最优 K
  3. 汇总跨折的 SHAP 共识特征 (哪些特征稳定进 top-K)

依赖: shap, lightgbm, scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Optuna 搜出的最佳超参 (优化②结果)
OPTUNA_PARAMS = {
    "formation_energy_per_atom": dict(
        n_estimators=450, num_leaves=39, learning_rate=0.078, max_depth=10,
        min_child_samples=17, subsample=0.902, colsample_bytree=0.728,
        reg_alpha=0.489, reg_lambda=0.278),
    "energy_above_hull": dict(
        n_estimators=454, num_leaves=105, learning_rate=0.044, max_depth=11,
        min_child_samples=23, subsample=0.848, colsample_bytree=0.733,
        reg_alpha=0.745, reg_lambda=0.002),
}
K_VALUES = [20, 40, 60, 80]


def shap_feature_selection_cv(X, y, feat_names, target_name, seed=42):
    """5折CV: 每折内 SHAP选特征 + 多K评估。"""
    print(f"\n[SHAP特征选择] {target_name}", flush=True)
    cv = KFold(5, shuffle=True, random_state=seed)
    params = OPTUNA_PARAMS[target_name]
    params_full = {**params, "n_jobs": 1, "verbose": -1, "subsample_freq": 1,
                   "random_state": seed}

    # baseline (全特征 + Optuna超参)
    oof_base = np.full(len(y), np.nan)
    # 各 K 的 OOF
    oof_by_k = {k: np.full(len(y), np.nan) for k in K_VALUES}
    # 跨折 SHAP 重要性累计
    shap_counts = {name: 0 for name in feat_names}

    for fold, (tr, te) in enumerate(cv.split(X)):
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))

        # 全特征 baseline
        m_base = LGBMRegressor(**params_full).fit(Xtr, y[tr])
        oof_base[te] = m_base.predict(Xte)

        # SHAP 重要性 (在该折训练集)
        explainer = shap.TreeExplainer(m_base)
        sv = explainer.shap_values(Xtr)
        # sv shape (n_tr, n_feat), 取均值绝对值
        imp_shap = np.mean(np.abs(sv), axis=0)
        # 排序
        order = np.argsort(-imp_shap)

        for k in K_VALUES:
            top_k_idx = order[:k]
            m_k = LGBMRegressor(**params_full).fit(Xtr[:, top_k_idx], y[tr])
            oof_by_k[k][te] = m_k.predict(Xte[:, top_k_idx])

        # 累计 top-40 共识
        for idx in order[:40]:
            shap_counts[feat_names[idx]] += 1

        print(f"  fold{fold+1} done", flush=True)

    # 汇总
    r2_base = r2_score(y, oof_base)
    print(f"\n  全特征(110) R²={r2_base:.4f}", flush=True)
    best_k, best_r2 = None, r2_base
    for k in K_VALUES:
        r2_k = r2_score(y, oof_by_k[k])
        mark = " ★" if r2_k > best_r2 else ""
        print(f"  top-{k:3d} 特征 R²={r2_k:.4f} ({r2_k-r2_base:+.4f}){mark}", flush=True)
        if r2_k > best_r2:
            best_k, best_r2 = k, r2_k

    return {
        "target": target_name,
        "r2_full": r2_base,
        "r2_by_k": {k: r2_score(y, oof_by_k[k]) for k in K_VALUES},
        "best_k": best_k, "best_r2": best_r2,
        "shap_consensus": shap_counts,
    }


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    X = df[feat].values
    print("=" * 64, flush=True)
    print("  优化③: SHAP 特征选择 (配合 Optuna 超参)", flush=True)
    print("=" * 64, flush=True)

    all_res = []
    consensus_all = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        y = df[target].values
        res = shap_feature_selection_cv(X, y, feat, target)
        all_res.append(res)
        # 共识特征 top-20
        consensus = sorted(res["shap_consensus"].items(), key=lambda x: -x[1])[:20]
        for name, count in consensus:
            consensus_all.append({"target": target, "feature": name,
                                   "appearances_in_top40": count})
        print(f"\n  {target} top-20 SHAP共识特征:", flush=True)
        for name, c in consensus[:10]:
            print(f"    {name:40s} {c}/5 折", flush=True)

    # 持久化
    summary = []
    for r in all_res:
        summary.append({"target": r["target"], "r2_full": r["r2_full"],
                        "r2_top20": r["r2_by_k"][20], "r2_top40": r["r2_by_k"][40],
                        "r2_top60": r["r2_by_k"][60], "r2_top80": r["r2_by_k"][80],
                        "best_k": r["best_k"], "best_r2": r["best_r2"]})
    pd.DataFrame(summary).to_csv(METRICS_DIR / "shap_selection_results.csv",
                                  index=False, encoding="utf-8-sig")
    pd.DataFrame(consensus_all).to_csv(METRICS_DIR / "shap_consensus_features.csv",
                                        index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] shap_selection_results.csv, shap_consensus_features.csv", flush=True)


if __name__ == "__main__":
    main()
