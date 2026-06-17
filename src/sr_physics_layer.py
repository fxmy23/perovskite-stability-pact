"""
================================================================
符号回归物理层 — CV 评估 (Step 2)
================================================================
把 PACT 物理层从 KernelRidge 替换为符号回归发现的显式方程。

★ 创新点 (论文方法章主线):
  - SymbolicPhysicsLayer: gplearn 发现的解析公式作为物理锚点
  - 与 KernelRidge 物理层对照 (诚实报告 SR 通常 R² 略低但可解释)
  - CV 内每折独立 fit SR (无泄露)
  - 输出: SR 方程字符串 + SR物理层R² + (SR+ML残差)总R²

依赖: gplearn, scikit-learn, lightgbm, numpy, pandas

注: gplearn 0.4.3 的 SymbolicRegressor 在 gplearn.genetic (非 .symbolic)。

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from gplearn.genetic import SymbolicRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# 默认 SR 配置 (Step 1 探索选定):
#   basic_sqrt_p0.001 得 R²=0.42 (形成能全量), rich_p0.001 得 0.41。
#   选 basic_sqrt: sqrt 有物理依据 (容差因子含 √2, 半径比非线性),
#   且公式比 rich 更紧凑。parsimony=0.001 让公式充分生长。
DEFAULT_FUNCTION_SET = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1}
DEFAULT_PARSIMONY = 0.001


def make_sr_model(function_set=None, parsimony=DEFAULT_PARSIMONY, seed=42,
                  population_size=2000, generations=40):
    """构造 gplearn SymbolicRegressor (统一配置入口)。

    ★ generations=40 (探索用 30, CV 评估用 40 给 SR 更充分搜索)。
    stopping_criteria=0.0 (不提前停, 让 SR 充分进化)。
    """
    return SymbolicRegressor(
        function_set=function_set or DEFAULT_FUNCTION_SET,
        population_size=population_size,
        generations=generations,
        tournament_size=20,
        parsimony_coefficient=parsimony,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        metric="mean absolute error",
        stopping_criteria=0.0,  # 不提前停, 充分搜索
        n_jobs=1,
        random_state=seed,
        verbose=0,
    )


def evaluate_sr_vs_krr_cv(df, target, function_set=None, parsimony=DEFAULT_PARSIMONY,
                          n_splits=5, n_ml_models=5, seed=42):
    """
    5 折 CV 对比: SR物理层+ML残差 vs KernelRidge物理层+ML残差 vs 纯ML。
    每折内独立 fit SR (无泄露)。

    返回 dict: 各方法的 OOF R²/MAE + SR 发现的公式 (每折一个, 看一致性)。
    """
    feat_cols = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)

    oof_sr = np.full(n, np.nan)          # SR物理层 单独
    oof_sr_ml = np.full(n, np.nan)       # SR物理层 + ML残差
    oof_krr = np.full(n, np.nan)         # KRR物理层 单独
    oof_krr_ml = np.full(n, np.nan)      # KRR物理层 + ML残差 (主线 PACT)
    oof_ml = np.full(n, np.nan)          # 纯 ML
    sr_equations = []

    print(f"[SR-CV] {target}, n={n}, 物理层 SR vs KRR, {n_splits}折", flush=True)

    for fold, (tr, te) in enumerate(cv.split(X)):
        # ---- 预处理: 物理层 (impute+scale 在折内 fit) ----
        imp_p = SimpleImputer(strategy="median"); sca_p = StandardScaler()
        Xtr_p = sca_p.fit_transform(imp_p.fit_transform(X[tr][:, phys_idx]))
        Xte_p = sca_p.transform(imp_p.transform(X[te][:, phys_idx]))
        # ---- 预处理: ML 层 (全特征) ----
        imp_a = SimpleImputer(strategy="median"); sca_a = StandardScaler()
        Xtr_a = sca_a.fit_transform(imp_a.fit_transform(X[tr]))
        Xte_a = sca_a.transform(imp_a.transform(X[te]))

        # ===== SR 物理层 (每折独立 fit, 无泄露) =====
        sr = make_sr_model(function_set, parsimony, seed=seed + fold)
        sr.fit(Xtr_p, y[tr])
        sr_eq = str(sr._program)
        sr_equations.append(sr_eq)
        sr_pred_tr = sr.predict(Xtr_p)
        sr_pred_te = sr.predict(Xte_p)
        oof_sr[te] = sr_pred_te
        # SR + ML 残差
        resid_sr = y[tr] - sr_pred_tr
        ml_preds_sr = []
        for m in range(n_ml_models):
            lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42 + m, n_jobs=1, verbose=-1,
                                subsample=0.8, subsample_freq=1)
            lgb.fit(Xtr_a, resid_sr)
            ml_preds_sr.append(lgb.predict(Xte_a))
        oof_sr_ml[te] = sr_pred_te + np.mean(ml_preds_sr, axis=0)

        # ===== KRR 物理层 (对照) =====
        krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
        krr.fit(Xtr_p, y[tr])
        krr_pred_tr = krr.predict(Xtr_p)
        krr_pred_te = krr.predict(Xte_p)
        oof_krr[te] = krr_pred_te
        resid_krr = y[tr] - krr_pred_tr
        ml_preds_krr = []
        for m in range(n_ml_models):
            lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42 + m, n_jobs=1, verbose=-1,
                                subsample=0.8, subsample_freq=1)
            lgb.fit(Xtr_a, resid_krr)
            ml_preds_krr.append(lgb.predict(Xte_a))
        oof_krr_ml[te] = krr_pred_te + np.mean(ml_preds_krr, axis=0)

        # ===== 纯 ML (无物理层) =====
        ml_pure = []
        for m in range(n_ml_models):
            lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42 + m, n_jobs=1, verbose=-1,
                                subsample=0.8, subsample_freq=1)
            lgb.fit(Xtr_a, y[tr])
            ml_pure.append(lgb.predict(Xte_a))
        oof_ml[te] = np.mean(ml_pure, axis=0)

        if (fold + 1) % 1 == 0:
            print(f"  fold {fold+1}/{n_splits} done | SR公式: {sr_eq[:60]}",
                  flush=True)

    # ---- 汇总指标 ----
    def m(p):
        return {"R2": float(r2_score(y, p)),
                "MAE": float(mean_absolute_error(y, p)),
                "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

    results = {
        "target": target,
        "SR_physics_only": m(oof_sr),
        "SR_physics_plus_ML": m(oof_sr_ml),
        "KRR_physics_only": m(oof_krr),
        "KRR_physics_plus_ML": m(oof_krr_ml),
        "pure_ML": m(oof_ml),
    }
    print(f"\n  [结果] {target}:", flush=True)
    print(f"    SR物理层单独:     R²={results['SR_physics_only']['R2']:.4f}", flush=True)
    print(f"    SR物理层+ML残差:  R²={results['SR_physics_plus_ML']['R2']:.4f}", flush=True)
    print(f"    KRR物理层单独:    R²={results['KRR_physics_only']['R2']:.4f}", flush=True)
    print(f"    KRR物理层+ML残差: R²={results['KRR_physics_plus_ML']['R2']:.4f}", flush=True)
    print(f"    纯ML:             R²={results['pure_ML']['R2']:.4f}", flush=True)
    print(f"\n  各折 SR 公式 (一致性检查):", flush=True)
    for i, eq in enumerate(sr_equations):
        print(f"    fold{i+1}: {eq[:90]}", flush=True)

    return results, sr_equations, {
        "y": y, "oof_sr": oof_sr, "oof_sr_ml": oof_sr_ml,
        "oof_krr": oof_krr, "oof_krr_ml": oof_krr_ml, "oof_ml": oof_ml,
    }


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  符号回归物理层 CV 评估 (SR vs KRR vs 纯ML)", flush=True)
    print("=" * 64, flush=True)

    all_results = []
    all_eqs = {}
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        res, eqs, _ = evaluate_sr_vs_krr_cv(df, target)
        all_results.append(res)
        all_eqs[target] = eqs

    # 持久化
    flat = []
    for r in all_results:
        for method, metrics in r.items():
            if method == "target":
                continue
            row = {"target": r["target"], "method": method, **metrics}
            flat.append(row)
    df_out = pd.DataFrame(flat)
    out = METRICS_DIR / "sr_vs_krr_cv.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)

    # 公式持久化
    eq_rows = []
    for t, eqs in all_eqs.items():
        for i, eq in enumerate(eqs):
            eq_rows.append({"target": t, "fold": i + 1, "equation": eq})
    pd.DataFrame(eq_rows).to_csv(METRICS_DIR / "sr_equations_per_fold.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"[SAVE] sr_equations_per_fold.csv", flush=True)


if __name__ == "__main__":
    main()
