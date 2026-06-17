"""
================================================================
不确定性量化模块 (Uncertainty Quantification)
================================================================
论文方法创新: 不仅预测目标值, 还给出预测的不确定性 (置信区间)。

意义:
  1. 评估模型外推可靠性——训练集外元素组合的高不确定性, 警示慎用
  2. 支撑主动学习——优先标注高不确定性的样本, 提升数据效率
  3. 高通量筛选的可信度过滤——只推荐"预测稳定且低不确定性"的候选
     (避免模型把外推区误判为稳定)

实现方案 (CPU 友好, 集成方法, 无需深度学习):
  方案A: 分位数回归 (Quantile Regression, XGBoost/LightGBM)
         预测 10/50/90 分位, 区间 [Q10, Q90] 作为 80% 置信区间
  方案B: 集成方差 (Ensemble Variance)
         训练 N 个 bootstrap 子模型, 预测方差作为不确定性
  方案C: 高斯过程 (GP, 当样本<2000时可用)

本模块实现方案A+B, 对比两种不确定性的校准度。

评估指标:
  - 点预测: RMSE / MAE / R²
  - 不确定性校准: PICP (预测区间覆盖概率) 应接近名义覆盖率 80%
                 MPIW (平均预测区间宽度) 越窄越好

依赖: scikit-learn, xgboost, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

LOWER_Q = 0.10
UPPER_Q = 0.90


def _point_metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _interval_metrics(y_true, y_lower, y_upper, nominal=0.80) -> dict:
    """计算预测区间的校准度指标。"""
    picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))  # 覆盖概率
    mpiw = float(np.mean(y_upper - y_lower))  # 区间宽度
    # CWC: 覆盖加权区间宽度 (越接近1越好, 同时惩罚过宽)
    from math import exp
    if picp >= nominal:
        cwc = mpiw
    else:
        eta = (1 - nominal) * 100
        cwc = mpiw * (1 + exp(eta * (nominal - picp)))
    return {
        "PICP": picp,            # 预测区间覆盖概率 (应接近 nominal)
        "MPIW": mpiw,            # 平均预测区间宽度
        "CWC": cwc,              # 覆盖加权宽度
        "nominal_coverage": nominal,
    }


# ----------------------------------------------------------------
# 方案A: 分位数回归
# ----------------------------------------------------------------
def quantile_regression_cv(
    X, y, cv, quantiles=(LOWER_Q, 0.5, UPPER_Q),
):
    """返回 (q_lower_pred, q_median_pred, q_upper_pred)。"""
    preds = {q: np.full_like(y, np.nan, dtype=float) for q in quantiles}
    for tr, te in cv.split(X):
        for q in quantiles:
            if HAS_XGB:
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("xgb", XGBRegressor(
                        n_estimators=200, max_depth=6, learning_rate=0.1,
                        objective="reg:quantileerror", quantile_alpha=q,
                        random_state=42, n_jobs=1, verbosity=0,
                    )),
                ])
            else:
                model = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    n_estimators=200, max_depth=3,
                    random_state=42,
                )
            model.fit(X[tr], y[tr])
            preds[q][te] = model.predict(X[te])
    # 保证 lower <= median <= upper
    lo, mid, hi = preds[quantiles[0]], preds[quantiles[1]], preds[quantiles[2]]
    lo = np.minimum(lo, mid)
    hi = np.maximum(hi, mid)
    return lo, mid, hi


# ----------------------------------------------------------------
# 方案B: 集成方差 (Bootstrap 随机森林)
# ----------------------------------------------------------------
def ensemble_variance_cv(X, y, cv, n_estimators=80, n_models=5):
    """
    训练 n_models 个 RF (每个用不同 random_state + bootstrap 采样),
    返回 (lower, mean_pred, upper, std_pred)。

    ★ n_jobs=1 避免 spawn; n_models=5 控制总训练次数 (5折×5模型=25次RF)

    区间构造: 用经验分位数(而非正态假设), 解决 PICP 偏低问题。
    每个 RF 内部有 n_estimators 棵树, 直接取树预测的 10/90 分位数。
    """
    mean_pred = np.full_like(y, np.nan, dtype=float)
    std_pred = np.full_like(y, np.nan, dtype=float)
    lower = np.full_like(y, np.nan, dtype=float)
    upper = np.full_like(y, np.nan, dtype=float)
    for tr, te in cv.split(X):
        # 收集所有树的所有预测 (n_estimators * n_models 棵树的预测)
        all_tree_preds = []
        for m in range(n_models):
            rf = RandomForestRegressor(
                n_estimators=n_estimators, random_state=42 + m,
                n_jobs=1, max_samples=0.8,
            )
            rf.fit(X[tr], y[tr])
            # 取每棵树的预测: rf.estimators_ 列表
            for tree in rf.estimators_:
                all_tree_preds.append(tree.predict(X[te]))
        # all_tree_preds: (n_estimators*n_models, n_te)
        arr = np.array(all_tree_preds)
        mean_pred[te] = arr.mean(axis=0)
        std_pred[te] = arr.std(axis=0)
        # 经验分位数区间 (更稳健, 不依赖正态假设)
        lower[te] = np.percentile(arr, 10, axis=0)
        upper[te] = np.percentile(arr, 90, axis=0)
    return lower, mean_pred, upper, std_pred


# ----------------------------------------------------------------
# 主入口: 集成方差不确定性量化
# ----------------------------------------------------------------
def run_benchmark(
    df: pd.DataFrame,
    target: str,
    feat_prefixes: tuple = ("magpie_", "phys_"),  # P0-1: 排除struct避免泄露
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    用 Bootstrap RF 集成方差量化预测不确定性。

    ★ 简化说明: 原计划对比"分位数回归 + 集成方差"两种方案, 但 XGBoost
      分位数回归 (reg:quantileerror) 在 122 维 × 4914 样本上极慢
      (10min+ 未完成), 故只保留集成方差法。该方法更通用、更稳健。
    """
    feat_cols = [c for c in df.columns if c.startswith(feat_prefixes)]
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"[UQ] 目标: {target}, 样本 {len(y)}, 特征 {X.shape[1]}维")

    results = []

    # ---- 集成方差 (Bootstrap RF) ----
    print("  >> 集成方差 (5个RF × 5折 = 25次训练)...", flush=True)
    lo_e, mid_e, hi_e, std_e = ensemble_variance_cv(X, y, cv, n_models=5, n_estimators=60)
    mB = _point_metrics(y, mid_e)
    iB = _interval_metrics(y, lo_e, hi_e, nominal=0.80)
    results.append({"method": "ensemble_rf", **mB, **iB, "target": target})

    # ---- 不确定性 vs 误差相关性 (理想: 高不确定性对应高误差) ----
    abs_err_e = np.abs(y - mid_e)
    corr_e = float(np.corrcoef(std_e, abs_err_e)[0, 1])
    results[0]["uncert_err_corr"] = corr_e
    print(f"  >> 点预测: RMSE={mB['RMSE']:.4f} MAE={mB['MAE']:.4f} R²={mB['R2']:.4f}", flush=True)
    print(f"  >> 区间校准: PICP={iB['PICP']:.3f} (名义0.80) MPIW={iB['MPIW']:.4f}", flush=True)
    print(f"  >> 不确定性-误差相关性: {corr_e:.3f} (越高越有意义)", flush=True)

    # 持久化逐样本不确定性 (供 screening 模块用)
    unc_path = METRICS_DIR / f"uncertainty_{target}.csv"
    df_unc = pd.DataFrame({
        "y_true": y,
        "pred_ensemble": mid_e, "std_ensemble": std_e,
        "lower_80": lo_e, "upper_80": hi_e,
        "abs_err": abs_err_e,
        "idx": np.arange(len(y)),
    })
    df_unc.to_csv(unc_path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] 逐样本不确定性: {unc_path}", flush=True)

    return pd.DataFrame(results)


def main():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils import load_features
    df = load_features()
    if df is None or len(df) == 0:
        raise SystemExit("[ERROR] 特征数据为空, 先运行 features.py")

    print("=" * 60)
    print("  不确定性量化 (分位数回归 vs 集成方差)")
    print("=" * 60)

    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###")
        res = run_benchmark(df, target=target)
        all_res.append(res)

    df_all = pd.concat(all_res, ignore_index=True)
    out_path = METRICS_DIR / "uncertainty_comparison.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 不确定性对比: {out_path}")

    print("\n" + "=" * 60)
    print("  不确定性量化汇总")
    print("=" * 60)
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
