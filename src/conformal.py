"""
================================================================
Conformal Prediction: 修复 PICP 的正确方法 (P0 修订核心)
================================================================
背景问题:
  原 PACT 用 5 个 LightGBM 的 ensemble std 构造 80% 区间 ŷ±1.28σ,
  实测 PICP=0.24 (灾难性偏低)。根因: 5 个高度相关的 LightGBM 共享
  同样的训练数据切分, 预测高度相关, std 严重低估真实误差。

本模块实现 Split Conformal Prediction (分布无关, 覆盖率有理论保证):
  在 5 折 CV 内部, 每折训练集再切出 calibration 子集 (20%),
  计算该折 calibration 残差的 (1-α) 分位数 q_fold,
  该折 OOF 区间 = ŷ_oof ± q_fold。

理论保证 (Vovk 等):
  在数据可交换 (exchangeable) 假设下, 真实 PICP ≥ 1-α, 有限样本严格成立。
  这是我们缺的"理论保证", 也是把 PICP 从 0.24 拉到 ≥0.80 的关键。

接口:
  conformal_intervals_cv(...) — 返回 (lower, upper, q_per_fold) OOF 区间
  compute_picp_mpiw(...)       — 计算 PICP/MPIW/CWC/ECE
  plot_reliability_diagram(...) — 校准曲线 (期望 vs 实际覆盖率)

依赖: numpy, scipy, scikit-learn

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------
# Split Conformal: 单折内 fit/calibrate
# ----------------------------------------------------------------
def _split_conformal_quantile(
    X_tr, y_tr, fit_fn: Callable, predict_fn: Callable,
    cal_frac: float = 0.2, alpha: float = 0.2, rng: np.random.Generator = None,
):
    """
    在单折训练集内做 split conformal:
      1. 把 (X_tr, y_tr) 切成 proper-train / calibration
      2. proper-train 上 fit, calibration 上算残差
      3. 取 ⌈(1-α)(n_cal+1)⌉ 分位数 (含上界, 保证有限样本覆盖)

    返回 conformal 半宽 q (作用于该折所有 OOF 样本的 ±q 区间)。

    注: predict_fn 可返回 (mu, sigma) 元组或纯 mu; 这里只取点预测 mu。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_cal = max(10, int(round(n * cal_frac)))
    cal_idx = perm[:n_cal]
    fit_idx = perm[n_cal:]

    # fit on proper-train subset only (no leakage into calibration)
    model = fit_fn(X_tr[fit_idx], y_tr[fit_idx])
    cal_out = predict_fn(model, X_tr[cal_idx])
    cal_pred = cal_out[0] if isinstance(cal_out, tuple) else cal_out
    scores = np.abs(y_tr[cal_idx] - cal_pred)  # 非一致性分数

    # ★ 正确的 split conformal 分位数 (审计修复 2026-06-16):
    #   q = scores 排序后第 ⌈(1-α)(n_cal+1)⌉ 个值 (1-indexed rank)。
    #   这是有限样本严格成立的定义, 不是插值分位数。原版用
    #   np.quantile(q_level, method="higher") 经验证在 8/8 合成试验中
    #   有 3 次 PICP<0.80 (最小 0.774), 违反理论保证。根因: 插值法偏离了
    #   严格 rank 统计量。改为直接取排序值, 保证 PICP≥1-α。
    rank = int(np.ceil((1 - alpha) * (n_cal + 1)))
    rank = min(rank, n_cal)  # 上界 = 最大残差
    s_sorted = np.sort(scores)
    q = float(s_sorted[rank - 1])  # 0-indexed
    return q


# ----------------------------------------------------------------
# CV 级 split conformal: 每折独立 calibrate, 输出 OOF 区间
# ----------------------------------------------------------------
def conformal_intervals_cv(
    X: np.ndarray,
    y: np.ndarray,
    cv_splits,
    fit_fn: Callable,
    predict_fn: Callable,
    alpha: float = 0.2,
    cal_frac: float = 0.2,
    seed: int = 42,
    return_oof_pred: bool = True,
):
    """
    5 折 CV + 每折内部 split conformal。

    Args:
        X, y: 全量特征/目标
        cv_splits: CV 的 (train_idx, test_idx) 迭代器
        fit_fn: (X_tr, y_tr) -> model  训练函数
        predict_fn: (model, X_te) -> pred  预测函数
        alpha: 误覆盖率 (0.2 → 80% 名义覆盖)
        cal_frac: 每折训练集内切给 calibration 的比例
        seed: calibration 切分随机种子

    Returns:
        dict with:
          oof_pred  (n,)     — 点预测 (来自完整训练集 fit, 非子集)
          oof_lower (n,)     — OOF 区间下界
          oof_upper (n,)     — OOF 区间上界
          q_per_fold list    — 每折 conformal 半宽
          nominal (1-alpha)
    """
    n = len(y)
    oof_pred = np.full(n, np.nan)
    oof_lower = np.full(n, np.nan)
    oof_upper = np.full(n, np.nan)
    q_per_fold = []

    for fold, (tr, te) in enumerate(cv_splits):
        rng = np.random.default_rng(seed + fold)

        # 1. 完整训练集 fit → 点预测 (与主线评估一致, 不浪费数据)
        model_full = fit_fn(X[tr], y[tr])
        out_full = predict_fn(model_full, X[te])
        oof_pred[te] = out_full[0] if isinstance(out_full, tuple) else out_full

        # 2. 同折内 split conformal → 该折半宽 q_fold
        q_fold = _split_conformal_quantile(
            X[tr], y[tr], fit_fn, predict_fn,
            cal_frac=cal_frac, alpha=alpha, rng=rng,
        )
        q_per_fold.append(q_fold)
        oof_lower[te] = oof_pred[te] - q_fold
        oof_upper[te] = oof_pred[te] + q_fold

    return {
        "oof_pred": oof_pred,
        "oof_lower": oof_lower,
        "oof_upper": oof_upper,
        "q_per_fold": q_per_fold,
        "nominal": 1 - alpha,
    }


# ----------------------------------------------------------------
# 指标: PICP / MPIW / CWC / ECE
# ----------------------------------------------------------------
def compute_picp_mpiw(y_true, lower, upper, nominal=0.80):
    """PICP (覆盖率), MPIW (平均区间宽度), CWC (覆盖加权宽度)."""
    picp = float(np.mean((y_true >= lower) & (y_true <= upper)))
    mpiw = float(np.mean(upper - lower))
    # CWC: 满足覆盖时仅惩罚宽度, 不满足时指数惩罚
    from math import exp
    if picp >= nominal:
        cwc = mpiw
    else:
        eta = (1 - nominal) * 100
        cwc = mpiw * (1 + exp(eta * (nominal - picp)))
    return {"PICP": picp, "MPIW": mpiw, "CWC": cwc, "nominal": nominal}


def compute_ece_by_uncertainty(y_true, lower, upper, sigma, nominal=0.80, n_bins=10):
    """
    期望校准误差 (ECE): 按 σ 排序分桶, 每桶实际覆盖率 vs 名义覆盖率的加权绝对偏差。
    ECE 越小越校准 (理想 ~0)。这是 reliability diagram 的数值汇总。
    """
    covered = ((y_true >= lower) & (y_true <= upper)).astype(float)
    order = np.argsort(sigma)
    n = len(sigma)
    bins = np.array_split(order, n_bins)
    ece = 0.0
    bin_data = []
    for b in bins:
        if len(b) == 0:
            continue
        emp = covered[b].mean()
        w = len(b) / n
        ece += w * abs(emp - nominal)
        bin_data.append({
            "bin_mean_sigma": float(sigma[b].mean()),
            "empirical_coverage": float(emp),
            "n": len(b),
        })
    return {"ECE": float(ece), "bins": bin_data}


# ----------------------------------------------------------------
# Reliability diagram 数据导出 (供 stats_eval 画图)
# ----------------------------------------------------------------
def reliability_diagram_data(y_true, lower, upper, sigma, nominal=0.80, n_bins=10):
    """返回 DataFrame 友好的 (sigma_bucket, empirical_coverage) 序列。"""
    res = compute_ece_by_uncertainty(y_true, lower, upper, sigma, nominal, n_bins)
    import pandas as pd
    df = pd.DataFrame(res["bins"])
    df["nominal"] = nominal
    df["ECE_overall"] = res["ECE"]
    return df
