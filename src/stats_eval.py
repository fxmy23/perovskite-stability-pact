"""
================================================================
统计严谨性评估模块 (Statistical Rigor)
================================================================
补齐审稿人要求的统计指标, 把"统计中等"升级为"统计严谨":

  1. Wilcoxon signed-rank 检验 — 5-seed 配对显著性, 报 p 值
  2. AUC-PR / Average Precision — 不平衡分类必备 (ROC 的补充)
  3. Enrichment Factor (EF) — top-k 富集因子, DAF 的变体
  4. Bootstrap 95% CI — R²/MAE 的 [lo, hi] 区间
  5. MRE / Max error — 补全回归指标
  6. Spearman/Pearson σ-误差相关性 + p 值

依赖: scipy, scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    average_precision_score, precision_score, recall_score, f1_score,
)


# ----------------------------------------------------------------
# 回归指标全集
# ----------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    """R²/RMSE/MAE/MRE/MaxErr/95pct-err 全套。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    # MRE: 平均相对误差, 避免除零
    denom = np.abs(y_true).copy()
    denom[denom < 1e-6] = 1e-6
    mre = float(np.mean(abs_err / denom))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MRE": mre,
        "Max_error": float(abs_err.max()),
        "p95_error": float(np.percentile(abs_err, 95)),
    }


# ----------------------------------------------------------------
# 分类指标全集 (含 AUC-PR, EF)
# ----------------------------------------------------------------
def classification_metrics(y_true, y_score_or_pred, thresh=0.05,
                           is_score=False, top_frac=0.1) -> dict:
    """
    y_true: 真实稳定标签 (0/1) 或连续 E_hull (会内部阈值化)
    y_score_or_pred: 若 is_score=True 视为"稳定度分数"(越大越稳定, 如 -μ);
                     否则视为已阈值化的 0/1 预测。
    """
    y_true = np.asarray(y_true, dtype=float)
    # 若传的是连续 E_hull, 阈值化
    if set(np.unique(y_true)).issubset({0.0, 1.0}):
        y_bin = y_true.astype(int)
    else:
        y_bin = (y_true < thresh).astype(int)

    if is_score:
        score = np.asarray(y_score_or_pred, dtype=float)
        pred_bin = (score > np.median(score)).astype(int)  # 仅用于 P/R/F1
    else:
        pred_bin = np.asarray(y_score_or_pred, dtype=int)

    out = {
        "Precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
        "Recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
        "F1": float(f1_score(y_bin, pred_bin, zero_division=0)),
    }
    # AUC-PR: 用 score = -μ (低 μ = 高稳定度)
    if is_score:
        score = np.asarray(y_score_or_pred, dtype=float)
    else:
        score = -np.asarray(y_score_or_pred, dtype=float)  # 退化
    out["AUC_PR"] = float(average_precision_score(y_bin, score))
    out["EnrichmentFactor"] = enrichment_factor(y_bin, score, top_frac=top_frac)
    return out


def enrichment_factor(y_bin, score, top_frac=0.1):
    """EF_top = (top-k 内阳性比例) / (总阳性比例)。"""
    y_bin = np.asarray(y_bin)
    score = np.asarray(score)
    base_rate = y_bin.mean()
    if base_rate == 0:
        return float("nan")
    n_top = max(1, int(len(score) * top_frac))
    top_idx = np.argsort(-score)[:n_top]  # score 越大越好
    return float(y_bin[top_idx].mean() / base_rate)


# ----------------------------------------------------------------
# Bootstrap 置信区间
# ----------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, metric_fn=None, n_boot=1000, ci=0.95, seed=42):
    """
    对任意 metric_fn(y_true, y_pred) -> scalar 做 bootstrap CI。
    默认 metric_fn = R²。
    """
    if metric_fn is None:
        metric_fn = lambda a, b: r2_score(a, b)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # 至少要有 2 个不同值才能算 R²
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            boots.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    boots = np.array(boots)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return {
        "point": float(metric_fn(y_true, y_pred)),
        "lo": lo, "hi": hi, "n_boot": len(boots),
    }


# ----------------------------------------------------------------
# Wilcoxon signed-rank: 5-seed 配对显著性
# ----------------------------------------------------------------
def paired_significance(seeds_a: Iterable[float], seeds_b: Iterable[float],
                        alternative="two-sided") -> dict:
    """
    对两个模型在相同 5 个 seed 上的指标做 Wilcoxon signed-rank。
    返回 statistic, p_value, mean_diff。
    """
    a = np.asarray(list(seeds_a), dtype=float)
    b = np.asarray(list(seeds_b), dtype=float)
    diff = a - b
    try:
        stat, p = stats.wilcoxon(a, b, alternative=alternative)
    except ValueError:
        # 全零差值会抛错
        stat, p = float("nan"), 1.0
    return {
        "mean_A": float(a.mean()), "std_A": float(a.std(ddof=1)),
        "mean_B": float(b.mean()), "std_B": float(b.std(ddof=1)),
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }


# ----------------------------------------------------------------
# σ-误差相关性 (带 p 值)
# ----------------------------------------------------------------
def uncertainty_error_correlation(sigma, y_true, y_pred):
    abs_err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    sigma = np.asarray(sigma)
    pear_r, pear_p = stats.pearsonr(sigma, abs_err)
    spear_r, spear_p = stats.spearmanr(sigma, abs_err)
    return {
        "pearson_r": float(pear_r), "pearson_p": float(pear_p),
        "spearman_r": float(spear_r), "spearman_p": float(spear_p),
    }


# ----------------------------------------------------------------
# 主入口: 对一个 (y_true, y_pred, sigma, y_bin, score) 数据集出全报告
# ----------------------------------------------------------------
def full_report(y_true, y_pred, sigma=None, y_bin=None, score=None,
                n_boot=1000, seed=42):
    """一键产出回归+分类+不确定性+CI 全报告 (dict)。"""
    rep = {"regression": regression_metrics(y_true, y_pred)}
    rep["R2_bootstrap_CI"] = bootstrap_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    rep["MAE_bootstrap_CI"] = bootstrap_ci(
        y_true, y_pred, metric_fn=mean_absolute_error, n_boot=n_boot, seed=seed,
    )
    if sigma is not None:
        rep["uncertainty"] = uncertainty_error_correlation(sigma, y_true, y_pred)
    if y_bin is not None and score is not None:
        rep["classification"] = classification_metrics(y_bin, score, is_score=True)
    return rep


if __name__ == "__main__":
    # 自检: 随机数据
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 500)
    pred = y + rng.normal(0, 0.3, 500)
    sigma = np.abs(rng.normal(0.3, 0.1, 500))
    rep = full_report(y, pred, sigma)
    print("regression:", {k: round(v, 4) for k, v in rep["regression"].items()})
    print("R2 CI:", {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in rep["R2_bootstrap_CI"].items()})
    print("uncertainty:", {k: round(v, 4) for k, v in rep["uncertainty"].items()})
    # paired sig
    a = [0.91, 0.90, 0.905, 0.904, 0.911]
    b = [0.886, 0.885, 0.796, 0.797, 0.802]
    print("paired:", paired_significance(a, b))
