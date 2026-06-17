"""
================================================================
迁移学习模块 (Transfer Learning via Feature Pretraining)
================================================================
论文 P1-1 (修正版): 原"多任务 Ridge"在数学上与单任务等价, 无法体现优势。
改用"特征预训练迁移"叙事, 这是更站得住脚的迁移学习方案。

核心思想:
  形成能 Ef (R²=0.91, 易预测) 和 凸包能 Ehull (R²=0.80, 难, 稳定仅9.4%)
  共享底层物理 (化学键合、离子匹配)。
  问题: 能否用 Ef 的预测作为 Ehull 的额外特征, 提升难目标的性能?

实验设计:
  Baseline: 直接用原始特征预测 Ehull
  Transfer: 先训练 Ef 模型 → 用 Ef 预测值作为额外特征 → 再预测 Ehull
  对比两者在不同训练集大小下的 R², 验证迁移增益。

这种"知识迁移"比多任务 Ridge 更有物理意义:
  - Ef 模型学到了"键合强度"的表示
  - 把这个表示喂给 Ehull 模型, 等于传递了"热力学背景知识"

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# 迁移方向: 用易目标(Ef)辅助难目标(Ehull)
SOURCE = "formation_energy_per_atom"   # 源任务 (易)
TARGET = "energy_above_hull"           # 目标任务 (难)
TRAIN_SIZES = [200, 500, 1000, 2000, 4914]


def _make_model():
    if HAS_LGBM:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def run_benchmark(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    迁移学习实验: 用 Ef 预测值作为 Ehull 的额外特征。
    """
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y_source = df[SOURCE].values
    y_target = df[TARGET].values
    n_total = len(df)

    print(f"[TRANSFER] 源任务: {SOURCE} (易)", flush=True)
    print(f"           目标任务: {TARGET} (难)", flush=True)
    print(f"           学习曲线: {TRAIN_SIZES}", flush=True)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for size in TRAIN_SIZES:
        actual_size = min(size, n_total)
        print(f"\n  >> 训练集大小 = {actual_size}", flush=True)

        baseline_r2s, transfer_r2s = [], []

        for fold_id, (all_idx, test_idx) in enumerate(cv.split(X)):
            np.random.seed(42 + fold_id)
            if actual_size < len(all_idx):
                sampled = np.random.choice(all_idx, actual_size, replace=False)
            else:
                sampled = all_idx

            X_tr, X_te = X[sampled], X[test_idx]
            ys_tr, ys_te = y_source[sampled], y_source[test_idx]
            yt_tr, yt_te = y_target[sampled], y_target[test_idx]

            # ---- Baseline: 直接预测 Ehull ----
            m_base = _make_model()
            m_base.fit(X_tr, yt_tr)
            pred_base = m_base.predict(X_te)

            # ---- Transfer: 先训练 Ef 模型, 预测值作为额外特征 ----
            m_src = _make_model()
            m_src.fit(X_tr, ys_tr)
            # 用源模型生成"知识特征" (Ef 预测值)
            src_feat_tr = m_src.predict(X_tr).reshape(-1, 1)
            src_feat_te = m_src.predict(X_te).reshape(-1, 1)
            # 拼接: 原始特征 + Ef 预测特征
            X_tr_aug = np.hstack([X_tr, src_feat_tr])
            X_te_aug = np.hstack([X_te, src_feat_te])
            m_tgt = _make_model()
            m_tgt.fit(X_tr_aug, yt_tr)
            pred_transfer = m_tgt.predict(X_te_aug)

            baseline_r2s.append(r2_score(yt_te, pred_base))
            transfer_r2s.append(r2_score(yt_te, pred_transfer))

        b_r2 = float(np.mean(baseline_r2s))
        t_r2 = float(np.mean(transfer_r2s))
        gain = t_r2 - b_r2

        results.append({
            "train_size": actual_size,
            "baseline_R2": b_r2,
            "transfer_R2": t_r2,
            "gain": gain,
            "gain_pct": gain / (abs(b_r2) + 1e-9) * 100,
        })
        print(f"    Baseline R²={b_r2:.4f} | Transfer R²={t_r2:.4f} | "
              f"增益={gain:+.4f} ({gain/(abs(b_r2)+1e-9)*100:+.1f}%)", flush=True)

    return pd.DataFrame(results)


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  迁移学习 (Ef知识 → Ehull预测)", flush=True)
    print("=" * 60, flush=True)

    df_out = run_benchmark(df)
    out_path = METRICS_DIR / "multitask_comparison.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 迁移学习结果: {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  迁移增益汇总", flush=True)
    print("=" * 60, flush=True)
    print(df_out[["train_size", "baseline_R2", "transfer_R2", "gain", "gain_pct"]].to_string(index=False), flush=True)

    best = df_out.loc[df_out["gain"].idxmax()]
    print(f"\n  最大增益: train_size={best['train_size']}, "
          f"gain={best['gain']:+.4f} ({best['gain_pct']:+.1f}%)", flush=True)


if __name__ == "__main__":
    main()
