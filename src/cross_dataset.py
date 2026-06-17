"""
================================================================
跨数据集泛化模块 (Cross-Dataset Generalization)
================================================================
论文 P2-1: 在 wolverton_oxides 上训练, 在 matbench_perovskites 上测试,
评估模型跨数据源的泛化能力。

动机:
  wolverton (R²=0.91) 与 matbench (R²=0.23) 的巨大差距提示两个数据集
  化学空间分布不同。跨数据集测试直接回答"模型能否泛化到不同来源的数据"。

关键技术点:
  两个数据集的形成能可能有系统性偏差 (不同 DFT 泛函/设置),
  所以除 R²/MAE 外, 还报告 Spearman 秩相关 (衡量排序一致性, 对偏差鲁棒)。

依赖: matminer, scikit-learn, lightgbm, scipy, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols
from src.features import generate_magpie_features, generate_physical_features

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def _features_from_formulas(formulas):
    """从化学式生成纯预测特征 (magpie + phys)。"""
    mag = generate_magpie_features(formulas)
    phys = generate_physical_features(formulas)
    df = pd.concat([mag.reset_index(drop=True), phys.reset_index(drop=True)], axis=1)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def main():
    print("=" * 60, flush=True)
    print("  跨数据集泛化 (wolverton → matbench)", flush=True)
    print("=" * 60, flush=True)

    # ---- 训练集: wolverton ----
    print("[1] 加载 wolverton 训练集...", flush=True)
    df_tr = load_features()
    feat_cols = get_feature_cols(df_tr, exclude_struct=True)
    X_tr = df_tr[feat_cols].values
    y_tr = df_tr["formation_energy_per_atom"].values
    print(f"    样本 {len(y_tr)}, 特征 {len(feat_cols)}", flush=True)

    # ---- 测试集: matbench ----
    print("[2] 加载 matbench 测试集...", flush=True)
    from matminer.datasets import load_dataset
    df_mb = load_dataset("matbench_perovskites")
    df_mb["formula_pretty"] = df_mb["structure"].apply(
        lambda s: s.composition.reduced_formula)
    # 去重 (matbench 可能有同化学式多结构)
    df_mb = df_mb.drop_duplicates(subset="formula_pretty").reset_index(drop=True)
    print(f"    样本 {len(df_mb)} (去重后)", flush=True)

    print("[3] 生成 matbench 特征...", flush=True)
    mb_feat = _features_from_formulas(df_mb["formula_pretty"].tolist())
    # 对齐特征列 (matbench 可能缺某些列, 用训练集中位数补)
    for c in feat_cols:
        if c not in mb_feat.columns:
            mb_feat[c] = df_tr[c].median()
    X_te = mb_feat[feat_cols].values
    y_te = df_mb["e_form"].values

    # ---- 训练 + 预测 ----
    print("[4] 训练 wolverton 模型 → 预测 matbench...", flush=True)
    if HAS_LGBM:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )),
        ])
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
        ])

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ---- 评估 ----
    r2 = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
    spearman_r, spearman_p = spearmanr(y_te, y_pred)

    print("\n" + "=" * 60, flush=True)
    print("  跨数据集泛化结果 (wolverton→matbench)", flush=True)
    print("=" * 60, flush=True)
    print(f"  R² (绝对预测):     {r2:.4f}", flush=True)
    print(f"  MAE:               {mae:.4f} eV/atom", flush=True)
    print(f"  RMSE:              {rmse:.4f} eV/atom", flush=True)
    print(f"  Spearman 相关:     {spearman_r:.4f} (p={spearman_p:.2e})", flush=True)
    print(f"  → 排序一致性 {'好' if spearman_r > 0.6 else '中等' if spearman_r > 0.3 else '差'}", flush=True)

    # 保存
    df_out = pd.DataFrame([{
        "metric": ["R2", "MAE", "RMSE", "Spearman", "n_test"],
        "value": [r2, mae, rmse, spearman_r, len(y_te)],
    }])
    out_path = METRICS_DIR / "cross_dataset_generalization.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    # 解读
    print("\n[解读]", flush=True)
    if r2 < 0:
        print(f"  R²={r2:.2f} < 0: 存在系统性偏差 (两数据集 DFT 设置不同)", flush=True)
    print(f"  Spearman={spearman_r:.2f}: ", end="", flush=True)
    if spearman_r > 0.6:
        print("模型学到了可迁移的'相对稳定性排序', 即使绝对值有偏差", flush=True)
    elif spearman_r > 0.3:
        print("部分可迁移, 化学空间差异显著", flush=True)
    else:
        print("泛化能力有限, 两数据集化学空间差异大", flush=True)


if __name__ == "__main__":
    main()
