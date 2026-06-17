"""
================================================================
Matbench 基准对比模块
================================================================
论文 P1-2: 在 matbench_perovskites (18928条) 上跑我们的方法,
与文献 SOTA 对比, 证明性能 competitive。

文献参考 (2025 tree-based benchmark):
  XGBoost on matbench perovskite: MAE = 0.227 eV/atom, R² = 0.79

注意: matbench_perovskites 是结构-based 任务 (含晶体结构),
但我们用纯组成特征 (magpie+phys) 以保持与主实验一致,
这实际上是对我们更难的设定 (不用结构信息)。

依赖: matminer, scikit-learn, lightgbm, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    HAS_BOOST = True
except ImportError:
    HAS_BOOST = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def main():
    print("=" * 60, flush=True)
    print("  Matbench Perovskite 基准对比", flush=True)
    print("=" * 60, flush=True)

    # 加载 matbench_perovskites
    print("[LOAD] matbench_perovskites (18928条)...", flush=True)
    from matminer.datasets import load_dataset
    df_mb = load_dataset("matbench_perovskites")
    print(f"  样本: {len(df_mb)}, 列: {list(df_mb.columns)}", flush=True)

    # 从结构提取化学式 (matbench 提供 structure 对象)
    df_mb["formula_pretty"] = df_mb["structure"].apply(lambda s: s.composition.reduced_formula)
    df_mb["elements"] = df_mb["structure"].apply(lambda s: "-".join([e.symbol for e in s.composition.elements]))

    # 用我们的特征工程 (纯组成, 不用结构)
    from src.features import generate_magpie_features, generate_physical_features
    print("[FEAT] 生成 Magpie 特征...", flush=True)
    mag = generate_magpie_features(df_mb["formula_pretty"].tolist())
    print("[FEAT] 生成物理特征...", flush=True)
    phys = generate_physical_features(df_mb["formula_pretty"].tolist())

    df_feat = pd.concat([
        df_mb[["formula_pretty", "e_form"]].reset_index(drop=True),
        mag.reset_index(drop=True),
        phys.reset_index(drop=True),
    ], axis=1)

    # 数值清洗
    feat_cols = [c for c in df_feat.columns if c.startswith(("magpie_", "phys_"))]
    for c in feat_cols:
        if df_feat[c].dtype == object:
            df_feat[c] = pd.to_numeric(df_feat[c], errors="coerce")
        if df_feat[c].isna().any():
            df_feat[c] = df_feat[c].fillna(df_feat[c].median())

    X = df_feat[feat_cols].values
    y = df_feat["e_form"].values
    print(f"[INFO] 特征 {len(feat_cols)} 维, 样本 {len(y)}", flush=True)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    models = {
        "RF": Pipeline([("s", StandardScaler()),
                        ("m", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1))]),
    }
    if HAS_BOOST:
        models["XGBoost"] = Pipeline([("s", StandardScaler()),
                                      ("m", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                                         random_state=42, n_jobs=1, verbosity=0))])
        models["LightGBM"] = Pipeline([("s", StandardScaler()),
                                       ("m", LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                                           random_state=42, n_jobs=1, verbose=-1))])

    for name, model in models.items():
        print(f"  >> {name} ...", end=" ", flush=True)
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        r2 = float(r2_score(y, y_pred))
        results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "dataset": "matbench_perovskites"})
        print(f"MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}", flush=True)

    df_out = pd.DataFrame(results)
    out_path = METRICS_DIR / "matbench_benchmark.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    # 与文献对比
    print("\n" + "=" * 60, flush=True)
    print("  与文献 SOTA 对比 (matbench perovskite, 组成特征)", flush=True)
    print("=" * 60, flush=True)
    print(f"  文献 XGBoost (2025 benchmark): MAE=0.227, R²=0.79", flush=True)
    for _, row in df_out.iterrows():
        print(f"  本研究 {row['model']}:           MAE={row['MAE']:.3f}, R²={row['R2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
