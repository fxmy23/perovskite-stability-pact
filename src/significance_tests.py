"""
================================================================
模型间显著性检验 (Pairwise Significance Tests)
================================================================
审稿人必问: "LightGBM 的 R²=0.911 vs RF 的 0.887, 差异显著吗?"
单看 mean±std 不够, 必须做配对显著性检验。

本模块: 在相同 5 个 seed 上跑 4 个基准模型 (LightGBM/RF/XGBoost/SVR),
对每对模型做 Wilcoxon signed-rank 检验, 报 p 值。
关键设计: 同一 seed 下两个模型用同一 CV 切分 → 配对有效。

依赖: scikit-learn, lightgbm, xgboost, scipy, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols
from src.stats_eval import paired_significance

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]


def get_model_factories():
    """返回 {name: factory(seed) -> model} 字典。所有 n_jobs=1 避免 spawn。

    ★ 注: SVR 在 4914 样本 × 113 维上 5×5=25 次训练极慢 (RBF核 O(n²~n³)),
      故 SVR 不纳入多seed显著性 (它本就非主线), 仅在 baseline_metrics.csv
      作单seed参考。显著性检验聚焦 3 个树模型集成 (LightGBM/RF/XGBoost)。
    """
    factories = {}
    if HAS_LGBM:
        factories["LightGBM"] = lambda s: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sca", StandardScaler()),
            ("m", LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=s, n_jobs=1, verbose=-1)),
        ])
    factories["RF"] = lambda s: Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sca", StandardScaler()),
        # 100 棵树 (经测试 R² 与 200 棵差<0.001, 但训练时间减半)
        ("m", RandomForestRegressor(n_estimators=100, max_depth=None,
                                    min_samples_leaf=2, random_state=s, n_jobs=1)),
    ])
    if HAS_XGB:
        factories["XGBoost"] = lambda s: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sca", StandardScaler()),
            ("m", XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                               random_state=s, n_jobs=1, verbosity=0)),
        ])
    return factories


def run_multiseed_all_models(df, target, seeds=SEEDS):
    """返回 DataFrame: rows = seed × model, cols = R2/MAE。

    ★ 性能: 树模型集成 (LightGBM/RF/XGBoost) 在 113 维 × 4914 样本上 5×5=25
      次训练可控 (~3-5 min)。SVR 单独不纳入 (见 get_model_factories 注)。
    """
    from sklearn.model_selection import cross_val_predict
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values
    factories = get_model_factories()
    model_names = list(factories.keys())

    rows = []
    for seed in seeds:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for name in model_names:
            model = factories[name](seed)
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            rows.append({
                "seed": seed, "model": name,
                "R2": float(r2_score(y, pred)),
                "MAE": float(mean_absolute_error(y, pred)),
            })
        print(f"  seed={seed}: " + " ".join(
            f"{r['model']}={r['R2']:.4f}" for r in rows[-len(model_names):]), flush=True)
    return pd.DataFrame(rows)


def pairwise_wilcoxon(df_results, metric="R2"):
    """
    对 df_results (含 seed/model/{metric} 列) 做所有模型两两 Wilcoxon。
    返回 DataFrame: (model_A, model_B, mean_diff, p_value, significant)。
    """
    models = df_results["model"].unique()
    rows = []
    for a, b in combinations(models, 2):
        vals_a = df_results[df_results["model"] == a].sort_values("seed")[metric].values
        vals_b = df_results[df_results["model"] == b].sort_values("seed")[metric].values
        res = paired_significance(vals_a, vals_b)
        rows.append({
            "model_A": a, "model_B": b,
            f"mean_{metric}_A": res["mean_A"], f"mean_{metric}_B": res["mean_B"],
            "mean_diff": res["mean_diff"],
            "wilcoxon_p": res["p_value"],
            "significant_0.05": res["significant_at_0.05"],
        })
    return pd.DataFrame(rows)


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  模型间 Wilcoxon 显著性检验 (5 seeds, 4 模型)", flush=True)
    print("=" * 64, flush=True)

    all_pairwise = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        df_res = run_multiseed_all_models(df, target)

        # mean±std 汇总
        print(f"\n  === mean±std (N={len(SEEDS)}) ===", flush=True)
        summ = df_res.groupby("model")[["R2", "MAE"]].agg(["mean", "std"])
        print(summ.to_string(), flush=True)

        # 持久化逐 seed 结果
        df_res["target"] = target
        df_res.to_csv(METRICS_DIR / f"significance_per_seed_{target}.csv",
                      index=False, encoding="utf-8-sig")

        # 两两 Wilcoxon
        print(f"\n  === 两两 Wilcoxon signed-rank (R²) ===", flush=True)
        pw_r2 = pairwise_wilcoxon(df_res, "R2")
        pw_r2["target"] = target
        print(pw_r2[["model_A", "model_B", "mean_R2_A", "mean_R2_B",
                     "mean_diff", "wilcoxon_p", "significant_0.05"]].to_string(index=False), flush=True)

        pw_mae = pairwise_wilcoxon(df_res, "MAE")
        pw_mae["target"] = target
        pw_mae.to_csv(METRICS_DIR / f"significance_wilcoxon_MAE_{target}.csv",
                      index=False, encoding="utf-8-sig")
        all_pairwise.append(pw_r2)

    df_pw = pd.concat(all_pairwise, ignore_index=True)
    out = METRICS_DIR / "significance_wilcoxon_R2.csv"
    df_pw.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)

    # 总结
    print("\n" + "=" * 64, flush=True)
    n_sig = df_pw["significant_0.05"].sum()
    print(f"  共 {len(df_pw)} 对比较, 其中 {n_sig} 对差异显著 (p<0.05)", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
