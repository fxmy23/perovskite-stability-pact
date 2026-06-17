"""
================================================================
SHAP 特征归因模块
================================================================
论文 P2-2: 对最优模型 (LightGBM) 做 SHAP 解释性分析。

输出:
  1. 全局特征重要性 Top20 (bar plot 数据)
  2. 物理特征 vs Magpie 特征的贡献占比
  3. 容忍因子/八面体因子的 SHAP 趋势 (验证物理规律)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def run_shap_analysis(df, target, top_n=20):
    """对单个目标做 SHAP 分析, 返回特征重要性 DataFrame。"""
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values

    # 训练 LightGBM (用 Pipeline 外的单独模型, SHAP 需要)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = LGBMRegressor(
        n_estimators=200, num_leaves=31, learning_rate=0.1,
        random_state=42, n_jobs=1, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    # SHAP TreeExplainer (对 LightGBM 精确且快)
    explainer = shap.TreeExplainer(model)
    # 用测试集计算 SHAP 值 (样本太多会慢, 限制 1000)
    n_explain = min(1000, len(X_te))
    shap_values = explainer.shap_values(X_te[:n_explain])

    # 全局重要性 = SHAP 绝对值均值
    global_imp = np.abs(shap_values).mean(axis=0)
    df_imp = pd.DataFrame({
        "feature": feat_cols,
        "shap_importance": global_imp,
    }).sort_values("shap_importance", ascending=False)

    # 分类: 物理 vs Magpie
    df_imp["category"] = df_imp["feature"].apply(
        lambda f: "physical" if f.startswith("phys_") else "magpie"
    )

    # 类别贡献占比
    cat_share = df_imp.groupby("category")["shap_importance"].sum()
    cat_share = cat_share / cat_share.sum()

    return df_imp.head(top_n), cat_share, shap_values, X_te[:n_explain], feat_cols


def main():
    if not HAS_SHAP:
        raise SystemExit("[ERROR] 缺少 shap: pip install shap")

    df = load_features()
    print("=" * 60, flush=True)
    print("  SHAP 特征归因分析", flush=True)
    print("=" * 60, flush=True)

    all_imp = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        df_top, cat_share, shap_vals, X_te, feat_cols = run_shap_analysis(df, target)
        df_top["target"] = target
        all_imp.append(df_top)

        print(f"  Top10 特征:", flush=True)
        print(df_top.head(10)[["feature", "shap_importance", "category"]].to_string(index=False), flush=True)
        print(f"\n  类别贡献占比:", flush=True)
        for cat, share in cat_share.items():
            print(f"    {cat}: {share:.1%}", flush=True)

        # 检查物理特征的具体排名
        phys_rank = df_top[df_top["category"] == "physical"]
        if len(phys_rank) > 0:
            print(f"\n  物理特征在Top{len(df_top)}中的排名:", flush=True)
            print(phys_rank[["feature", "shap_importance"]].to_string(index=False), flush=True)

    df_all = pd.concat(all_imp, ignore_index=True)
    out_path = METRICS_DIR / "shap_importance.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  SHAP 分析完成 - 论文核心图数据已生成", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
