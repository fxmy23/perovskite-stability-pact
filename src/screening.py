"""
================================================================
材料筛选模块 (Materials Discovery & Anomaly Analysis)
================================================================
论文"发现"卖点核心, 两条腿:

  Part A: 异常材料剖析 (Physics Anomaly Mining)
    找出"经典判据说该不稳定、但实际 E_hull 很低"的反常钙钛矿。
    这类材料违背容忍因子经验规律, 背后可能藏着新的稳定机制
    (Jahn-Teller 畸变、电子构型稳定、共价键贡献等), 是物理发现的金矿。

  Part B: 高通量新候选筛选 (High-Throughput Screening)
    扫描未收录在训练集中的 A-B 元素组合, 用训练好的模型预测,
    结合不确定性过滤, 输出"高置信度的新稳定钙钛矿候选"清单。
    这是实打实的材料发现, 论文最强卖点。

依赖: scikit-learn, numpy, pandas, itertools

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import generate_physical_features

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
DISCOVERY_DIR = RESULTS_DIR / "discovery"
DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# Part A: 异常材料剖析
# ----------------------------------------------------------------
def find_physics_anomalies(
    df: pd.DataFrame,
    e_hull_col: str = "energy_above_hull",
    stable_thresh: float = 0.05,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    找出违背经典物理判据的反常钙钛矿。

    反常稳定: phys_in_stable_zone == 0 (不在经典稳定区)
              但 energy_above_hull < stable_thresh (实际稳定)
    反常不稳定: phys_in_stable_zone == 1 (在经典区)
                但 energy_above_hull > 0.5 (实际不稳定)
    """
    print("[DISCOVERY-A] 挖掘物理异常材料...", flush=True)

    if "phys_in_stable_zone" not in df.columns:
        raise ValueError("缺少 phys_in_stable_zone 列, 先运行 features.py")

    # 反常稳定: 经典判据说不行, 但实际稳定
    anomaly_stable = df[
        (df["phys_in_stable_zone"] == 0) & (df[e_hull_col] < stable_thresh)
    ].copy()
    anomaly_stable["anomaly_type"] = "unexpectly_stable"
    print(f"  反常稳定 (判据说不行但实际稳定): {len(anomaly_stable)} 个", flush=True)

    # 反常不稳定: 经典判据说行, 但实际不稳定
    anomaly_unstable = df[
        (df["phys_in_stable_zone"] == 1) & (df[e_hull_col] > 0.5)
    ].copy()
    anomaly_unstable["anomaly_type"] = "unexpectly_unstable"
    print(f"  反常不稳定 (判据说行但实际不稳): {len(anomaly_unstable)} 个", flush=True)

    # 合并并按"惊讶度"排序
    anomaly_stable["surprise"] = -anomaly_stable[e_hull_col]
    anomaly_unstable["surprise"] = anomaly_unstable[e_hull_col]

    df_anomaly = pd.concat([anomaly_stable, anomaly_unstable], ignore_index=True)
    df_anomaly = df_anomaly.sort_values("surprise", ascending=False).head(top_n)

    # 共性分析: 反常稳定材料的元素分布
    if len(anomaly_stable) > 0:
        print("\n  === 反常稳定材料的 B 位元素 Top10 ===", flush=True)
        if "b_site_element" in anomaly_stable.columns:
            print(anomaly_stable["b_site_element"].value_counts().head(10).to_string(), flush=True)
        print("\n  === 反常稳定材料的结构畸变类型 ===", flush=True)
        if "lowest_distortion" in anomaly_stable.columns:
            print(anomaly_stable["lowest_distortion"].value_counts().to_string(), flush=True)

    return df_anomaly


# ----------------------------------------------------------------
# Part B: 高通量新候选筛选
# ----------------------------------------------------------------
def screen_candidates(
    df_train: pd.DataFrame,
    target: str = "energy_above_hull",
    feat_prefixes: tuple = ("magpie_", "phys_"),  # P0-1: 排除struct避免泄露
    stable_thresh: float = 0.05,
    n_models: int = 8,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    高通量筛选新候选稳定钙钛矿。

    流程:
      1. 用训练集 A/B 元素笛卡尔积构造候选空间
      2. 用物理特征子集训练 RF 集成 (候选只有物理特征)
      3. 预测 + 不确定性 (集成方差)
      4. 筛选: 预测 E_hull < thresh 且不确定性 < 中位数
      5. 优先未在训练集中的新候选
    """
    print(f"[DISCOVERY-B] 高通量筛选新候选 (目标: {target})...", flush=True)

    trained_formulas = set(df_train["formula_pretty"].tolist())
    a_elements = df_train["a_site_element"].dropna().unique().tolist()
    b_elements = df_train["b_site_element"].dropna().unique().tolist()

    if not a_elements or not b_elements:
        raise ValueError("需要 a_site_element / b_site_element 列")

    print(f"  候选空间: {len(a_elements)} A位 x {len(b_elements)} B位 = "
          f"{len(a_elements)*len(b_elements)} 组合", flush=True)
    rows = []
    for a, b in product(a_elements, b_elements):
        if a == b:
            continue
        rows.append({
            "formula_pretty": f"{a}{b}O3",
            "a_site_element": a,
            "b_site_element": b,
            "is_in_training": f"{a}{b}O3" in trained_formulas,
        })
    df_cand = pd.DataFrame(rows)
    n_new = (~df_cand["is_in_training"]).sum()
    print(f"  其中 {n_new} 个组合未在训练集中 (真正的新候选)", flush=True)

    # 候选特征生成 (只用物理特征, Magpie 对大候选空间太慢)
    print("  生成候选物理特征...", flush=True)
    phys_df = generate_physical_features(
        df_cand["formula_pretty"].tolist(),
        df_cand["a_site_element"].tolist(),
        df_cand["b_site_element"].tolist(),
    )
    # ★ P0-4 修复: 在 df_cand 阶段就 join 物理特征,
    #   这样筛选排序后特征自然跟随, 不会错位
    df_cand = df_cand.reset_index(drop=True)
    phys_df = phys_df.reset_index(drop=True)
    df_cand = pd.concat([df_cand, phys_df], axis=1)

    # 训练用的物理特征列 (P1-3 修复: 用有序列表强制对齐)
    feat_cols = [c for c in df_train.columns if c.startswith(feat_prefixes)]
    phys_feat_cols = sorted([c for c in feat_cols if c.startswith("phys_")])
    # 补齐候选缺失的物理特征列 + 强制列顺序一致
    for c in phys_feat_cols:
        if c not in df_cand.columns:
            df_cand[c] = df_train[c].median()
    X_cand_phys = df_cand[phys_feat_cols].fillna(
        df_train[phys_feat_cols].median()
    ).values
    # 训练集也用同一有序列
    X_train_phys = df_train[phys_feat_cols].values

    # 用物理特征子集训练集成
    y_tr = df_train[target].values
    print(f"  训练 {n_models} 个 RF 集成 (物理特征)...", flush=True)
    models_phys = []
    for m in range(n_models):
        rf = RandomForestRegressor(
            n_estimators=80, random_state=42 + m,
            n_jobs=1, max_samples=0.8,
        )
        rf.fit(X_train_phys, y_tr)
        models_phys.append(rf)

    # 集成预测 + 不确定性
    preds = np.array([m.predict(X_cand_phys) for m in models_phys])
    df_cand["pred_" + target] = preds.mean(axis=0)
    df_cand["pred_std"] = preds.std(axis=0)

    # 筛选: 预测稳定 + 低不确定性
    median_std = df_cand["pred_std"].median()
    mask = (df_cand["pred_" + target] < stable_thresh) & (df_cand["pred_std"] < median_std)
    df_screen = df_cand[mask].copy()

    # 优先未在训练集中的新候选
    df_screen["priority"] = (~df_screen["is_in_training"]).astype(int)
    df_screen = df_screen.sort_values(
        ["priority", "pred_" + target], ascending=[False, True]
    )

    print(f"  筛选通过 (预测稳定且低不确定性): {len(df_screen)} 个", flush=True)
    print(f"  其中新候选 (未在训练集): {(~df_screen['is_in_training']).sum()} 个", flush=True)

    # ★ P0-4 修复: 物理判据已在 df_cand 中, 无需再按位置赋值
    df_screen = df_screen.reset_index(drop=True)
    return df_screen.head(top_n)


# ----------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------
def main():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils import load_features
    df = load_features()
    if df is None or len(df) == 0:
        raise SystemExit("[ERROR] 特征数据为空, 先运行 features.py")

    print("=" * 60, flush=True)
    print("  材料发现与异常剖析", flush=True)
    print("=" * 60, flush=True)

    # ---- Part A: 异常材料 ----
    df_anomaly = find_physics_anomalies(df)
    anomaly_path = DISCOVERY_DIR / "physics_anomalies.csv"
    df_anomaly.to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 物理异常材料: {anomaly_path}  ({len(df_anomaly)} 条)", flush=True)

    # ---- Part B: 高通量筛选 ----
    df_screen = screen_candidates(df, target="energy_above_hull", top_n=50)
    screen_path = DISCOVERY_DIR / "screened_candidates.csv"
    df_screen.to_csv(screen_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 筛选候选: {screen_path}  ({len(df_screen)} 条)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  Top 15 新候选稳定钙钛矿 (未在训练集)", flush=True)
    print("=" * 60, flush=True)
    new_cands = df_screen[~df_screen["is_in_training"]].head(15)
    if len(new_cands) > 0:
        show_cols = [c for c in [
            "formula_pretty", "a_site_element", "b_site_element",
            "pred_energy_above_hull", "pred_std",
            "phys_tolerance_factor", "phys_in_stable_zone"
        ] if c in new_cands.columns]
        print(new_cands[show_cols].to_string(index=False), flush=True)
    else:
        print("  (无新候选, 所有筛选通过的均在训练集中)", flush=True)


if __name__ == "__main__":
    main()
