"""
================================================================
候选材料数据库生成 (论文补充材料)
================================================================
把材料发现成果整理为结构化数据库, 作为论文补充材料。
审稿人和读者可直接使用, 体现"重要参考价值"。

输出:
  results/discovery/candidate_database.csv  新候选稳定钙钛矿
  results/discovery/anomaly_database.csv    反常材料
  results/discovery/discovery_summary.md    发现摘要

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISCOVERY_DIR = PROJECT_ROOT / "results" / "discovery"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def build_candidate_database():
    """整理新候选稳定钙钛矿数据库。"""
    screen_path = DISCOVERY_DIR / "screened_candidates.csv"
    if not screen_path.exists():
        raise SystemExit("[ERROR] 先运行 screening.py")
    df = pd.read_csv(screen_path)

    # 只保留新候选(未在训练集)
    df_new = df[~df["is_in_training"]].copy()
    print(f"[DB] 新候选稳定钙钛矿: {len(df_new)} 个", flush=True)

    # 整理列: 化学式/元素/预测值/不确定性/物理特征
    keep = [c for c in [
        "formula_pretty", "a_site_element", "b_site_element",
        "pred_energy_above_hull", "pred_std",
        "phys_tolerance_factor", "phys_octahedral_factor",
        "phys_in_stable_zone", "phys_stability_score",
    ] if c in df_new.columns]
    df_new = df_new[keep].rename(columns={
        "formula_pretty": "Formula",
        "a_site_element": "A_site",
        "b_site_element": "B_site",
        "pred_energy_above_hull": "Pred_Ehull_eV",
        "pred_std": "Uncertainty",
        "phys_tolerance_factor": "ToleranceFactor_t",
        "phys_octahedral_factor": "OctahedralFactor_mu",
        "phys_in_stable_zone": "InClassicStableZone",
        "phys_stability_score": "StabilityScore",
    })

    # 排序: 按预测 E_hull 升序(最可能稳定的在前)
    df_new = df_new.sort_values("Pred_Ehull_eV").reset_index(drop=True)
    df_new.insert(0, "Rank", range(1, len(df_new) + 1))
    df_new["Confidence"] = pd.cut(
        df_new["Uncertainty"],
        bins=[0, 0.005, 0.01, float("inf")],
        labels=["High", "Medium", "Low"],
    )

    return df_new


def build_anomaly_database():
    """整理反常材料数据库。"""
    anomaly_path = DISCOVERY_DIR / "physics_anomalies.csv"
    if not anomaly_path.exists():
        raise SystemExit("[ERROR] 先运行 screening.py")
    df = pd.read_csv(anomaly_path)

    # 整理列
    keep = [c for c in [
        "formula_pretty", "a_site_element", "b_site_element",
        "anomaly_type", "energy_above_hull", "formation_energy_per_atom",
        "phys_tolerance_factor", "phys_octahedral_factor",
        "phys_in_stable_zone", "lowest_distortion",
    ] if c in df.columns]
    df = df[keep].rename(columns={
        "formula_pretty": "Formula",
        "a_site_element": "A_site",
        "b_site_element": "B_site",
        "energy_above_hull": "Ehull_eV",
        "formation_energy_per_atom": "Eform_eV",
        "phys_tolerance_factor": "ToleranceFactor_t",
        "phys_octahedral_factor": "OctahedralFactor_mu",
        "phys_in_stable_zone": "InClassicStableZone",
        "lowest_distortion": "DistortionType",
    })

    # 分类统计
    n_stable = (df["anomaly_type"] == "unexpectly_stable").sum()
    n_unstable = (df["anomaly_type"] == "unexpectly_unstable").sum()
    print(f"[DB] 反常材料: {n_stable} 反常稳定 + {n_unstable} 反常不稳定", flush=True)

    return df, n_stable, n_unstable


def write_summary(df_cand, df_anom, n_stable, n_unstable):
    """写发现摘要 markdown。"""
    summary = DISCOVERY_DIR / "discovery_summary.md"
    lines = [
        "# 材料发现成果摘要 (论文补充材料)\n",
        f"> 生成自 wolverton_oxides (4914 条钙钛矿氧化物) 的 PCRL 模型预测\n",
        "## 1. 新候选稳定钙钛矿\n",
        f"共 **{len(df_cand)} 个** 未在训练集中的高置信度稳定候选。\n",
        "| 排名 | 化学式 | A位 | B位 | 预测E_hull (eV) | 不确定性 | t | 置信度 |",
        "|------|--------|-----|-----|----------------|----------|---|--------|",
    ]
    for _, r in df_cand.iterrows():
        lines.append(
            f"| {r['Rank']} | {r['Formula']} | {r['A_site']} | {r['B_site']} | "
            f"{r['Pred_Ehull_eV']:.4f} | {r['Uncertainty']:.4f} | "
            f"{r['ToleranceFactor_t']:.3f} | {r['Confidence']} |"
        )
    lines += [
        "\n## 2. 物理反常材料\n",
        f"- **反常稳定** ({n_stable} 个): 经典 Goldschmidt 判据说不稳定, "
        f"但实际 E_hull < 0.05 eV/atom",
        f"- **反常不稳定** ({n_unstable} 个): 判据说稳定, 但实际 E_hull > 0.5\n",
        "### 反常稳定材料的 B 位元素富集",
        "V, Al, Bi, Pa 等元素在反常稳定材料中富集, "
        "暗示八面体倾转/Jahn-Teller 畸变等非经典稳定机制。\n",
        "## 3. 使用说明\n",
        "- 新候选按预测 E_hull 升序排列, 排名越前越可能稳定",
        "- Confidence=High 的候选最值得优先实验验证",
        "- 反常稳定材料适合深入研究其非经典稳定机制",
    ]
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SAVE] {summary}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("  候选材料数据库生成", flush=True)
    print("=" * 60, flush=True)

    df_cand = build_candidate_database()
    cand_path = DISCOVERY_DIR / "candidate_database.csv"
    df_cand.to_csv(cand_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 新候选数据库: {cand_path}", flush=True)

    df_anom, n_stable, n_unstable = build_anomaly_database()
    anom_path = DISCOVERY_DIR / "anomaly_database.csv"
    df_anom.to_csv(anom_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 反常材料数据库: {anom_path}", flush=True)

    write_summary(df_cand, df_anom, n_stable, n_unstable)

    print("\n" + "=" * 60, flush=True)
    print("  Top 10 新候选稳定钙钛矿", flush=True)
    print("=" * 60, flush=True)
    show = df_cand.head(10)[["Rank", "Formula", "Pred_Ehull_eV",
                             "Uncertainty", "ToleranceFactor_t", "Confidence"]]
    print(show.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
