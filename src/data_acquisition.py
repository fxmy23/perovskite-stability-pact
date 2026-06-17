"""
================================================================
钙钛矿氧化物数据获取模块(零登录版)
================================================================
从 matminer 内置数据集直接加载钙钛矿氧化物的 DFT 计算数据,
**完全不需要 Materials Project 账号或 API key**, 国内网络可直连。

数据来源(均托管于 figshare, 国内可直连):
  1. wolverton_oxides  [主力]
     - 4914 条钙钛矿氧化物 (ABO3 体系为主, 含 A2O3 变体)
     - 字段: formula, atom a (A位), atom b (B位), lowest distortion,
             e_form (形成能) ← 回归目标1
             e_hull  (凸包能) ← 回归目标2 / 稳定性判据
             mu_b (磁矩), vpa (体积/原子), gap pbe (PBE带隙),
             a/b/c/alpha/beta/gamma (晶格常数), e_form oxygen
     - 来源: Emery & Wolverton, 计算 DFT 数据

  2. matbench_perovskites  [补充, 可选]
     - 18928 条钙钛矿 (含晶体结构), 形成能 e_form
     - 用于结构特征扩展或样本量扩充

输出:
    data/raw/perovskite_raw.csv         原始合并数据
    data/processed/perovskite_clean.csv 清洗后数据(含 stable 标签)

依赖: matminer, pandas, numpy

用法:
    python src/data_acquisition.py

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# 字段标准化: 把 wolverton_oxides 的列名映射到统一规范
# ----------------------------------------------------------------
WOLVERTON_RENAME = {
    "formula": "formula_pretty",
    "atom a": "a_site_element",
    "atom b": "b_site_element",
    "lowest distortion": "lowest_distortion",
    "e_form": "formation_energy_per_atom",     # ★ 回归目标1
    "e_hull": "energy_above_hull",             # ★ 回归目标2
    "mu_b": "magnetic_moment",
    "vpa": "volume_per_atom",
    "gap pbe": "band_gap_pbe",
    "a": "lattice_a",
    "b": "lattice_b",
    "c": "lattice_c",
    "alpha": "lattice_alpha",
    "beta": "lattice_beta",
    "gamma": "lattice_gamma",
    "e_form oxygen": "e_form_oxygen",
}


# ----------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------
def load_wolverton_oxides() -> pd.DataFrame:
    """
    加载 wolverton_oxides 数据集并标准化列名。

    这是本项目的主力数据集:4914 条 ABO3 钙钛矿氧化物,
    同时包含形成能 E_f 和凸包能 E_hull 两个回归目标。
    """
    from matminer.datasets import load_dataset

    print("[LOAD] 加载 wolverton_oxides (主力, 4914条 ABO3 钙钛矿氧化物)...")
    df = load_dataset("wolverton_oxides")
    df = df.rename(columns=WOLVERTON_RENAME)
    df["source"] = "wolverton_oxides"
    print(f"  [成功] {len(df)} 行, {len(df.columns)} 列")
    return df


def load_matbench_perovskites() -> pd.DataFrame | None:
    """
    加载 matbench_perovskites 数据集(可选, 用于扩充样本量)。

    18928 条钙钛矿(含晶体结构), 仅含形成能。如加载失败则跳过,
    不影响主流程(wolverton_oxides 已足够支撑论文)。
    """
    from matminer.datasets import load_dataset

    print("[LOAD] 加载 matbench_perovskites (补充, 18928条, 含结构)...")
    try:
        df = load_dataset("matbench_perovskites")
        # structure 列是 pymatgen Structure 对象, 此处先保留字符串形式
        df["structure_str"] = df["structure"].astype(str)
        df = df.drop(columns=["structure"])
        df = df.rename(columns={"e_form": "formation_energy_per_atom"})
        df["source"] = "matbench_perovskites"
        # 该数据集无 e_hull, 故 stability 目标仅由 wolverton 提供
        df["energy_above_hull"] = np.nan
        print(f"  [成功] {len(df)} 行")
        return df
    except Exception as e:
        print(f"  [跳过] 加载失败: {type(e).__name__}: {str(e)[:100]}")
        print("         (不影响主流程, wolverton_oxides 已足够)")
        return None


# ----------------------------------------------------------------
# 数据清洗
# ----------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗规则:
      1. 去除两个回归目标(E_f / E_hull)同时缺失的样本
         (wolverton 都有, 此步主要针对合并后可能引入的 NaN)
      2. 去除重复化学式(保留首次出现)
      3. 形成能异常值过滤(E_f 超出 [-5, 3] eV/atom 视为计算异常)
      4. 凸包能负值修正(DFT 偶有微小负值, 截断为 0)
      5. 派生标签: stable (E_hull < 0.05 eV/atom 视为稳定/亚稳可用)
      6. 钙钛矿类型标注: single_ABO3 / variant
    """
    print(f"[CLEAN] 清洗前样本数: {len(df)}")
    df = df.copy()

    # 目标字段缺失剔除(两者皆缺才删; 仅缺 e_hull 的保留做形成能建模)
    df = df.dropna(
        subset=["formation_energy_per_atom", "energy_above_hull"], how="all"
    )
    print(f"  去除双目标缺失后: {len(df)}")

    # 去重(按化学式)
    if "formula_pretty" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["formula_pretty"], keep="first")
        print(f"  化学式去重后: {len(df)} (剔除 {before - len(df)})")

    # 形成能异常值过滤
    if "formation_energy_per_atom" in df.columns:
        mask_f = (df["formation_energy_per_atom"] >= -5) & (
            df["formation_energy_per_atom"] <= 3
        )
        df = df[mask_f]
        print(f"  形成能异常值过滤后: {len(df)}")

    # 凸包能负值截断(仅对非 NaN)
    if "energy_above_hull" in df.columns:
        df["energy_above_hull"] = df["energy_above_hull"].clip(lower=0)

    # 派生标签: 稳定性
    #   E_hull < 50 meV/atom 视为亚稳可用 (文献常用阈值)
    df["stable"] = (df["energy_above_hull"] < 0.05).astype(int)

    # 钙钛矿类型标注
    #   ABO3 -> single_ABO3; A2O3 等含双 A 位 -> variant
    if "formula_pretty" in df.columns:
        def _ptype(f):
            f = str(f)
            if f.endswith("O3") and f[:2] not in ("A2",):
                # 粗判: 简单 ABO3 (排除 A2O3 这类明显变体)
                return "single_ABO3"
            return "variant"
        df["perovskite_type"] = df["formula_pretty"].apply(_ptype)

    # 缺失的数值列填 NaN 占位(后续特征工程统一处理)
    df = df.reset_index(drop=True)
    print(f"[CLEAN] 清洗后样本数: {len(df)}")
    return df


# ----------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("  钙钛矿氧化物数据获取 (零登录版 / matminer 内置数据集)")
    print("=" * 60)
    print("  注: 本流程不需要 Materials Project 账号或 API key")
    print("      数据托管于 figshare, 国内网络可直连")
    print()

    # 1. 加载主力数据集
    df_main = load_wolverton_oxides()

    # 2. 尝试加载补充数据集(失败则跳过)
    df_supp = load_matbench_perovskites()

    # 3. 持久化原始数据
    raw_path = RAW_DIR / "perovskite_raw.csv"
    df_main.to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] 主力原始数据: {raw_path}  ({len(df_main)} 行)")

    if df_supp is not None:
        raw_supp_path = RAW_DIR / "matbench_perovskites_raw.csv"
        df_supp.to_csv(raw_supp_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 补充原始数据: {raw_supp_path}  ({len(df_supp)} 行)")

    # 4. 清洗主力数据(wolverton_oxides 是论文核心)
    df_clean = clean_data(df_main)

    # 5. 持久化清洗后数据
    clean_path = PROCESSED_DIR / "perovskite_clean.csv"
    df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 清洗数据: {clean_path}  ({len(df_clean)} 行)")

    # 6. 数据集统计摘要
    print("\n" + "=" * 60)
    print("  数据集统计摘要 (wolverton_oxides)")
    print("=" * 60)
    print(f"  总样本数:        {len(df_clean)}")
    if "perovskite_type" in df_clean.columns:
        n_single = (df_clean["perovskite_type"] == "single_ABO3").sum()
        n_variant = (df_clean["perovskite_type"] == "variant").sum()
        print(f"  单钙钛矿 ABO3:   {n_single}")
        print(f"  变体 (含 A2O3):  {n_variant}")
    print(f"  稳定样本占比:    {df_clean['stable'].mean():.2%}")
    print(
        f"  形成能均值:      "
        f"{df_clean['formation_energy_per_atom'].mean():.3f} eV/atom"
    )
    print(
        f"  凸包能均值:      "
        f"{df_clean['energy_above_hull'].mean():.3f} eV/atom"
    )
    if "band_gap_pbe" in df_clean.columns:
        print(f"  PBE 带隙均值:    {df_clean['band_gap_pbe'].mean():.3f} eV")
    # 涉及元素统计
    if "a_site_element" in df_clean.columns:
        n_a = df_clean["a_site_element"].nunique()
        n_b = df_clean["b_site_element"].nunique()
        print(f"  涉及 A 位元素:   {n_a} 种")
        print(f"  涉及 B 位元素:   {n_b} 种")

    print("\n[DONE] 数据获取流程结束。下一步: python src/features.py")


if __name__ == "__main__":
    main()
