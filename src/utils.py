"""
================================================================
共用工具模块
================================================================
提供所有下游模块共享的工具函数:
  - load_features(): 加载特征矩阵并强制数值化 + 控制是否含 struct 特征
  - get_feature_cols(): 获取特征列名

★ 重要设计 (P0-1 修复, 2026-06-16):
  默认 exclude_struct=True, 排除所有 struct_ 前缀特征。
  原因: struct_ 特征 (晶格常数/畸变类型/带隙) 是 DFT 弛豫后的计算结果,
  与目标 E_hull/E_f 同源, 构成 post-DFT 数据泄露 (见 review_findings.md P0-1)。
  审计: 排除 struct 后形成能 R² 0.966→0.893, 凸包能 0.933→0.760。
  这些 struct 特征对真实"新材料预测"不可得 (预测新材料时不知道其晶格常数)。

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 纯预测性特征前缀 (不含 struct, 避免 DFT 泄露)
PURE_PREFIXES = ("magpie_", "phys_")
# 全部特征前缀 (含 struct, 仅用于对比/消融实验)
ALL_PREFIXES = ("magpie_", "phys_", "struct_")

# ★ 审计修复 (2026-06-16): 应从 ML 输入中排除的特征
#   - phys_in_stable_zone, phys_stability_score: 衍生启发式 (由同一批物理量
#     组合而成), 喂给 ML 会形成"特征重叠/弱循环", 虽非硬泄露 (R²差<0.001)
#     但损害"物理贡献"叙事的纯净性。物理基线层 (PHYS_FEATURES) 已正确排除。
#   - phys_b_site_unpaired: 实测 100% NaN (元素性质表缺失), 死特征。
EXCLUDE_FROM_ML = {"phys_in_stable_zone", "phys_stability_score", "phys_b_site_unpaired"}

# ★ 物理特征子集 (单一来源, 所有模块从此引用)
# 统一框架: 物理层用全部有明确物理含义的特征 (14个纯物理量)
# ★ 修复 C2 (V9, 2026-06-16): 移除 phys_b_site_unpaired (100% NaN 死特征)。
#   原版列表含15项但实际只用14(被EXCLUDE_FROM_ML排除), 列表声明不一致。
#   现统一为14项, 列表与实际使用一致。
# 排除 phys_in_stable_zone(二元判据输出) 和 phys_stability_score(衍生量)
PHYS_FEATURES = [
    "phys_tolerance_factor", "phys_octahedral_factor",
    "phys_electroneg_diff_AB", "phys_radius_ratio_AB",
    "phys_b_site_valence", "phys_a_site_radius", "phys_b_site_radius",
    "phys_a_site_en", "phys_b_site_en",
    "phys_b_site_row", "phys_b_site_group",
    "phys_a_site_d_electrons", "phys_b_site_d_electrons",
    "phys_b_site_f_electrons",
]


def load_features(
    path: str | Path | None = None,
    exclude_struct: bool = True,
) -> pd.DataFrame:
    """
    加载特征矩阵并强制数值化。

    Args:
        path: 特征 CSV 路径 (默认 data/processed/perovskite_features.csv)
        exclude_struct: 是否排除 struct_ 特征 (默认 True, 避免泄露)

    Returns:
        清洗后的 DataFrame, 所有特征列均为数值类型。
        注意: 即使 exclude_struct=True, struct_ 列仍保留在 df 中 (供 EDA),
        但 get_feature_cols() 会根据前缀过滤。
    """
    if path is None:
        path = PROJECT_ROOT / "data" / "processed" / "perovskite_features.csv"
    df = pd.read_csv(path)

    # 所有可能的特征列 (含 struct, 用于数值清洗)
    all_feat = [c for c in df.columns if c.startswith(ALL_PREFIXES)]

    for c in all_feat:
        # bool 转 int
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
            continue
        # 强制数值化: 非数值(如'-')→ NaN
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # P0-1 修复: 不在全量上做中位数填充(会泄露测试集分布)
        # 保留 NaN, 让 sklearn Pipeline 的 SimpleImputer 在 CV 折内 fit

    return df


def get_feature_cols(
    df: pd.DataFrame,
    exclude_struct: bool = True,
    exclude_derived: bool = True,
) -> list[str]:
    """
    返回特征列名。

    Args:
        df: 特征 DataFrame
        exclude_struct: True=纯预测特征(magpie+phys), False=含struct全部
        exclude_derived: True=同时排除衍生启发式与死特征 (EXCLUDE_FROM_ML)。
            审计修复: 这些特征非硬泄露但损害叙事纯净性/为死特征。
    """
    prefixes = PURE_PREFIXES if exclude_struct else ALL_PREFIXES
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if exclude_derived:
        cols = [c for c in cols if c not in EXCLUDE_FROM_ML]
    return cols


if __name__ == "__main__":
    df = load_features()
    pure = get_feature_cols(df, exclude_struct=True)
    full = get_feature_cols(df, exclude_struct=False)
    print(f"样本数: {len(df)}")
    print(f"纯预测特征 (排除struct): {len(pure)} 维")
    print(f"全部特征 (含struct):    {len(full)} 维")
    print(f"被排除的struct特征:     {len(full) - len(pure)} 维")
    bad = [c for c in pure if df[c].dtype == object]
    print(f"残留字符串列: {bad}")
    print(f"NaN 总数: {df[pure].isna().sum().sum()}")
    print("[OK] 数据清洗完成")
