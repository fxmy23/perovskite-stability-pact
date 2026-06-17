"""
================================================================
特征工程模块 (三区升级版)
================================================================
为钙钛矿氧化物数据构建三类特征:

  A. Magpie 组成描述符 (~132 维元素性质统计量)
  B. 钙钛矿专属物理特征 (容忍因子 t / 八面体因子 μ 等, PGML 物理基线核心)
  C. 结构/电子原生特征 (wolverton 数据集的晶格常数、畸变类型、
     带隙、磁矩等, 来自 DFT 计算, 反映真实结构信息)

C 类特征是区别于"纯组成特征"工作的关键, 显著提升模型信息量。

依赖: matminer, pymatgen, pandas, numpy

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element

# ----------------------------------------------------------------
# 钙钛矿离子半径表 (Shannon 1976, 单位 Å)
# ----------------------------------------------------------------
R_A_12 = {  # 12 配位 A 位
    "Li": 0.92, "Na": 1.39, "K": 1.64, "Rb": 1.72, "Cs": 1.88,
    "Ca": 1.34, "Sr": 1.44, "Ba": 1.61, "Ra": 1.62,
    "Y": 1.159, "La": 1.36, "Ce": 1.34, "Pr": 1.32, "Nd": 1.30,
    "Sm": 1.24, "Eu": 1.206, "Gd": 1.193, "Tb": 1.18, "Dy": 1.167,
    "Ho": 1.155, "Er": 1.144, "Tm": 1.13, "Yb": 1.118, "Lu": 1.105,
    "Pb": 1.49, "Bi": 1.17, "Mg": 0.72, "Fe": 0.78, "Mn": 0.89,
    "Cd": 1.31, "Sc": 1.06, "Ag": 1.28, "In": 0.94, "Tl": 1.24,
}
R_B_6 = {  # 6 配位 B 位
    "Ti": 0.605, "Zr": 0.72, "Hf": 0.71, "V": 0.54, "Nb": 0.64,
    "Ta": 0.64, "Cr": 0.615, "Mo": 0.65, "W": 0.60, "Mn": 0.645,
    "Fe": 0.645, "Co": 0.61, "Ni": 0.60, "Cu": 0.73, "Zn": 0.74,
    "Al": 0.535, "Ga": 0.62, "In": 0.80, "Sc": 0.745, "Y": 0.90,
    "La": 1.032, "Ce": 1.01, "Pr": 0.99, "Nd": 0.983, "Sm": 0.958,
    "Eu": 0.947, "Gd": 0.938, "Tb": 0.923, "Dy": 0.912, "Ho": 0.901,
    "Er": 0.89, "Tm": 0.88, "Yb": 0.868, "Lu": 0.861,
    "Ru": 0.62, "Rh": 0.665, "Pd": 0.62, "Ir": 0.625, "Pt": 0.625,
    "Sn": 0.69, "Pb": 0.775, "Sb": 0.60, "Bi": 1.03, "Mg": 0.72,
    "Ge": 0.53, "Si": 0.40, "Cd": 0.95, "Hg": 0.69,
}
R_O = 1.40  # 氧六配位离子半径

VALENCES = {  # B 位常见价态
    "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
    "Cr": 3, "Mo": 6, "W": 6, "Mn": 2, "Fe": 3, "Co": 2,
    "Ni": 2, "Cu": 2, "Zn": 2, "Al": 3, "Ga": 3, "In": 3,
    "Sc": 3, "Y": 3, "La": 3, "Ce": 4, "Pr": 3, "Nd": 3,
    "Sm": 3, "Eu": 3, "Gd": 3, "Tb": 3, "Dy": 3, "Ho": 3,
    "Er": 3, "Tm": 3, "Yb": 3, "Lu": 3, "Ru": 4, "Rh": 3,
    "Pd": 2, "Ir": 4, "Pt": 4, "Sn": 4, "Pb": 2, "Sb": 5,
    "Bi": 3, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Na": 1,
    "K": 1, "Rb": 1, "Cs": 1, "Li": 1, "Ge": 4, "Si": 4,
}

# ----------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------
def _avg_radius(symbols, table):
    vals = [table[s] for s in symbols if s in table]
    return float(np.mean(vals)) if vals else np.nan


def _avg_electronegativity(symbols):
    vals = []
    for s in symbols:
        try:
            vals.append(Element(s).X)
        except Exception:
            continue
    return float(np.mean(vals)) if vals else np.nan


def _avg_valence(symbols):
    vals = [VALENCES[s] for s in symbols if s in VALENCES]
    return float(np.mean(vals)) if vals else np.nan


def _avg_property(symbols, prop_name):
    """对一组元素取某项 pymatgen 元素属性的均值。"""
    vals = []
    for s in symbols:
        try:
            el = Element(s)
            v = getattr(el, prop_name)
            if v is not None:
                vals.append(float(v))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else np.nan


def _valence_d_electrons(symbol):
    """
    价层 d 电子数 (只数最外层主量子数的 d 轨道)。

    ★ Bug 修复 (2026-06-16): 原版 sum(orb[2] for orb in es if "d" in orb[1])
    会把所有内层 d 轨道(3d/4d/5d, 各10电子)都计入, 导致 Ac=[Rn]6d¹ 得 31
    而非正确值 1。正确做法: 只数最高主量子数的 d 轨道。
    """
    try:
        el = Element(symbol)
        es = el.full_electronic_structure
        # 找出所有 d 轨道的最大主量子数
        d_orbs = [orb for orb in es if orb[1] == "d"]
        if not d_orbs:
            return 0.0
        max_n = max(orb[0] for orb in d_orbs)
        # 只数 max_n 的 d 轨道电子 (价层 d)
        return float(sum(orb[2] for orb in d_orbs if orb[0] == max_n))
    except Exception:
        return np.nan


def _valence_f_electrons(symbol):
    """
    价层 f 电子数 (只数最外层主量子数的 f 轨道)。
    同 d 电子的 bug 修复逻辑。
    """
    try:
        el = Element(symbol)
        es = el.full_electronic_structure
        f_orbs = [orb for orb in es if orb[1] == "f"]
        if not f_orbs:
            return 0.0
        max_n = max(orb[0] for orb in f_orbs)
        return float(sum(orb[2] for orb in f_orbs if orb[0] == max_n))
    except Exception:
        return np.nan


def _avg_d_electrons(symbols):
    """A/B 位平均价层 d 电子数 (影响磁性/Jahn-Teller畸变)。"""
    vals = []
    for s in symbols:
        v = _valence_d_electrons(s)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _avg_f_electrons(symbols):
    """B 位平均价层 f 电子数 (稀土特征)。"""
    vals = []
    for s in symbols:
        v = _valence_f_electrons(s)
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


def _avg_unpaired(symbols):
    """平均未配对电子数 (磁矩代理, 影响结构畸变)。"""
    vals = []
    for s in symbols:
        try:
            n = Element(s).number_of_unpaired_electrons
            if n is not None:
                vals.append(float(n))
        except Exception:
            continue
    return float(np.mean(vals)) if vals else np.nan


def _ionic_radius(symbol, table, valence=None):
    """
    获取元素离子半径: 先查手工配位数表(精确), 失败则用 pymatgen 兜底。

    pymatgen Element.ionic_radii 返回 {价态: 半径} 字典, 取最接近指定价态
    或常见价态的值。覆盖全周期表, 不会因元素缺失返回 NaN。
    """
    # 1. 优先用手工配位数表 (Shannon 12/6 配位, 更精确)
    if symbol in table:
        return float(table[symbol])
    # 2. pymatgen 兜底 (Shannon 离子半径, 按价态)
    try:
        el = Element(symbol)
        ir = getattr(el, "ionic_radii", None)
        if ir and isinstance(ir, dict) and len(ir) > 0:
            if valence is not None and valence in ir:
                return float(ir[valence])
            # 取最接近 VALENCES 表里记录的价态
            common_v = VALENCES.get(symbol)
            if common_v is not None and common_v in ir:
                return float(ir[common_v])
            # P0-3 修复: 取中位价态而非最大价态
            # 过渡金属最大价态(如Mn7+、Fe6+)在钙钛矿中几乎不存在,
            # 会导致半径系统性偏小。中位价态更接近实际。
            sorted_vs = sorted(ir.keys())
            med_v = sorted_vs[len(sorted_vs) // 2]
            return float(ir[med_v])
    except Exception:
        pass
    # 3. P0-3 修复: 不用原子半径兜底(量纲不同), 直接返回 NaN
    # 让上层 imputation 显式处理, 比返回错误量纲的值更安全
    return np.nan


# ----------------------------------------------------------------
# A. 组成描述符 (纯 pymatgen 实现, 绕开 matminer 的 multiprocessing)
# ----------------------------------------------------------------
# 元素性质查询表: 用 pymatgen Element 直接取, 完全单进程, 无 spawn 风险。
# 覆盖 Magpie 的核心性质子集, 对每个化合物按原子分数加权计算统计量。
# 说明: 原计划用 matminer 的 ElementProperty.from_preset("magpie"),
#       但其 BaseFeaturizer 默认 n_jobs=cpu_count()(本机32), 会 spawn 32
#       个子进程, Windows 下递归执行 __main__ 导致进程风暴+整机死机
#       (见 troubleshooting_log.md F-01)。set_n_jobs(1) 也无法彻底关闭。
#       故改用纯手算, 物理含义与 Magpie 等价, 速度更快且零并行风险。

# 要计算的元素性质及其 pymatgen Element 属性名
_ELEMENT_PROPS = {
    "Z": "Z",                          # 原子序数
    "atomic_mass": "atomic_mass",      # 原子质量
    "X": "X",                          # 电负性 (Allen)
    "atomic_radius": "atomic_radius",  # 原子半径
    "row": "row",                      # 周期
    "group": "group",                  # 族
    "mendeleev_no": "mendeleev_no",    # 门捷列夫序号
    "electrical_resistivity": "electrical_resistivity",
    "velocity_of_sound": "velocity_of_sound",
    "thermal_conductivity": "thermal_conductivity",
    "melting_point": "melting_point",
    "bulk_modulus": "bulk_modulus",
    "youngs_modulus": "youngs_modulus",
    "brinell_hardness": "brinell_hardness",
    "rigidity_modulus": "rigidity_modulus",
    "density": "density_of_solid",
}

# 每种性质计算的统计量
_STATS = ["mean", "wmean", "min", "max", "range", "std"]


def _safe_get(element, prop_name):
    """安全获取元素属性, 缺失返回 None。"""
    try:
        v = getattr(element, prop_name)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def generate_magpie_features(formulas: list[str]) -> pd.DataFrame:
    """
    手工计算组成描述符 (Magpie 等价), 纯单进程。

    对每个化学式:
      1. 解析元素组成 + 原子分数
      2. 对 16 种元素性质, 查每个组成元素的值
      3. 计算 6 种统计量: 均值/加权均值/最小/最大/极差/标准差
    => 共 16 × 6 = 96 维描述符, 列名 magpie_<prop>_<stat>
    """
    rows = []
    # 预建元素缓存, 避免重复构造 Element 对象
    elem_cache = {}

    for formula in formulas:
        try:
            comp = Composition(formula)
        except Exception:
            # 解析失败, 填全 NaN
            row = {f"magpie_{p}_{s}": np.nan
                   for p in _ELEMENT_PROPS for s in _STATS}
            rows.append(row)
            continue

        # 获取组成元素及其原子分数
        elems_fracs = [(el, amt) for el, amt in comp.items()]
        total = sum(amt for _, amt in elems_fracs)
        fracs = [amt / total for _, amt in elems_fracs]

        row = {}
        for prop_label, attr in _ELEMENT_PROPS.items():
            # 查每个元素的该性质
            vals = []
            for el, _ in elems_fracs:
                if el.symbol not in elem_cache:
                    elem_cache[el.symbol] = el
                v = _safe_get(elem_cache[el.symbol], attr)
                if v is not None:
                    vals.append(v)
                else:
                    vals.append(np.nan)

            vals = np.array(vals, dtype=float)
            if len(vals) == 0 or np.all(np.isnan(vals)):
                for s in _STATS:
                    row[f"magpie_{prop_label}_{s}"] = np.nan
                continue

            valid = vals[~np.isnan(vals)]
            valid_fracs = np.array(fracs)[~np.isnan(vals)] if len(fracs) == len(vals) else None

            row[f"magpie_{prop_label}_mean"] = float(np.nanmean(vals))
            # 加权均值 (按原子分数)
            if valid_fracs is not None and len(valid_fracs) == len(valid):
                wmean = float(np.average(valid, weights=valid_fracs))
            else:
                wmean = float(np.mean(valid))
            row[f"magpie_{prop_label}_wmean"] = wmean
            row[f"magpie_{prop_label}_min"] = float(np.nanmin(vals))
            row[f"magpie_{prop_label}_max"] = float(np.nanmax(vals))
            row[f"magpie_{prop_label}_range"] = (
                float(np.nanmax(vals) - np.nanmin(vals))
            )
            row[f"magpie_{prop_label}_std"] = float(np.nanstd(vals))

        rows.append(row)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# B. 钙钛矿专属物理特征 (PGML 物理基线核心)
# ----------------------------------------------------------------
def generate_physical_features(
    formulas: list[str],
    a_sites: list[str] | None = None,
    b_sites: list[str] | None = None,
) -> pd.DataFrame:
    """
    计算钙钛矿物理特征。优先用 wolverton 提供的 a_site/b_site 元素标注,
    缺失时回退到化学式解析。

    输出特征:
      phys_tolerance_factor   容忍因子 t (Goldschmidt)
      phys_octahedral_factor  八面体因子 μ = r_B / r_O
      phys_electroneg_diff    A/B 位电负性差
      phys_radius_ratio_AB    A/B 半径比
      phys_b_site_valence     B 位平均价态
      phys_a_site_radius      A 位平均半径 (12配位)
      phys_b_site_radius      B 位平均半径 (6配位)
      phys_in_stable_zone     是否落入经典稳定区 (t∈[0.8,1.0] 且 μ∈[0.414,0.732])
      phys_stability_score    连续型稳定度评分 (越接近经典区中心分越高)
    """
    rows = []
    for i, f in enumerate(formulas):
        # 优先使用 wolverton 的 a_site/b_site 标注
        if a_sites is not None and i < len(a_sites) and pd.notna(a_sites[i]):
            a_list = [str(a_sites[i])]
            b_list = [str(b_sites[i])] if b_sites is not None and pd.notna(b_sites[i]) else []
            if not b_list:
                a_list, b_list = _parse_sites_from_formula(f)
        else:
            a_list, b_list = _parse_sites_from_formula(f)

        # 离子半径: 先查精确配位数表, 失败用 pymatgen 兜底(覆盖全周期表)
        r_a_vals = [_ionic_radius(s, R_A_12) for s in a_list]
        r_b_vals = [_ionic_radius(s, R_B_6) for s in b_list]
        r_a_vals = [v for v in r_a_vals if np.isfinite(v)]
        r_b_vals = [v for v in r_b_vals if np.isfinite(v)]
        r_a = float(np.mean(r_a_vals)) if r_a_vals else np.nan
        r_b = float(np.mean(r_b_vals)) if r_b_vals else np.nan
        en_a = _avg_electronegativity(a_list)
        en_b = _avg_electronegativity(b_list)

        if np.isfinite(r_a) and np.isfinite(r_b):
            t = (r_a + R_O) / (np.sqrt(2) * (r_b + R_O))
            mu = r_b / R_O
            ratio = r_a / r_b
        else:
            t = mu = ratio = np.nan

        # 连续稳定度评分: 距经典区中心 (t=0.9, μ=0.573) 的负距离
        if np.isfinite(t) and np.isfinite(mu):
            # 归一化距离, t 中心0.9带宽0.1, μ 中心0.573带宽0.159
            dt = (t - 0.9) / 0.1
            dmu = (mu - 0.573) / 0.159
            score = float(np.exp(-0.5 * (dt**2 + dmu**2)))
            in_zone = int(0.8 <= t <= 1.0 and 0.414 <= mu <= 0.732)
        else:
            score = np.nan
            in_zone = 0

        rows.append({
            "phys_tolerance_factor": t,
            "phys_octahedral_factor": mu,
            "phys_electroneg_diff_AB": (
                abs(en_a - en_b) if np.isfinite(en_a) and np.isfinite(en_b) else np.nan
            ),
            "phys_radius_ratio_AB": ratio,
            "phys_b_site_valence": _avg_valence(b_list),
            "phys_a_site_radius": r_a,
            "phys_b_site_radius": r_b,
            "phys_a_site_en": en_a,
            "phys_b_site_en": en_b,
            "phys_b_site_row": _avg_property(b_list, "row"),
            "phys_b_site_group": _avg_property(b_list, "group"),
            # ★ 电子构型特征 (借鉴 Nature Comm 2024)
            # d/f 电子数影响磁性和结构畸变, 是稳定性的物理本质
            "phys_a_site_d_electrons": _avg_d_electrons(a_list),
            "phys_b_site_d_electrons": _avg_d_electrons(b_list),
            "phys_b_site_f_electrons": _avg_f_electrons(b_list),
            "phys_b_site_unpaired": _avg_unpaired(b_list),
            "phys_in_stable_zone": in_zone,
            "phys_stability_score": score,
        })
    return pd.DataFrame(rows)


def _parse_sites_from_formula(formula: str) -> tuple[list, list]:
    """
    从化学式解析 A/B 位。
    规则: 含量最高=A位, 其余=B位。
    P0-2 修复: 含量并列时用元素符号字典序做 tie-break, 保证可复现。
    (并列时返回 NaN, 让上层用 wolverton 的 a_site/b_site 标注兜底)
    """
    try:
        comp = Composition(formula)
    except Exception:
        return [], []
    items = [(el, amt) for el, amt in comp.items() if el.symbol != "O"]
    if not items:
        return [], []
    # 按含量降序, 并列时按原子序数升序(A位通常是较重/较大的碱土/稀土)
    items.sort(key=lambda x: (-x[1], -x[0].Z))
    return [items[0][0].symbol], [el.symbol for el, _ in items[1:]]


# ----------------------------------------------------------------
# C. 原生结构特征 (wolverton 数据集字段, 反映 DFT 计算的真实结构)
# ----------------------------------------------------------------
STRUCTURAL_FIELDS = [
    "lattice_a", "lattice_b", "lattice_c",
    "lattice_alpha", "lattice_beta", "lattice_gamma",
    "volume_per_atom", "band_gap_pbe", "magnetic_moment",
    "lowest_distortion",
]


def generate_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 wolverton 原生字段构造结构特征:
      - 晶格常数比 (b/a, c/a 反映畸变程度)
      - 畸变类型 one-hot (cubic/tetragonal/orthorhombic/rhombohedral)
      - 单胞体积、带隙、磁矩 (DFT 计算值)
      - Goldschmidt 容忍因子 vs 实际 c/a 的偏差 (结构畸变度量)
    """
    out = pd.DataFrame(index=df.index)

    # 晶格常数比 (畸变量度)
    if all(c in df.columns for c in ["lattice_a", "lattice_b", "lattice_c"]):
        a = df["lattice_a"].replace(0, np.nan)
        b = df["lattice_b"].replace(0, np.nan)
        c = df["lattice_c"].replace(0, np.nan)
        out["struct_ba_ratio"] = b / a
        out["struct_ca_ratio"] = c / a
        # P1 修复: 用已替换0为NaN的变量, 避免除零; 清理 inf
        abc = pd.concat([a, b, c], axis=1)
        out["struct_abc_spread"] = (abc.std(axis=1) / abc.mean(axis=1)).replace(
            [np.inf, -np.inf], np.nan
        )

    # 晶格角 (理想立方为 90, 偏离量度畸变)
    for ang in ["lattice_alpha", "lattice_beta", "lattice_gamma"]:
        if ang in df.columns:
            out[f"struct_{ang}_dev"] = (df[ang] - 90.0).abs()

    # 直接传递的结构/电子特征
    rename_map = {
        "volume_per_atom": "struct_vpa",
        "band_gap_pbe": "struct_band_gap",
        "magnetic_moment": "struct_mag_moment",
    }
    for orig, new in rename_map.items():
        if orig in df.columns:
            out[new] = df[orig]

    # 畸变类型 one-hot
    if "lowest_distortion" in df.columns:
        dummies = pd.get_dummies(df["lowest_distortion"], prefix="struct_distort")
        out = pd.concat([out, dummies], axis=1)

    return out


# ----------------------------------------------------------------
# 主入口: 拼接三类特征
# ----------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    formula_col: str = "formula_pretty",
    include_magpie: bool = True,
    include_physical: bool = True,
    include_structural: bool = True,
) -> pd.DataFrame:
    """
    构建完整特征矩阵: Magpie 组成 + 物理特征 + 原生结构特征。
    """
    print(f"[FEAT] 对 {len(df)} 条样本生成特征 ...")
    formulas = df[formula_col].tolist()
    out = df.copy()

    parts = []
    if include_magpie:
        print("  [1/3] Magpie 组成描述符 (~132维)...")
        parts.append(generate_magpie_features(formulas))
    if include_physical:
        print("  [2/3] 钙钛矿物理特征 (容忍因子等)...")
        a_sites = df["a_site_element"].tolist() if "a_site_element" in df.columns else None
        b_sites = df["b_site_element"].tolist() if "b_site_element" in df.columns else None
        parts.append(generate_physical_features(formulas, a_sites, b_sites))
    if include_structural:
        print("  [3/3] 原生结构特征 (晶格/畸变/带隙/磁矩)...")
        parts.append(generate_structural_features(df))

    # 拼接 (所有部分已 reset index 对齐)
    feat_dfs = [p.reset_index(drop=True) for p in parts]
    out = pd.concat([out.reset_index(drop=True)] + feat_dfs, axis=1)

    # ★ 泄露闭环修复 (2026-06-16):
    # 不在全量数据上做中位数填充(会泄露测试集分布到训练)。
    # 只做 bool→int 转换, NaN 保留, 由 sklearn Pipeline 的 SimpleImputer
    # 在 CV 折内 fit_transform, 确保无泄露。
    feat_cols = [c for c in out.columns if c.startswith(("magpie_", "phys_", "struct_"))]
    for c in feat_cols:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)
        # NaN 保留, 不填充 (Pipeline 折内处理)

    n_feat = len(feat_cols)
    print(f"[FEAT] 特征矩阵构建完成: {n_feat} 维 (NaN保留, Pipeline折内填充)")
    return out


# ----------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------
if __name__ == "__main__":
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    clean_path = PROJECT_ROOT / "data" / "processed" / "perovskite_clean.csv"
    if not clean_path.exists():
        raise SystemExit(f"[ERROR] 未找到 {clean_path}, 先运行 data_acquisition.py")

    df = pd.read_csv(clean_path)
    df_feat = build_features(df)

    out_path = PROJECT_ROOT / "data" / "processed" / "perovskite_features.csv"
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] 特征数据: {out_path}  ({df_feat.shape})")

    # 物理特征与目标的相关性速览 (PGML 分析的前置)
    phys_cols = [c for c in df_feat.columns if c.startswith("phys_")]
    if phys_cols:
        print("\n=== 物理特征 vs 形成能/凸包能 相关性 ===")
        corr = df_feat[phys_cols + ["formation_energy_per_atom", "energy_above_hull"]].corr()
        for target in ["formation_energy_per_atom", "energy_above_hull"]:
            print(f"\n  与 {target}:")
            print(corr[target].drop(target).sort_values(key=abs, ascending=False).head(8).to_string())
