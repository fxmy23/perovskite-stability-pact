"""
================================================================
候选材料三层验证 (Candidate Validation)
================================================================
背景: 原 screening 输出 9-50 个候选稳定钙钛矿, 但零独立验证, 是实用性
  最大短板。本科生单机无法做 DFT, MP API 实测不可达 (heartbeat 无响应)。

本模块实现三层廉价但有力的验证:

  ★ Tier 1 (跨数据集): 用 matbench_perovskites (独立 DFT 源, 18928 条)
    交叉核对我们候选的形成能。若候选在 matbench 中存在且形成能<0
    (热力学稳定方向), 是独立佐证。

  ★ Tier 2 (留出元素交叉验证): 对每个候选 (A,B), 从训练集移除该 A 或 B
    元素的所有样本, 重训模型, 看是否仍预测 E_hull<0.05。
    这是"候选对训练集的依赖度"测试——若移除该元素仍预测稳定,
    说明预测非过拟合, 是真正可外推的发现。

  ★ Tier 3 (文献合成可达性): 用物理判据 (容差因子 t∈[0.8,1.05], 八面体因子
    μ_oct∈[0.4,0.9]) 评估候选的"可合成性先验", 作为弱验证。

输出: candidate_validation.csv, 每个候选标注三层验证结论。

依赖: matminer, pymatgen, scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DISCOVERY_DIR = PROJECT_ROOT / "results" / "discovery"
DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05


# ----------------------------------------------------------------
# Tier 1: matbench 跨数据集交叉核对
# ----------------------------------------------------------------
def tier1_matbench_crosscheck(candidates, matbench_df=None):
    """
    用 matbench_perovskites 核对候选是否被独立 DFT 研究过。

    ★ 审计修正 (2026-06-16): 经跨库量纲诊断 (src/cross_dataset_v2.py),
      matbench 与 wolverton 的形成能不同量纲 (Pearson r=0.358, 系统偏差
      +2.5 eV), 因此 matbench e_form 数值不能直接佐证 wolverton 预测。
      本 Tier 改为检验"候选是否被独立 DFT 收录" (存在性证据),
      而非数值一致性。LaAlO3 在 matbench 存在 = 该化合物被独立研究,
      但其 matbench e_form=-0.10 不等同于我们的 wolverton 预测可信。
    """
    print("[Tier1] matbench 跨数据集收录核对 (存在性, 非数值一致)...", flush=True)
    if matbench_df is None:
        from matminer.datasets import load_dataset
        matbench_df = load_dataset("matbench_perovskites", download_if_missing=False)

    # 建 formula → e_form 查找表 (取最小值, 若有多型)
    lookup = {}
    for s, ef in zip(matbench_df["structure"], matbench_df["e_form"]):
        f = s.composition.reduced_formula
        lookup.setdefault(f, []).append(float(ef))
    lookup = {k: min(v) for k, v in lookup.items()}  # 最稳定多型

    rows = []
    for _, cand in candidates.iterrows():
        f = cand["formula_pretty"]
        in_mb = f in lookup
        rows.append({
            "formula_pretty": f,
            "in_matbench": in_mb,
            "matbench_e_form": lookup.get(f, np.nan),
            # ★ 修正: 不再用 matbench e_form<0 判 formable (量纲不同)
            # 改为: 被独立 DFT 收录即算弱佐证 (该化合物被研究过)
            "independently_studied": bool(in_mb),
        })
    df_t1 = pd.DataFrame(rows)
    n_in = df_t1["in_matbench"].sum()
    print(f"  候选 {len(df_t1)} 个, 被 matbench 独立 DFT 收录: {n_in}", flush=True)
    print(f"  (注: matbench 与 wolverton 不同量纲, 仅证收录, 不证数值一致)", flush=True)
    return df_t1


# ----------------------------------------------------------------
# Tier 2: 留出元素交叉验证
# ----------------------------------------------------------------
def tier2_leave_element_out(df_train, candidates, target="energy_above_hull",
                            feat_prefixes=("magpie_", "phys_"),
                            n_estimators=80, stable_thresh=STABLE_THRESH):
    """
    对每个候选 (A,B):
      1. 从训练集移除所有 a_site==A 的样本 (留出 A 位元素)
      2. 重训 RF, 预测候选 E_hull
      3. 同样留出 B 位
      4. 取两者预测的 max 作为"最坏情况下"预测
    若 max 仍 < thresh → 候选对训练集依赖低, 预测稳健 (强佐证)。

    ★ 审计修复 (2026-06-16): 原版只用 15 维物理特征重训 (与主模型 113 维不一致,
      过悲观)。改为用与主模型同特征集 (magpie+phys, 排除衍生)。
      但候选缺 magpie 特征 (screening 只算物理), 故 LOEO 仍限物理子集 +
      明确标注 "phys-only LOEO", 不与主线 113 维混淆。
    """
    print("[Tier2] 留出元素交叉验证 (LOEO-style, 物理特征子集)...", flush=True)
    feat_cols = [c for c in df_train.columns if c.startswith(feat_prefixes)]
    X_full = df_train[feat_cols].values
    y_full = df_train[target].values

    # 候选物理特征 (用 screening 已算好的)
    phys_cols = sorted([c for c in feat_cols if c.startswith("phys_")])
    medians = df_train[phys_cols].median()

    rows = []
    for _, cand in candidates.iterrows():
        a, b = cand["a_site_element"], cand["b_site_element"]
        f = cand["formula_pretty"]
        # 候选物理特征向量
        cand_phys = np.array([[cand[c] if c in cand and not pd.isna(cand[c])
                               else medians[c] for c in phys_cols]])

        results = {"formula_pretty": f, "a": a, "b": b}
        for leave_el, site_col in [(a, "a_site_element"), (b, "b_site_element")]:
            mask = df_train[site_col] != leave_el
            if mask.sum() < 100 or df_train.loc[mask, target].notna().sum() < 10:
                results[f"pred_leave_{leave_el}"] = np.nan
                continue
            # ★ 修复: 用物理特征子集 (与候选可得特征一致), 明确标注
            X_tr = df_train.loc[mask, phys_cols].values
            y_tr = df_train.loc[mask, target].values
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            cand_p = imp.transform(cand_phys)
            rf = RandomForestRegressor(n_estimators=n_estimators,
                                       random_state=42, n_jobs=1)
            rf.fit(X_tr, y_tr)
            results[f"pred_leave_{leave_el}"] = float(rf.predict(cand_p)[0])

        # 最坏情况 (两个留出预测的 max)
        p_a = results.get(f"pred_leave_{a}", np.nan)
        p_b = results.get(f"pred_leave_{b}", np.nan)
        valid = [v for v in [p_a, p_b] if not np.isnan(v)]
        results["pred_worst_loeo"] = max(valid) if valid else np.nan
        results["robust_stable"] = bool(results["pred_worst_loeo"] < stable_thresh) \
            if not np.isnan(results["pred_worst_loeo"]) else False
        rows.append(results)

    df_t2 = pd.DataFrame(rows)
    n_robust = df_t2["robust_stable"].sum()
    print(f"  候选 {len(df_t2)} 个, 留出元素后仍预测稳定: {n_robust}", flush=True)
    return df_t2


# ----------------------------------------------------------------
# Tier 3: 物理可合成性先验
# ----------------------------------------------------------------
def tier3_synthesizability(candidates, t_lo=0.8, t_hi=1.05,
                           mu_lo=0.4, mu_hi=0.9):
    """
    用经典容差因子 t 和八面体因子 μ_oct 评估可合成性先验。
    t∈[0.8,1.05] 且 μ_oct∈[0.4,0.9] → 经典钙钛矿可合成区 (Goldschmidt)。
    """
    print("[Tier3] 物理可合成性先验 (Goldschmidt 判据)...", flush=True)
    rows = []
    for _, cand in candidates.iterrows():
        t = cand.get("phys_tolerance_factor", np.nan)
        mu = cand.get("phys_octahedral_factor", np.nan)
        t_ok = (t_lo <= t <= t_hi) if not pd.isna(t) else False
        mu_ok = (mu_lo <= mu <= mu_hi) if not pd.isna(mu) else False
        rows.append({
            "formula_pretty": cand["formula_pretty"],
            "tolerance_factor": t,
            "octahedral_factor": mu,
            "in_goldschmidt_zone": bool(t_ok and mu_ok),
            "t_in_range": bool(t_ok),
            "mu_in_range": bool(mu_ok),
        })
    df_t3 = pd.DataFrame(rows)
    n_gold = df_t3["in_goldschmidt_zone"].sum()
    print(f"  候选 {len(df_t3)} 个, 在 Goldschmidt 可合成区: {n_gold}", flush=True)
    return df_t3


# ----------------------------------------------------------------
# 主入口: 三层验证合并
# ----------------------------------------------------------------
def validate_candidates(candidates_path=None, df_train=None):
    if candidates_path is None:
        candidates_path = DISCOVERY_DIR / "screened_candidates.csv"
    candidates = pd.read_csv(candidates_path)
    print(f"=" * 64, flush=True)
    print(f"  候选材料三层验证 (n={len(candidates)})", flush=True)
    print(f"=" * 64, flush=True)

    # 只验证真正的新候选 (不在训练集)
    new_cands = candidates[~candidates["is_in_training"]].copy()
    print(f"  新候选 (未在训练集): {len(new_cands)} 个", flush=True)
    if len(new_cands) == 0:
        new_cands = candidates.copy()
        print(f"  (无新候选, 验证全部 {len(new_cands)} 个)", flush=True)

    # Tier 1
    df_t1 = tier1_matbench_crosscheck(new_cands)
    # Tier 2
    if df_train is None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils import load_features
        df_train = load_features()
    df_t2 = tier2_leave_element_out(df_train, new_cands)
    # Tier 3
    df_t3 = tier3_synthesizability(new_cands)

    # 合并 (按 formula_pretty join)
    df_val = new_cands[["formula_pretty", "a_site_element", "b_site_element",
                        "pred_energy_above_hull", "pred_std"]].copy()
    df_val = df_val.merge(df_t1, on="formula_pretty", how="left")
    df_val = df_val.merge(df_t3, on="formula_pretty", how="left")
    # Tier 2 列选择性合并 (避免重复 a/b)
    t2_keep = ["formula_pretty", "pred_worst_loeo", "robust_stable"]
    df_val = df_val.merge(df_t2[t2_keep], on="formula_pretty", how="left")

    # 综合验证等级 (修正: matbench 仅证收录, 不再加权 formable)
    def grade(row):
        score = 0
        if row.get("independently_studied", False) or row.get("in_matbench", False):
            score += 1  # 被独立 DFT 收录 (存在性证据)
        if row.get("robust_stable", False):
            score += 2  # LOEO 留出后仍预测稳定 (强)
        if row.get("in_goldschmidt_zone", False):
            score += 1  # 经典可合成区
        if score >= 3:
            return "strong"
        if score >= 2:
            return "moderate"
        return "weak"
    df_val["validation_grade"] = df_val.apply(grade, axis=1)

    # 持久化
    out_path = DISCOVERY_DIR / "candidate_validation.csv"
    df_val.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    # 汇总
    print("\n" + "=" * 64, flush=True)
    print("  三层验证汇总", flush=True)
    print("=" * 64, flush=True)
    print(df_val.groupby("validation_grade").size().to_string(), flush=True)
    print(f"\n  强验证候选 (strong):", flush=True)
    strong = df_val[df_val["validation_grade"] == "strong"]
    show = ["formula_pretty", "pred_energy_above_hull", "matbench_e_form",
            "pred_worst_loeo", "tolerance_factor"]
    show = [c for c in show if c in strong.columns]
    if len(strong) > 0:
        print(strong[show].to_string(index=False), flush=True)
    else:
        print("  (无强验证候选, 展示 moderate)", flush=True)
        mod = df_val[df_val["validation_grade"] == "moderate"]
        if len(mod) > 0:
            print(mod[show].head(10).to_string(index=False), flush=True)

    return df_val


if __name__ == "__main__":
    validate_candidates()
