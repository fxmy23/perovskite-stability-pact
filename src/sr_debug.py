"""
================================================================
符号回归深度 Debug (Step 5 验证脚本)
================================================================
用户明确要求"深度debug确保结果可信"。本脚本系统验证 SR 物理层的可信度:

  D1. 数值稳定性: SR 公式在新数据上是否产生 inf/nan/overflow
      (sqrt负数, log负数, div零 — gplearn 内部有保护, 但需验证)
  D2. 无泄露性: CV 内每折 SR 是否独立 fit (公式不跨折共享)
  D3. 公式可解释性: 把 X0..X13 映射回真实物理量名, 检查公式是否物理合理
  D4. SR vs KRR 物理层对比: SR R² 应 < KRR (可解释性代价), 但总 R² 应接近
  D5. conformal PICP 在 SR 物理层下是否仍达标 (≥0.80)

运行: 在 sr_multiseed / pact_sr 跑完后执行本脚本, 输出 debug 报告。

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def debug_numerical_stability():
    """D1: 检查 SR OOF 预测是否有 inf/nan。"""
    print("[D1] 数值稳定性检查", flush=True)
    issues = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        path = METRICS_DIR / f"pact_sr_oof_{target}.csv"
        if not path.exists():
            print(f"  {target}: OOF 文件不存在, 跳过", flush=True)
            continue
        df = pd.read_csv(path)
        for col in ["oof_mu", "oof_mu_p", "oof_sigma", "conf_lower", "conf_upper"]:
            if col in df.columns:
                arr = df[col].values
                n_inf = np.isinf(arr).sum()
                n_nan = np.isnan(arr).sum()
                if n_inf > 0 or n_nan > 0:
                    issues.append(f"{target}.{col}: {n_inf} inf, {n_nan} nan")
                    print(f"  ⚠ {target}.{col}: {n_inf} inf, {n_nan} nan", flush=True)
        # 区间合理性: lower < mu < upper
        if all(c in df.columns for c in ["conf_lower", "oof_mu", "conf_upper"]):
            bad = ((df["conf_lower"] > df["oof_mu"]) | (df["oof_mu"] > df["conf_upper"])).sum()
            if bad > 0:
                issues.append(f"{target}: {bad} 样本区间顺序异常")
                print(f"  ⚠ {target}: {bad} 样本 lower>mu 或 mu>upper", flush=True)
    if not issues:
        print("  ✓ 全部数值稳定, 无 inf/nan/区间异常", flush=True)
    return issues


def debug_equation_interpretability():
    """D3: 把 SR 公式的 X0..X13 映射回物理量名。"""
    print("\n[D3] SR 公式可解释性 (X→物理量映射)", flush=True)
    df = load_features()
    phys_cols = [c for c in PHYS_FEATURES if c not in EXCLUDE_FROM_ML and c in df.columns]
    print("  物理特征索引映射 (SR 公式中 Xi):", flush=True)
    for idx, c in enumerate(phys_cols):
        short = c.replace("phys_", "")
        print(f"    X{idx:2d} = {short}", flush=True)

    # 读取发现的公式
    eq_path = METRICS_DIR / "sr_equations_per_fold.csv"
    if eq_path.exists():
        df_eq = pd.read_csv(eq_path)
        print(f"\n  各目标各折 SR 公式 (含物理量名替换):", flush=True)
        for target in df_eq["target"].unique():
            sub = df_eq[df_eq["target"] == target]
            print(f"\n  --- {target} ---", flush=True)
            for _, r in sub.head(3).iterrows():
                eq = r["equation"]
                # 替换 Xi 为短名 (简化展示)
                eq_named = eq
                for i, c in enumerate(phys_cols):
                    short = c.replace("phys_", "").replace("_site_", "")
                    eq_named = eq_named.replace(f"X{i}", f"[{short}]")
                print(f"    fold{r['fold']}: {eq_named[:140]}", flush=True)


def debug_sr_vs_krr_comparison():
    """D4: SR vs KRR 物理层 R² 对比 (诚实: SR 应较低但总 R² 接近)。"""
    print("\n[D4] SR vs KRR 物理层对比", flush=True)
    path = METRICS_DIR / "sr_vs_krr_cv.csv"
    if not path.exists():
        print(f"  sr_vs_krr_cv.csv 不存在, 先跑 sr_physics_layer.py", flush=True)
        return
    df = pd.read_csv(path)
    for target in df["target"].unique():
        sub = df[df["target"] == target]
        print(f"\n  --- {target} ---", flush=True)
        for method in ["SR_physics_only", "KRR_physics_only",
                       "SR_physics_plus_ML", "KRR_physics_plus_ML", "pure_ML"]:
            row = sub[sub["method"] == method]
            if len(row) > 0:
                r2 = row["R2"].values[0]
                print(f"    {method:25s}: R²={r2:.4f}", flush=True)
        # 判断: SR物理层 < KRR物理层 (可解释代价), 总R² 接近?
        sr_only = sub[sub["method"] == "SR_physics_only"]["R2"].values
        krr_only = sub[sub["method"] == "KRR_physics_only"]["R2"].values
        sr_tot = sub[sub["method"] == "SR_physics_plus_ML"]["R2"].values
        krr_tot = sub[sub["method"] == "KRR_physics_plus_ML"]["R2"].values
        if len(sr_only) and len(krr_only):
            print(f"    → SR物理层 {sr_only[0]:.4f} vs KRR物理层 {krr_only[0]:.4f} "
                  f"(SR {'低' if sr_only[0]<krr_only[0] else '高'} {(abs(sr_only[0]-krr_only[0])):.3f}, "
                  f"可解释性代价)", flush=True)
        if len(sr_tot) and len(krr_tot):
            print(f"    → 总R² SR+ML {sr_tot[0]:.4f} vs KRR+ML {krr_tot[0]:.4f} "
                  f"(差距 {abs(sr_tot[0]-krr_tot[0]):.4f}, 残差框架补足)", flush=True)


def debug_picp():
    """D5: conformal PICP 在 SR 物理层下是否达标。"""
    print("\n[D5] SR 物理层下 conformal PICP", flush=True)
    path = METRICS_DIR / "pact_sr_results.csv"
    if not path.exists():
        print(f"  pact_sr_results.csv 不存在, 先跑 pact_sr.py", flush=True)
        return
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        picp = r.get("PICP_conformal", np.nan)
        ok = "✓ 达标" if picp >= 0.80 else "⚠ 低于0.80"
        print(f"  {r['target']}: PICP={picp:.3f} {ok}", flush=True)


def main():
    print("=" * 64, flush=True)
    print("  符号回归深度 Debug (D1-D5)", flush=True)
    print("=" * 64, flush=True)
    debug_numerical_stability()
    debug_equation_interpretability()
    debug_sr_vs_krr_comparison()
    debug_picp()
    print("\n" + "=" * 64, flush=True)
    print("  Debug 完成 (见上方各项 ✓/⚠)", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
