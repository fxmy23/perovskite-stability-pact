"""
================================================================
符号回归物理层 — 探索阶段 (Step 1)
================================================================
目标: 用 gplearn 在 15 物理特征上发现可解释的形成能/凸包能解析公式。

★ 这是论文方法章的真创新:
  把 PACT 物理层从不可解释的 KernelRidge 替换为符号回归发现的显式方程。
  物理贡献不再是黑盒 R²=0.789, 而是可写进论文的解析式。

本阶段 (探索, 非最终评估):
  - 全量训练集跑 gplearn (探索用, 最终评估在 CV 内, 见 Step 2)
  - 扫描 function_set 与 parsimony_coefficient, 输出"公式复杂度 vs R²"帕累托表
  - 人工挑选最可解释且 R² 合理的配置

注: gplearn 0.4.3 的 SymbolicRegressor 在 gplearn.genetic (非 gplearn.symbolic)。

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from gplearn.genetic import SymbolicRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, PHYS_FEATURES, EXCLUDE_FROM_ML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# 候选 function_set (从简单到复杂, 物理可解释性递减)
FUNCTION_SETS = {
    "basic": {"add": 2, "sub": 2, "mul": 2, "div": 2},
    "basic_sqrt": {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1},
    "basic_log": {"add": 2, "sub": 2, "mul": 2, "div": 2, "log": 1, "sqrt": 1},
    "rich": {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1, "log": 1, "abs": 1, "neg": 1},
}
PARSIMONY = [0.001, 0.01, 0.05]


def safe_eval_equation(est):
    """提取 SR 发现的公式字符串。"""
    try:
        return str(est._program)
    except Exception:
        return "(unknown)"


def program_complexity(est):
    """公式复杂度 = 节点数 (gplearn 内部 length)。"""
    try:
        return len(est._program.program)
    except Exception:
        return -1


def explore_sr(X, y, target_name, seed=42):
    """扫 function_set × parsimony, 返回帕累托表。"""
    print(f"\n{'='*60}", flush=True)
    print(f"  探索 SR: {target_name} (n={len(y)}, 特征={X.shape[1]})", flush=True)
    print(f"{'='*60}", flush=True)

    rows = []
    for fs_name, fs in FUNCTION_SETS.items():
        for par in PARSIMONY:
            tag = f"{fs_name}_p{par}"
            print(f"\n  >> {tag} ...", flush=True)
            try:
                est = SymbolicRegressor(
                    function_set=fs,
                    population_size=2000,
                    generations=30,
                    tournament_size=20,
                    parsimony_coefficient=par,
                    p_crossover=0.7,
                    p_subtree_mutation=0.1,
                    p_hoist_mutation=0.05,
                    p_point_mutation=0.1,
                    metric="mean absolute error",
                    stopping_criteria=0.1,  # MAE 达 0.1 即停 (避免无限跑)
                    n_jobs=1,
                    random_state=seed,
                    verbose=0,
                )
                est.fit(X, y)
                pred = est.predict(X)
                r2 = r2_score(y, pred)
                mae = mean_absolute_error(y, pred)
                eq = safe_eval_equation(est)
                cx = program_complexity(est)
                print(f"     R²={r2:.4f} MAE={mae:.4f} complexity={cx}", flush=True)
                print(f"     公式: {eq[:120]}{'...' if len(eq)>120 else ''}", flush=True)
                rows.append({
                    "target": target_name, "config": tag,
                    "function_set": fs_name, "parsimony": par,
                    "R2_train": r2, "MAE_train": mae,
                    "complexity": cx, "equation": eq,
                })
            except Exception as e:
                print(f"     [ERROR] {type(e).__name__}: {str(e)[:100]}", flush=True)
                rows.append({"target": target_name, "config": tag,
                             "function_set": fs_name, "parsimony": par,
                             "R2_train": np.nan, "MAE_train": np.nan,
                             "complexity": -1, "equation": f"ERROR:{e}"})
    return pd.DataFrame(rows)


def main():
    df = load_features()
    # 物理特征 (排除衍生启发 + 死特征, 与 ML 一致)
    phys_cols = [c for c in PHYS_FEATURES if c not in EXCLUDE_FROM_ML and c in df.columns]
    print(f"[SR-探索] 物理特征 {len(phys_cols)} 维: {phys_cols}", flush=True)

    # 预处理: impute + scale (gplearn 对量纲敏感)
    X_raw = df[phys_cols].values
    y_form = df["formation_energy_per_atom"].values
    y_hull = df["energy_above_hull"].values
    imp = SimpleImputer(strategy="median")
    sca = StandardScaler()
    X = sca.fit_transform(imp.fit_transform(X_raw))

    all_rows = []
    for tname, y in [("formation_energy_per_atom", y_form),
                     ("energy_above_hull", y_hull)]:
        df_res = explore_sr(X, y, tname, seed=42)
        all_rows.append(df_res)

    df_all = pd.concat(all_rows, ignore_index=True)
    out = METRICS_DIR / "sr_exploration.csv"
    df_all.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)

    # 帕累托汇总: 按 target 分组, 选 R² 高且 complexity 低的
    print("\n" + "=" * 60, flush=True)
    print("  帕累托汇总 (R² 高 + complexity 低 = 最优)", flush=True)
    print("=" * 60, flush=True)
    for tname in df_all["target"].unique():
        sub = df_all[df_all["target"] == tname].dropna(subset=["R2_train"])
        sub = sub.sort_values("R2_train", ascending=False)
        print(f"\n  --- {tname} (按 R² 排序) ---", flush=True)
        for _, r in sub.head(6).iterrows():
            print(f"    {r['config']:18s} R²={r['R2_train']:.4f} "
                  f"cx={r['complexity']:3d} | {r['equation'][:80]}", flush=True)


if __name__ == "__main__":
    main()
