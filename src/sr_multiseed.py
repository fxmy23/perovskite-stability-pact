"""
================================================================
符号回归物理层 — 多 seed 稳定性 (Step 3)
================================================================
对 SR vs KRR CV 评估跑 3 个 seed, 报:
  - SR 物理层 R² mean±std (稳定性)
  - 各 seed 发现的公式是否同结构 (一致性)
  - 总 R² (SR物理+ML残差) mean±std

依赖: src.sr_physics_layer

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
from src.utils import load_features
from src.sr_physics_layer import evaluate_sr_vs_krr_cv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

SEEDS = [42, 123, 456]


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  SR 物理层多 seed 稳定性 (N_SEEDS=3)", flush=True)
    print("=" * 64, flush=True)

    rows = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n########## {target} ##########", flush=True)
        for seed in SEEDS:
            print(f"\n  === seed={seed} ===", flush=True)
            res, eqs, _ = evaluate_sr_vs_krr_cv(df, target, seed=seed)
            for method, metrics in res.items():
                if method == "target":
                    continue
                rows.append({
                    "target": target, "seed": seed, "method": method,
                    **metrics,
                })
            # 记录该 seed 的公式 (fold 1 作代表)
            if eqs:
                rows.append({"target": target, "seed": seed,
                             "method": "SR_equation_fold1",
                             "R2": np.nan, "MAE": np.nan, "RMSE": np.nan,
                             "equation": eqs[0]})

    df_out = pd.DataFrame(rows)
    out = METRICS_DIR / "sr_multiseed.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)

    # mean±std 汇总
    print("\n" + "=" * 64, flush=True)
    print("  多 seed 汇总 (mean±std)", flush=True)
    print("=" * 64, flush=True)
    for target in df_out["target"].dropna().unique():
        sub = df_out[df_out["target"] == target]
        for method in ["SR_physics_only", "SR_physics_plus_ML",
                       "KRR_physics_only", "KRR_physics_plus_ML", "pure_ML"]:
            ms = sub[sub["method"] == method]["R2"].dropna()
            if len(ms) > 0:
                print(f"  {target:30s} {method:22s}: "
                      f"R²={ms.mean():.4f}±{ms.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
