"""
================================================================
Multi-seed PICP 评估 (审计修复补充)
================================================================
背景: PACT v2 单 seed PICP=0.823 是单次试验值。conformal 保证是边际期望
  (200 次合成试验均值 0.805), 单次会波动 (std~0.019)。
  审稿人会问 "PICP 是否稳定"。本模块跑 5 个 seed 报 mean±std, 给出诚实区间。

依赖: src.pact_v2 (复用其 fit/predict 闭包)

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES
from src.conformal import conformal_intervals_cv, compute_picp_mpiw

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

SEEDS = [42, 123, 456, 789, 2024]


def run_multiseed_picp(df, target, seeds=SEEDS, alpha=0.2):
    """复用 pact_v2 的 fit/predict 闭包, 跑 5 seed PICP。"""
    from sklearn.model_selection import KFold
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from lightgbm import LGBMRegressor

    feat_cols = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]
    X = df[feat_cols].values
    y = df[target].values

    # fit/predict 闭包 (与 pact_v2 一致)
    n_ml = 5
    def fit_fn(Xtr, ytr):
        state = {}
        imp_p = SimpleImputer(strategy="median"); sca_p = StandardScaler()
        Xp = sca_p.fit_transform(imp_p.fit_transform(Xtr[:, phys_idx]))
        krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(Xp, ytr)
        state.update(imp_p=imp_p, sca_p=sca_p, krr=krr)
        resid = ytr - krr.predict(Xp)
        imp_a = SimpleImputer(strategy="median"); sca_a = StandardScaler()
        Xa = sca_a.fit_transform(imp_a.fit_transform(Xtr))
        state["ml"] = [LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                     random_state=42+m, n_jobs=1, verbose=-1,
                                     subsample=0.8, subsample_freq=1).fit(Xa, resid)
                       for m in range(n_ml)]
        state.update(imp_a=imp_a, sca_a=sca_a, phys_idx=phys_idx)
        return state

    def predict_fn(state, Xte):
        Xp = state["sca_p"].transform(state["imp_p"].transform(Xte[:, state["phys_idx"]]))
        mu_p = state["krr"].predict(Xp)
        Xa = state["sca_a"].transform(state["imp_a"].transform(Xte))
        preds = np.array([m.predict(Xa) for m in state["ml"]])
        return mu_p + preds.mean(axis=0), preds.std(axis=0)

    pics, mpiws = [], []
    for seed in seeds:
        cv = list(KFold(5, shuffle=True, random_state=seed).split(X))
        res = conformal_intervals_cv(X, y, cv, fit_fn, predict_fn, alpha=alpha, seed=seed)
        m = compute_picp_mpiw(y, res["oof_lower"], res["oof_upper"], nominal=1-alpha)
        pics.append(m["PICP"]); mpiws.append(m["MPIW"])
        print(f"    seed={seed}: PICP={m['PICP']:.3f} MPIW={m['MPIW']:.4f}", flush=True)
    return np.array(pics), np.array(mpiws)


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  Multi-seed PICP (conformal 稳定性, N_SEEDS=5)", flush=True)
    print("=" * 60, flush=True)

    rows = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        pics, mpiws = run_multiseed_picp(df, target)
        row = {
            "target": target,
            "nominal": 0.80,
            "PICP_mean": float(pics.mean()),
            "PICP_std": float(pics.std(ddof=1)),
            "PICP_min": float(pics.min()),
            "PICP_max": float(pics.max()),
            "all_meet_nominal": bool((pics >= 0.80).all()),
            "MPIW_mean": float(mpiws.mean()),
        }
        rows.append(row)
        print(f"  → PICP={pics.mean():.3f}±{pics.std(ddof=1):.3f} "
              f"[{pics.min():.3f},{pics.max():.3f}] "
              f"(名义0.80, 全部达标: {(pics>=0.80).all()})", flush=True)

    df_out = pd.DataFrame(rows)
    out = METRICS_DIR / "multiseed_picp.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)


if __name__ == "__main__":
    main()
