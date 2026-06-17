"""
================================================================
LOEO 全元素外推实验 (提升1) — 强化"物理层外推价值"证据
================================================================
背景: 5 元素预实验发现 LOEO 外推下 SR+ML(0.422) > 纯ML(0.373)。
本模块扩展到全部有足够样本的 A 位元素, 给出完整外推对比:
  纯ML vs KRR+ML vs SR+ML, 每个元素 + 汇总。

★ 增量保存: 每跑完一个元素立即写盘, 断点续跑。

依赖: gplearn, scikit-learn, lightgbm, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML
from src.sr_physics_layer import make_sr_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
LOEO_DIR = METRICS_DIR / "loeo_sr_full"
LOEO_DIR.mkdir(parents=True, exist_ok=True)


def loeo_one_element(X, y, phys_idx, mask, n_ml=3):
    """留一元素: mask=True 为训练, mask=False 为该元素测试集。"""
    Xtr, ytr = X[mask], y[mask]
    Xte, yte = X[~mask], y[~mask]
    if len(yte) < 5 or len(set(yte)) < 2:
        return None

    imp_a = SimpleImputer(strategy="median"); sca_a = StandardScaler()
    Xtr_a = sca_a.fit_transform(imp_a.fit_transform(Xtr))
    Xte_a = sca_a.transform(imp_a.transform(Xte))
    imp_p = SimpleImputer(strategy="median"); sca_p = StandardScaler()
    Xtr_p = sca_p.fit_transform(imp_p.fit_transform(Xtr[:, phys_idx]))
    Xte_p = sca_p.transform(imp_p.transform(Xte[:, phys_idx]))

    # 纯 ML
    preds = []
    for m in range(n_ml):
        lgb = LGBMRegressor(n_estimators=200, random_state=42+m, n_jobs=1, verbose=-1).fit(Xtr_a, ytr)
        preds.append(lgb.predict(Xte_a))
    r2_ml = r2_score(yte, np.mean(preds, axis=0))

    # KRR 物理层 + ML 残差
    krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(Xtr_p, ytr)
    resid = ytr - krr.predict(Xtr_p)
    preds = []
    for m in range(n_ml):
        lgb = LGBMRegressor(n_estimators=200, random_state=42+m, n_jobs=1, verbose=-1).fit(Xtr_a, resid)
        preds.append(lgb.predict(Xte_a))
    r2_krr_ml = r2_score(yte, krr.predict(Xte_p) + np.mean(preds, axis=0))
    r2_krr_only = r2_score(yte, krr.predict(Xte_p))

    # SR 物理层 + ML 残差
    sr = make_sr_model(parsimony=0.001, seed=42)
    sr.fit(Xtr_p, ytr)
    resid = ytr - sr.predict(Xtr_p)
    preds = []
    for m in range(n_ml):
        lgb = LGBMRegressor(n_estimators=200, random_state=42+m, n_jobs=1, verbose=-1).fit(Xtr_a, resid)
        preds.append(lgb.predict(Xte_a))
    r2_sr_ml = r2_score(yte, sr.predict(Xte_p) + np.mean(preds, axis=0))
    r2_sr_only = r2_score(yte, sr.predict(Xte_p))

    return {
        "n_test": int(len(yte)),
        "r2_pure_ml": float(r2_ml),
        "r2_krr_ml": float(r2_krr_ml),
        "r2_sr_ml": float(r2_sr_ml),
        "r2_krr_only": float(r2_krr_only),
        "r2_sr_only": float(r2_sr_only),
    }


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat) if c in PHYS_FEATURES]
    X = df[feat].values
    y = df["formation_energy_per_atom"].values
    a_sites = df["a_site_element"].values

    from collections import Counter
    counts = Counter(a_sites)
    # 选样本>=50 的元素 (足够算 R²)
    elems = sorted([e for e, n in counts.items() if n >= 50])
    print("=" * 72, flush=True)
    print(f"  LOEO 全元素外推 ({len(elems)} 个 A 位元素, 样本>=50)", flush=True)
    print("=" * 72, flush=True)

    # 断点续跑
    done = set()
    for f in LOEO_DIR.glob("elem_*.json"):
        try:
            done.add(f.stem.split("_", 1)[1])
        except Exception:
            pass
    todo = [e for e in elems if e not in done]
    print(f"  已完成 {len(done)}, 待跑 {len(todo)}: {todo[:10]}{'...' if len(todo)>10 else ''}", flush=True)

    for i, elem in enumerate(todo):
        mask = a_sites != elem
        print(f"  [{i+1+len(done)}/{len(elems)}] {elem} (n_test={counts[elem]}) ...", flush=True, end="")
        try:
            res = loeo_one_element(X, y, phys_idx, mask)
            if res is None:
                print(" 跳过(样本不足)", flush=True)
                continue
            res["element"] = elem
            (LOEO_DIR / f"elem_{elem}.json").write_text(
                json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print(f" ML={res['r2_pure_ml']:+.2f} KRR+ML={res['r2_krr_ml']:+.2f} "
                  f"SR+ML={res['r2_sr_ml']:+.2f} [SAVE]", flush=True)
        except Exception as e:
            print(f" ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ===== 汇总 =====
    print("\n" + "=" * 72, flush=True)
    print("  LOEO 汇总 (各方法跨元素平均 R²)", flush=True)
    print("=" * 72, flush=True)
    rows = []
    for f in sorted(LOEO_DIR.glob("elem_*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        print("  无结果", flush=True)
        return
    df_res = pd.DataFrame(rows)

    # 平均 (排除极端单元素 R² 的稳健性: 同时报 mean 和 median)
    for col in ["r2_pure_ml", "r2_krr_ml", "r2_sr_ml", "r2_krr_only", "r2_sr_only"]:
        if col in df_res.columns:
            vals = df_res[col].dropna()
            print(f"  {col:14s}: mean={vals.mean():+.3f}  median={vals.median():+.3f}  "
                  f"(n={len(vals)})", flush=True)

    # 关键对比: SR+ML vs 纯ML 的逐元素胜率
    if "r2_sr_ml" in df_res.columns and "r2_pure_ml" in df_res.columns:
        sr_better = (df_res["r2_sr_ml"] > df_res["r2_pure_ml"]).sum()
        print(f"\n  SR+ML 优于 纯ML 的元素数: {sr_better}/{len(df_res)} "
              f"({sr_better/len(df_res)*100:.0f}%)", flush=True)
    if "r2_krr_ml" in df_res.columns and "r2_pure_ml" in df_res.columns:
        krr_better = (df_res["r2_krr_ml"] > df_res["r2_pure_ml"]).sum()
        print(f"  KRR+ML 优于 纯ML 的元素数: {krr_better}/{len(df_res)} "
              f"({krr_better/len(df_res)*100:.0f}%)", flush=True)

    df_res.to_csv(METRICS_DIR / "loeo_sr_full_results.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] loeo_sr_full_results.csv ({len(df_res)} 元素)", flush=True)


if __name__ == "__main__":
    main()
