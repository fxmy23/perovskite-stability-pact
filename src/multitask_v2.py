"""
================================================================
多任务学习: 形成能帮助凸包能 (算法升级)
================================================================
动机: 形成能R²=0.918(高), 凸包能R²=0.813(低)。两者物理相关(r=0.59)。
  共享底层表征, 形成能的精确预测可能提升凸包能。

实现 (stacking式多任务, 简单有效):
  Stage 1: 训练形成能模型 (已优化, R²=0.918)
  Stage 2: 把形成能预测作为额外特征, 加入凸包能模型
  → 凸包能模型借形成能信息

对照: 单任务凸包能 (无形成能特征) vs 多任务 (含形成能预测特征)
无泄露: 形成能预测用OOF (out-of-fold), 不含该样本的真实值

依赖: lightgbm, scikit-learn
作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

LGB_PARAMS = dict(n_estimators=450, num_leaves=39, learning_rate=0.078,
                  max_depth=10, min_child_samples=17, subsample=0.902,
                  colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278,
                  n_jobs=1, verbose=-1, subsample_freq=1)


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    X = df[feat].values
    y_f = df["formation_energy_per_atom"].values
    y_h = df["energy_above_hull"].values

    print("=" * 64, flush=True)
    print("  多任务学习: 形成能预测 → 凸包能", flush=True)
    print("=" * 64, flush=True)

    cv = KFold(5, shuffle=True, random_state=42)
    n = len(y_f)

    # Stage 1: 形成能OOF (无泄露)
    oof_f = np.full(n, np.nan)
    for tr, te in cv.split(X):
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        m = LGBMRegressor(random_state=42, **LGB_PARAMS).fit(Xtr, y_f[tr])
        oof_f[te] = m.predict(Xte)
    print(f"  形成能OOF R²={r2_score(y_f, oof_f):.4f}", flush=True)

    # Stage 2a: 单任务凸包能 (baseline)
    oof_h_single = np.full(n, np.nan)
    # Stage 2b: 多任务凸包能 (加入形成能OOF特征)
    oof_h_multi = np.full(n, np.nan)
    # 把形成能OOF作为额外特征
    X_aug = np.column_stack([X, oof_f.reshape(-1, 1)])

    for tr, te in cv.split(X):
        # 单任务
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        m = LGBMRegressor(random_state=42, **LGB_PARAMS).fit(Xtr, y_h[tr])
        oof_h_single[te] = m.predict(Xte)
        # 多任务 (含形成能OOF)
        imp2 = SimpleImputer(strategy="median"); sca2 = StandardScaler()
        Xtr2 = sca2.fit_transform(imp2.fit_transform(X_aug[tr]))
        Xte2 = sca2.transform(imp2.transform(X_aug[te]))
        m2 = LGBMRegressor(random_state=42, **LGB_PARAMS).fit(Xtr2, y_h[tr])
        oof_h_multi[te] = m2.predict(Xte2)
        print(f"  fold done", flush=True)

    r2_single = r2_score(y_h, oof_h_single)
    r2_multi = r2_score(y_h, oof_h_multi)
    mae_single = mean_absolute_error(y_h, oof_h_single)
    mae_multi = mean_absolute_error(y_h, oof_h_multi)
    print(f"\n  [结果] 凸包能:", flush=True)
    print(f"    单任务 (无形成能特征):  R²={r2_single:.4f} MAE={mae_single:.4f}", flush=True)
    print(f"    多任务 (含形成能OOF):   R²={r2_multi:.4f} MAE={mae_multi:.4f}", flush=True)
    print(f"    提升: R² {r2_multi-r2_single:+.4f}, MAE {mae_multi-mae_single:+.4f}", flush=True)
    if r2_multi > r2_single:
        print(f"    ✓ 多任务(形成能→凸包能) 有效", flush=True)
    else:
        print(f"    ⚠ 多任务未提升凸包能", flush=True)

    # 持久化
    pd.DataFrame([{
        "r2_hull_single": r2_single, "r2_hull_multi": r2_multi,
        "improvement": r2_multi - r2_single,
        "mae_single": mae_single, "mae_multi": mae_multi,
        "r2_formation_oof": r2_score(y_f, oof_f),
    }]).to_csv(METRICS_DIR / "multitask_results.csv",
                index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] multitask_results.csv", flush=True)


if __name__ == "__main__":
    main()
