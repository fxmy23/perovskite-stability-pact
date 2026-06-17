"""
================================================================
形成能物理分解: PACT-Madelung (准理论创新)
================================================================
E_f = α·E_Madelung(物理解析) + ML_residual(学共价+畸变+电子)

理论: 见 docs/madelung_theory.md
文献: RSC Adv 2021, Inorg Chem generalized Kapustinskii

Madelung能量近似 (无需DFT结构):
  理想立方ABO3 Madelung常数 M≈1.716
  U_Madelung ∝ -M·z²/a, a≈(r_A+r_O)/√2
  电荷: z_A=+2, z_B=+4, z_O=-2 (典型价态)

★ pre-debug: 验证Madelung能量与形成能的相关性 (应负相关)
★ post-debug: 验证无泄露 + 物理层R²合理 + 总R²持平

依赖: scikit-learn, lightgbm, numpy, pandas
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Madelung 常数 (理想立方 ABO3)
MADELUONG_M = 1.716
# 典型价态
Z_A, Z_B, Z_O = 2, 4, -2
R_O = 1.40  # O2- 6配位半径


def compute_madelung_energy(df):
    """
    计算每个ABO3的Madelung能量近似。
    U ∝ -M · z_eff² / a
    a (晶格常数估计) ≈ 2*(r_B + r_O) (B-O键长×2, 立方)
    z_eff² = |z_A·z_O| + |z_B·z_O| 的加权 (简化)
    返回: U_madelung (无量纲, 因我们只关心相关性)
    """
    r_a = df["phys_a_site_radius"].values
    r_b = df["phys_b_site_radius"].values
    # 晶格常数估计: 立方ABO3, a≈2(r_B+r_O), 但也受A位约束
    a = 2.0 * (r_b + R_O)
    # 有效电荷平方 (简化: 用价态乘积)
    # ABO3: 1个A(+2), 1个B(+4), 3个O(-2)
    # 静电能主导项 ~ z_A*|z_O| + z_B*|z_O|
    z_eff_sq = abs(Z_A * Z_O) + abs(Z_B * Z_O)  # = 4+8 = 12 (常数)
    # Madelung 能量 (负, 越大|U|越稳定)
    U = -MADELUONG_M * z_eff_sq / a
    return U


def pre_debug(df, y):
    """验证 Madelung 能量与形成能的相关性。"""
    U = compute_madelung_energy(df)
    valid = ~np.isnan(U) & ~np.isnan(y)
    from scipy.stats import pearsonr, spearmanr
    r_p, p_p = pearsonr(U[valid], y[valid])
    r_s, _ = spearmanr(U[valid], y[valid])
    print(f"[pre-debug] Madelung能量 vs 形成能:", flush=True)
    print(f"  Pearson r={r_p:.3f} (p={p_p:.1e})", flush=True)
    print(f"  Spearman r={r_s:.3f}", flush=True)
    print(f"  U范围: [{np.nanmin(U):.3f}, {np.nanmax(U):.3f}]", flush=True)
    # 单独R² (线性映射后)
    U_v = U[valid].reshape(-1, 1)
    y_v = y[valid]
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(U_v, y_v)
    r2 = lr.score(U_v, y_v)
    print(f"  Madelung线性映射 R²={r2:.4f} (物理层独立解释力)", flush=True)
    print(f"  → {'✓有信息' if r2>0.05 else '⚠信息弱'} (期望>0.1, 因离子项主导形成能)", flush=True)
    return U, r2


def run_madelung_decomposition_cv(df, target, n_splits=5, n_ml=5, seed=42):
    """CV评估: Madelung物理层 + ML残差 vs 纯ML vs KRR物理层。"""
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat) if c in PHYS_FEATURES]
    X = df[feat].values
    y = df[target].values
    U_full = compute_madelung_energy(df)
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(y)

    oof_madelung = np.full(n, np.nan)
    oof_mad_ml = np.full(n, np.nan)
    oof_ml = np.full(n, np.nan)

    for fold, (tr, te) in enumerate(cv.split(X)):
        # Madelung 物理层: 用训练折 fit α (线性缩放)
        U_tr, U_te = U_full[tr], U_full[te]
        valid_tr = ~np.isnan(U_tr)
        if valid_tr.sum() > 10:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(U_tr[valid_tr].reshape(-1,1), y[tr][valid_tr])
            mu_p_te = lr.predict(U_te.reshape(-1,1))
            mu_p_tr = lr.predict(U_tr.reshape(-1,1))
            resid_tr = y[tr] - mu_p_tr
        else:
            mu_p_te = np.zeros(len(te)); resid_tr = y[tr].copy()
        oof_madelung[te] = mu_p_te

        # ML 残差
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        preds = []
        for m in range(n_ml):
            lgb = LGBMRegressor(n_estimators=450, num_leaves=39, learning_rate=0.078,
                                max_depth=10, min_child_samples=17, subsample=0.902,
                                colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278,
                                random_state=42+m, n_jobs=1, verbose=-1,
                                subsample_freq=1).fit(Xtr, resid_tr)
            preds.append(lgb.predict(Xte))
        oof_mad_ml[te] = mu_p_te + np.mean(preds, axis=0)

        # 纯ML对照
        preds_ml = []
        for m in range(n_ml):
            lgb = LGBMRegressor(n_estimators=450, num_leaves=39, learning_rate=0.078,
                                max_depth=10, min_child_samples=17, subsample=0.902,
                                colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278,
                                random_state=42+m, n_jobs=1, verbose=-1,
                                subsample_freq=1).fit(Xtr, y[tr])
            preds_ml.append(lgb.predict(Xte))
        oof_ml[te] = np.mean(preds_ml, axis=0)
        print(f"  fold{fold+1} done", flush=True)

    r2_mad = r2_score(y, oof_madelung)
    r2_mad_ml = r2_score(y, oof_mad_ml)
    r2_ml = r2_score(y, oof_ml)
    print(f"\n  {target} 结果:", flush=True)
    print(f"    Madelung物理层独立 R²={r2_mad:.4f}", flush=True)
    print(f"    Madelung+ML残差      R²={r2_mad_ml:.4f}", flush=True)
    print(f"    纯ML                 R²={r2_ml:.4f}", flush=True)
    print(f"    (Madelung+ML - 纯ML = {r2_mad_ml-r2_ml:+.4f})", flush=True)
    return {"target": target, "r2_madelung_only": r2_mad,
            "r2_madelung_plus_ml": r2_mad_ml, "r2_pure_ml": r2_ml}


def post_debug(df, target, res):
    """post-debug: 无泄露 + 合理性。"""
    print(f"\n[post-debug] {target}:", flush=True)
    # Madelung物理层R²应>0 (有信息) 但<0.6 (粗近似)
    r2_mad = res["r2_madelung_only"]
    if 0 < r2_mad < 0.6:
        print(f"  ✓ Madelung物理层R²={r2_mad:.4f} 在合理区间 (0, 0.6)", flush=True)
    elif r2_mad <= 0:
        print(f"  ⚠ Madelung物理层R²={r2_mad:.4f} ≤0, 物理近似太粗", flush=True)
    else:
        print(f"  ⚠ Madelung物理层R²={r2_mad:.4f} >0.6, 检查是否泄露", flush=True)
    # 总R²应与纯ML持平 (±0.01)
    diff = res["r2_madelung_plus_ml"] - res["r2_pure_ml"]
    if abs(diff) < 0.02:
        print(f"  ✓ Madelung+ML 与纯ML持平 (diff={diff:+.4f}), 物理层不损害精度", flush=True)
    else:
        print(f"  ⚠ 差异 {diff:+.4f} 较大", flush=True)


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  形成能物理分解 PACT-Madelung (准理论创新)", flush=True)
    print("=" * 64, flush=True)

    # pre-debug
    y = df["formation_energy_per_atom"].values
    pre_debug(df, y)

    # CV评估 (仅形成能, 因Madelung对凸包能意义不同)
    res = run_madelung_decomposition_cv(df, "formation_energy_per_atom")
    # post-debug
    post_debug(df, "formation_energy_per_atom", res)

    # 持久化
    pd.DataFrame([res]).to_csv(METRICS_DIR / "madelung_decomposition.csv",
                                index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] madelung_decomposition.csv", flush=True)


if __name__ == "__main__":
    main()
