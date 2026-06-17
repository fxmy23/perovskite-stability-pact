"""
================================================================
PACT-Final: 统一连贯管线 (四项必修合并)
================================================================
★ 终极架构 (基于 four_fixes_justification.md 论证):

  物理层(KRR) + ML残差(stacking LGB+XGB+HistGBT) → 点预测 μ (R²0.918)  [项1]
  CQR解耦 → 区间 (ECE改善, 与点预测独立)                          [项2]
  conformal + AD三方法 → 不确定性+应用域

设计要点 (文献佐证):
- stacking放残差层: Residual-Aware Stacking (SSRN 2025), 不绕过物理层
- 点/区间解耦: conformal灵活性 (Romano 2019 NeurIPS CQR)
- 嵌套CV无泄露: 外层5折评估, 内层3折生成stacking OOF

依赖: lightgbm/xgboost/scikit-learn/numpy/pandas
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML
from src.stats_eval import regression_metrics, bootstrap_ci, uncertainty_error_correlation
from src.ad_methods import compare_ad_methods

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05
ALPHA = 0.2
LOWER_Q, UPPER_Q = 0.1, 0.9

LGB_BASE = dict(n_estimators=450, num_leaves=39, learning_rate=0.078,
                max_depth=10, min_child_samples=17, subsample=0.902,
                colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278,
                n_jobs=1, verbose=-1, subsample_freq=1)


def base_models(seed=42):
    """3个GBDT基模型 (多样性: leaf-wise/level-wise/hist)."""
    return {
        "LGB": lambda: LGBMRegressor(random_state=seed, **LGB_BASE),
        "XGB": lambda: XGBRegressor(n_estimators=450, max_depth=8, learning_rate=0.07,
                                    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5,
                                    reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=0),
        "HGB": lambda: HistGradientBoostingRegressor(max_iter=450, max_depth=8,
                                                      learning_rate=0.07, l2_regularization=1.0,
                                                      random_state=seed),
    }


def run_pact_final(df, target, n_outer=5, seed=42, alpha=ALPHA, cal_frac=0.2):
    """PACT-Final 统一管线: KRR物理层+stacking残差(点) + CQR解耦(区间) + AD."""
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat) if c in PHYS_FEATURES]
    X = df[feat].values
    y = df[target].values
    outer_cv = KFold(n_outer, shuffle=True, random_state=seed)
    n = len(y)
    factories = base_models(seed)
    names = list(factories.keys())

    # 点预测 (stacking残差)
    oof_mu = np.full(n, np.nan)
    oof_mu_p = np.full(n, np.nan)
    oof_sigma = np.full(n, np.nan)
    # CQR区间 (解耦)
    oof_cqr_lo = np.full(n, np.nan)
    oof_cqr_hi = np.full(n, np.nan)
    # 标准conformal对照
    oof_std_lo = np.full(n, np.nan)
    oof_std_hi = np.full(n, np.nan)

    for fold, (tr, te) in enumerate(outer_cv.split(X)):
        rng = np.random.default_rng(seed + fold)
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))
        imp_p = SimpleImputer(strategy="median"); sca_p = StandardScaler()
        Xtr_p = sca_p.fit_transform(imp_p.fit_transform(X[tr][:, phys_idx]))
        Xte_p = sca_p.transform(imp_p.transform(X[te][:, phys_idx]))

        # === 物理层 KRR (可解释锚点) ===
        krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1).fit(Xtr_p, y[tr])
        mu_p_tr = krr.predict(Xtr_p)
        mu_p_te = krr.predict(Xte_p)
        resid_tr = y[tr] - mu_p_tr
        oof_mu_p[te] = mu_p_te

        # === ML残差层: stacking (项1) ===
        # 内层CV生成基模型OOF + 全量fit
        inner_cv = KFold(3, shuffle=True, random_state=seed + fold)
        meta_tr = np.zeros((len(tr), len(names)))
        meta_te = np.zeros((len(te), len(names)))
        base_preds_std = []  # for σ
        for j, nm in enumerate(names):
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in inner_cv.split(Xtr):
                m = factories[nm]().fit(Xtr[itr], resid_tr[itr])
                inner_oof[ite] = m.predict(Xtr[ite])
            meta_tr[:, j] = inner_oof
            m_full = factories[nm]().fit(Xtr, resid_tr)
            p_te = m_full.predict(Xte)
            meta_te[:, j] = p_te
            base_preds_std.append(p_te)
        # Ridge元学习器
        meta_lr = Ridge(alpha=1.0).fit(meta_tr, resid_tr)
        resid_pred_te = meta_lr.predict(meta_te)
        oof_mu[te] = mu_p_te + resid_pred_te
        # σ (基模型预测的std, 反映模型不确定性)
        oof_sigma[te] = np.std(base_preds_std, axis=0)

        # === CQR区间 (解耦, 项2): 分位数模型独立于点预测 ===
        # 切calibration (从训练集)
        perm = rng.permutation(len(tr))
        n_cal = max(20, int(len(tr) * cal_frac))
        cal_idx = perm[:n_cal]; fit_idx = perm[n_cal:]
        Xf, yf = Xtr[fit_idx], y[tr][fit_idx]
        Xc, yc = Xtr[cal_idx], y[tr][cal_idx]
        q_lo = LGBMRegressor(objective="quantile", alpha=LOWER_Q, **LGB_BASE).fit(Xf, yf)
        q_hi = LGBMRegressor(objective="quantile", alpha=UPPER_Q, **LGB_BASE).fit(Xf, yf)
        lo_te = np.minimum(q_lo.predict(Xte), q_hi.predict(Xte))
        hi_te = np.maximum(q_lo.predict(Xte), q_hi.predict(Xte))
        lo_c = np.minimum(q_lo.predict(Xc), q_hi.predict(Xc))
        hi_c = np.maximum(q_lo.predict(Xc), q_hi.predict(Xc))
        cqr_scores = np.maximum(lo_c - yc, yc - hi_c)
        rank = min(int(np.ceil((1 - alpha) * (n_cal + 1))), n_cal)
        d_cqr = float(np.sort(cqr_scores)[rank - 1])
        oof_cqr_lo[te] = lo_te - d_cqr
        oof_cqr_hi[te] = hi_te + d_cqr
        # === 标准conformal对照 (公平: 基于与点预测相同的stacking主线) ===
        # ★ 修复 C1/H1: 原版用独立Ridge预测作对照(不公平, Ridge弱于stacking)。
        #   公平对照应基于"点预测模型在calibration上的残差"。
        #   但stacking主线是物理层KRR+stacking残差, 点预测μ_p+stacking已在上面算。
        #   calibration集的点预测 = KRR(物理层) + stacking(ML层) on calibration。
        #   为避免重复stacking fit (贵), 用简化的公平对照: 单LightGBM点预测 (与stacking同家族)。
        #   这比原Ridge更接近主线精度, 是更公平的对照。
        mid_cal = LGBMRegressor(**LGB_BASE).fit(Xf, yf)
        mid_cal_pred = mid_cal.predict(Xc)
        std_scores = np.abs(yc - mid_cal_pred)
        d_std = float(np.sort(std_scores)[rank - 1])
        mid_te_pred = mid_cal.predict(Xte)
        oof_std_lo[te] = mid_te_pred - d_std
        oof_std_hi[te] = mid_te_pred + d_std

        print(f"  fold{fold+1} done", flush=True)

    # === 评估 ===
    reg = regression_metrics(y, oof_mu)
    r2_ci = bootstrap_ci(y, oof_mu, n_boot=1000, seed=seed)
    unc = uncertainty_error_correlation(oof_sigma, y, oof_mu)

    # CQR vs 标准区间
    picp_cqr = float(np.mean((y >= oof_cqr_lo) & (y <= oof_cqr_hi)))
    mpiw_cqr = float(np.mean(oof_cqr_hi - oof_cqr_lo))
    picp_std = float(np.mean((y >= oof_std_lo) & (y <= oof_std_hi)))
    mpiw_std = float(np.mean(oof_std_hi - oof_std_lo))
    # ECE (按σ分10桶)
    covered_cqr = ((y >= oof_cqr_lo) & (y <= oof_cqr_hi)).astype(float)
    covered_std = ((y >= oof_std_lo) & (y <= oof_std_hi)).astype(float)
    order = np.argsort(oof_sigma)
    bins = np.array_split(order, 10)
    ece_cqr = np.mean([abs(covered_cqr[b].mean() - (1-alpha)) for b in bins if len(b)])
    ece_std = np.mean([abs(covered_std[b].mean() - (1-alpha)) for b in bins if len(b)])

    # AD三方法
    cv_splits = list(outer_cv.split(X))
    ad_rows, trust_sigma, _, _ = compare_ad_methods(y, oof_mu, oof_sigma, X, cv_splits)

    # 物理贡献
    base_r2 = float(r2_score(y, oof_mu_p))
    total_r2 = reg["R2"]

    print(f"\n  [{target}] PACT-Final 结果:", flush=True)
    print(f"    点预测 R²={total_r2:.4f} [{r2_ci['lo']:.4f},{r2_ci['hi']:.4f}] MAE={reg['MAE']:.4f}", flush=True)
    print(f"    CQR: PICP={picp_cqr:.3f} MPIW={mpiw_cqr:.4f} ECE={ece_cqr:.3f}", flush=True)
    print(f"    标准conformal: PICP={picp_std:.3f} MPIW={mpiw_std:.4f} ECE={ece_std:.3f}", flush=True)
    print(f"    ECE改善: {(ece_std-ece_cqr)/ece_std*100:.0f}%", flush=True)
    print(f"    σ-|err| r={unc['pearson_r']:.3f} (p={unc['pearson_p']:.1e})", flush=True)
    print(f"    物理层独立R²={base_r2:.4f} (可解释锚点)", flush=True)

    # 持久化OOF (为画图)
    pd.DataFrame({
        "y_true": y, "oof_mu": oof_mu, "oof_sigma": oof_sigma, "oof_mu_p": oof_mu_p,
        "cqr_lower": oof_cqr_lo, "cqr_upper": oof_cqr_hi,
        "std_lower": oof_std_lo, "std_upper": oof_std_hi,
        "abs_err": np.abs(y - oof_mu), "trust_sigma": trust_sigma,
    }).to_csv(METRICS_DIR / f"pact_final_oof_{target}.csv", index=False, encoding="utf-8-sig")

    return {"target": target, "R2": total_r2, "R2_CI_lo": r2_ci["lo"], "R2_CI_hi": r2_ci["hi"],
            "MAE": reg["MAE"], "RMSE": reg["RMSE"],
            "PICP_cqr": picp_cqr, "MPIW_cqr": mpiw_cqr, "ECE_cqr": ece_cqr,
            "PICP_std": picp_std, "MPIW_std": mpiw_std, "ECE_std": ece_std,
            "sigma_pearson_r": unc["pearson_r"], "sigma_pearson_p": unc["pearson_p"],
            "phys_baseline_R2": base_r2,
            "AD_sigma_R2_trusted": float(ad_rows[0]["R2_trusted"]),
            "AD_sigma_R2_untrusted": float(ad_rows[0]["R2_untrusted"]),
            "AD_knn_R2_trusted": float(ad_rows[1]["R2_trusted"])}


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  PACT-Final: 统一连贯管线", flush=True)
    print("  KRR物理层 + stacking残差(点) + CQR解耦(区间) + AD", flush=True)
    print("=" * 64, flush=True)
    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        res = run_pact_final(df, target)
        all_res.append(res)
    pd.DataFrame(all_res).to_csv(METRICS_DIR / "pact_final_results.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] pact_final_results.csv + per-sample OOF", flush=True)


if __name__ == "__main__":
    main()
