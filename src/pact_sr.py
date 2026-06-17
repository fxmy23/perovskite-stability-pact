"""
================================================================
PACT-SR: Symbolic-Regression Anchored Conformal-Trust (Step 4)
================================================================
PACT v2 的物理层从 KernelRidge 替换为符号回归发现的显式方程。
完整集成 conformal UQ + 多方法 AD + 统计严谨, 与 pact_v2.py 同构。

★ 论文创新主线:
  物理锚点 = SR 发现的可解释方程 (非黑盒 KernelRidge)
  ML 残差 = LightGBM 集成
  不确定性 = ensemble σ (排序) + split conformal (定宽, PICP≥1-α)
  应用域 = σ / kNN / leverage 三方法对照

依赖: gplearn, scikit-learn, lightgbm, scipy, numpy, pandas
      + src.conformal, src.stats_eval, src.ad_methods

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score,
)
from gplearn.genetic import SymbolicRegressor

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML
from src.conformal import (
    conformal_intervals_cv, compute_picp_mpiw, compute_ece_by_uncertainty,
)
from src.stats_eval import (
    regression_metrics, bootstrap_ci, uncertainty_error_correlation,
    enrichment_factor,
)
from src.ad_methods import compare_ad_methods
from src.sr_physics_layer import make_sr_model

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05
ALPHA = 0.2

# SR 配置 (Step 1 探索选定: basic_sqrt_p0.001, 形成能全量 R²=0.42)
SR_FUNCTION_SET = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1}
SR_PARSIMONY = 0.001

phys_idx_global = None


def compute_daf(y_true, y_pred, thresh=STABLE_THRESH, top_frac=0.1):
    truly_stable = (y_true < thresh)
    base_rate = truly_stable.mean()
    if base_rate == 0:
        return float("nan")
    n_top = max(1, int(len(y_pred) * top_frac))
    top_idx = np.argsort(y_pred)[:n_top]
    return float(truly_stable[top_idx].mean() / base_rate)


def _make_fit_fn(n_ml_models=5, sr_function_set=None, sr_parsimony=SR_PARSIMONY):
    """闭包: fit_fn(X_tr, y_tr) -> state (SR 物理层 + ML 残差集成)。"""

    def fit_fn(X_tr, y_tr):
        state = {}
        # ---- SR 物理层 (替代 KernelRidge) ----
        imp_p = SimpleImputer(strategy="median"); sca_p = StandardScaler()
        Xp = sca_p.fit_transform(imp_p.fit_transform(X_tr[:, phys_idx_global]))
        sr = make_sr_model(sr_function_set, sr_parsimony, seed=42)
        sr.fit(Xp, y_tr)
        state.update(imp_p=imp_p, sca_p=sca_p, sr=sr)
        state["sr_equation"] = str(sr._program)
        resid = y_tr - sr.predict(Xp)

        # ---- ML 残差集成 ----
        imp_a = SimpleImputer(strategy="median"); sca_a = StandardScaler()
        Xa = sca_a.fit_transform(imp_a.fit_transform(X_tr))
        ml_models = []
        for m in range(n_ml_models):
            lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42 + m, n_jobs=1, verbose=-1,
                                subsample=0.8, subsample_freq=1)
            lgb.fit(Xa, resid)
            ml_models.append(lgb)
        state.update(imp_a=imp_a, sca_a=sca_a, ml_models=ml_models)
        return state

    return fit_fn


def _make_predict_fn():
    def predict_fn(state, X_te):
        Xp = state["sca_p"].transform(state["imp_p"].transform(X_te[:, phys_idx_global]))
        mu_p = state["sr"].predict(Xp)
        Xa = state["sca_a"].transform(state["imp_a"].transform(X_te))
        preds = np.array([m.predict(Xa) for m in state["ml_models"]])
        return mu_p + preds.mean(axis=0), preds.std(axis=0)

    return predict_fn


def run_pact_sr(df, target, n_splits=5, n_ml_models=5, random_state=42,
                alpha=ALPHA, cal_frac=0.2,
                sr_function_set=None, sr_parsimony=SR_PARSIMONY):
    global phys_idx_global

    feat_cols = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]
    phys_idx_global = phys_idx
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X))
    n = len(y)

    print(f"[PACT-SR] {target}, n={n}", flush=True)
    print(f"          物理层=SR (fs={sr_function_set}, par={sr_parsimony}), "
          f"ML层={len(feat_cols)}维, conformal α={alpha}", flush=True)

    fit_fn = _make_fit_fn(n_ml_models, sr_function_set, sr_parsimony)
    predict_fn = _make_predict_fn()

    oof_mu = np.full(n, np.nan)
    oof_sigma = np.full(n, np.nan)
    oof_mu_p = np.full(n, np.nan)
    sr_eqs = []
    for tr, te in cv_splits:
        state = fit_fn(X[tr], y[tr])
        mu_te, sigma_te = predict_fn(state, X[te])
        oof_mu[te] = mu_te
        oof_sigma[te] = sigma_te
        Xp = state["sca_p"].transform(state["imp_p"].transform(X[te][:, phys_idx]))
        oof_mu_p[te] = state["sr"].predict(Xp)
        sr_eqs.append(state["sr_equation"])
    print(f"  点预测: R²={r2_score(y, oof_mu):.4f} "
          f"MAE={mean_absolute_error(y, oof_mu):.4f}", flush=True)

    # conformal 区间
    conf = conformal_intervals_cv(X, y, cv_splits, fit_fn, predict_fn,
                                  alpha=alpha, cal_frac=cal_frac, seed=random_state)
    interval_m = compute_picp_mpiw(y, conf["oof_lower"], conf["oof_upper"], nominal=1 - alpha)
    print(f"  Conformal PICP={interval_m['PICP']:.3f} MPIW={interval_m['MPIW']:.4f}", flush=True)

    ece_res = compute_ece_by_uncertainty(y, conf["oof_lower"], conf["oof_upper"],
                                         oof_sigma, nominal=1 - alpha, n_bins=10)

    # 物理贡献 (SR 物理层独立 R²)
    base_r2 = float(r2_score(y, oof_mu_p))
    total_r2 = float(r2_score(y, oof_mu))
    ml_increment = total_r2 - base_r2
    print(f"  SR物理层独立 R²={base_r2:.4f}; ML残差增量 +{ml_increment:.4f} → 总 {total_r2:.4f}", flush=True)

    # AD 多方法对照
    ad_rows, trust_sigma, _, _ = compare_ad_methods(
        y, oof_mu, oof_sigma, X, cv_splits, knn_k=5, knn_pct=95, pca_nc=20)

    # 统计
    reg = regression_metrics(y, oof_mu)
    r2_ci = bootstrap_ci(y, oof_mu, n_boot=1000, seed=random_state)
    unc = uncertainty_error_correlation(oof_sigma, y, oof_mu)
    print(f"  R²={reg['R2']:.4f} [{r2_ci['lo']:.4f},{r2_ci['hi']:.4f}] "
          f"σ-|err| r={unc['pearson_r']:.3f} (p={unc['pearson_p']:.1e})", flush=True)

    # SR 公式一致性 (各折)
    print(f"  各折 SR 公式:", flush=True)
    for i, eq in enumerate(sr_eqs):
        print(f"    fold{i+1}: {eq[:100]}", flush=True)

    # 分类
    cls_results = {}
    if target == "energy_above_hull":
        y_cls = (y < STABLE_THRESH).astype(int)
        score = -oof_mu
        cls_results = {
            "Precision": float(precision_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "Recall": float(recall_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "F1": float(f1_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "AUC": float(roc_auc_score(y_cls, score)),
            "AUC_PR": float(average_precision_score(y_cls, score)),
            "DAF_top10%": compute_daf(y, oof_mu, top_frac=0.10),
            "EF_top10%": enrichment_factor(y_cls, score, top_frac=0.10),
        }
        print(f"  分类: F1={cls_results['F1']:.3f} AUC={cls_results['AUC']:.3f} "
              f"DAF={cls_results['DAF_top10%']:.2f}", flush=True)

    # 持久化 OOF
    df_oof = pd.DataFrame({
        "y_true": y, "oof_mu": oof_mu, "oof_sigma": oof_sigma, "oof_mu_p": oof_mu_p,
        "conf_lower": conf["oof_lower"], "conf_upper": conf["oof_upper"],
        "abs_err": np.abs(y - oof_mu), "trust_sigma": trust_sigma,
    })
    df_oof.to_csv(METRICS_DIR / f"pact_sr_oof_{target}.csv",
                  index=False, encoding="utf-8-sig")

    return {
        "target": target,
        "R2": reg["R2"], "RMSE": reg["RMSE"], "MAE": reg["MAE"],
        "R2_CI_lo": r2_ci["lo"], "R2_CI_hi": r2_ci["hi"],
        "PICP_conformal": interval_m["PICP"], "MPIW_conformal": interval_m["MPIW"],
        "ECE": ece_res["ECE"],
        "sigma_pearson_r": unc["pearson_r"], "sigma_pearson_p": unc["pearson_p"],
        "SR_baseline_R2": base_r2, "ml_increment_R2": ml_increment,
        "SR_equation_fold1": sr_eqs[0] if sr_eqs else "",
        **cls_results,
        "AD_sigma_R2_trusted": float(ad_rows[0]["R2_trusted"]),
        "AD_sigma_R2_untrusted": float(ad_rows[0]["R2_untrusted"]),
    }


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  PACT-SR: Symbolic-Regression Anchored Conformal-Trust", flush=True)
    print("  (仅形成能: 凸包能 SR 失败, 见根因分析)", flush=True)
    print("=" * 64, flush=True)

    all_res = []
    # ★ 仅形成能: 凸包能 SR 物理层 R²=0.19 且 fold5 塌缩为常数,
    #   根因 = 凸包能 SNR=0.94(噪声主导) + 强非线性(线性R²仅0.29) +
    #   尖峰右偏分布(偏度1.0)。凸包能物理层用 KernelRidge+SHAP (见 pact_v2.py)。
    for target in ["formation_energy_per_atom"]:
        print(f"\n{'='*48}", flush=True)
        res = run_pact_sr(df, target, sr_function_set=SR_FUNCTION_SET,
                          sr_parsimony=SR_PARSIMONY)
        all_res.append(res)

    summary_rows = [{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in all_res]
    df_out = pd.DataFrame(summary_rows)
    out = METRICS_DIR / "pact_sr_results.csv"
    df_out.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out}", flush=True)


if __name__ == "__main__":
    main()
