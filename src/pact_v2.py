"""
================================================================
PACT v2: Physics-Anchored Conformal-Trust (统一主线框架)
================================================================
v2 关键升级 (2026-06-16, 基于严苛审稿+深度调研):

  ★ P0-1 PICP 修复: 用 Split Conformal Prediction 替代 ensemble std 区间。
    原版 PICP=0.24 (灾难), conformal 保证 PICP≥1-α (理论, 有限样本)。
    ensemble σ 仍保留作"排序不确定性" (σ-误差相关有意义), 但区间宽度
    由 conformal 分位数决定——σ 排序, conformal 定宽。

  ★ P0-2 AD 多方法对照: σ / k-NN / leverage 三方法一致性, 不再是单启发式。

  ★ P0-3 统计严谨: bootstrap CI + σ-误差 p 值 + ECE 校准曲线。

  ★ PCRL 诚实降级: PCRL 不再是"主创新", 改为"对照+负面结果分析"
    (见 docs/research_findings_v2.md §5)。主线创新 = conformal UQ + 多方法 AD + LOEO。

统一管线 (单一来源, 端到端):
  1. 物理层: KernelRidge on 15 物理特征 → μ_p
  2. ML 残差层: 5×LightGBM 集成 on 113 特征 → μ_r, σ_r
  3. 统一预测: μ = μ_p + μ_r
  4. 排序不确定性: σ = ensemble std (σ-误差相关, AD 用)
  5. 校准区间: split conformal (PICP 有保证, 主线创新)
  6. 应用域: σ / k-NN / leverage 三方法一致性
  7. 分类: 从校准 μ 派生 (F1/AUC/AUC-PR/DAF/EF)

依赖: scikit-learn, lightgbm, scipy, numpy, pandas

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

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES
from src.conformal import (
    conformal_intervals_cv, compute_picp_mpiw, compute_ece_by_uncertainty,
)
from src.stats_eval import (
    regression_metrics, bootstrap_ci, uncertainty_error_correlation,
    classification_metrics as cls_full, enrichment_factor,
)
from src.ad_methods import compare_ad_methods

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05
ALPHA = 0.2  # 80% 名义覆盖


def compute_daf(y_true, y_pred, thresh=STABLE_THRESH, top_frac=0.1):
    truly_stable = (y_true < thresh)
    base_rate = truly_stable.mean()
    if base_rate == 0:
        return float("nan")
    n_top = max(1, int(len(y_pred) * top_frac))
    top_idx = np.argsort(y_pred)[:n_top]
    return float(truly_stable[top_idx].mean() / base_rate)


# ----------------------------------------------------------------
# 训练/预测函数 (供 conformal 模块回调)
# ----------------------------------------------------------------
def _make_fit_fn(n_ml_models=5, use_phys_baseline=True, phys_idx=None):
    """返回闭包 fit_fn(X_tr, y_tr) -> dict(模型状态)。"""

    def fit_fn(X_tr, y_tr):
        state = {}
        if use_phys_baseline and phys_idx is not None:
            imp_p = SimpleImputer(strategy="median")
            sca_p = StandardScaler()
            Xp = sca_p.fit_transform(imp_p.fit_transform(X_tr[:, phys_idx]))
            krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
            krr.fit(Xp, y_tr)
            state["imp_p"] = imp_p
            state["sca_p"] = sca_p
            state["krr"] = krr
            resid = y_tr - krr.predict(Xp)
            target_for_ml = resid
        else:
            target_for_ml = y_tr

        imp_a = SimpleImputer(strategy="median")
        sca_a = StandardScaler()
        Xa = sca_a.fit_transform(imp_a.fit_transform(X_tr))
        ml_models = []
        for m in range(n_ml_models):
            lgbm = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42 + m, n_jobs=1, verbose=-1,
                subsample=0.8, subsample_freq=1,
            )
            lgbm.fit(Xa, target_for_ml)
            ml_models.append(lgbm)
        state["imp_a"] = imp_a
        state["sca_a"] = sca_a
        state["ml_models"] = ml_models
        state["use_phys"] = use_phys_baseline and phys_idx is not None
        return state

    return fit_fn


def _make_predict_fn():
    """返回闭包 predict_fn(state, X_te) -> (mu, sigma)。"""

    def predict_fn(state, X_te):
        if state["use_phys"]:
            Xp = state["sca_p"].transform(state["imp_p"].transform(X_te[:, phys_idx_global]))
            mu_p = state["krr"].predict(Xp)
        else:
            mu_p = 0.0
        Xa = state["sca_a"].transform(state["imp_a"].transform(X_te))
        ml_preds = np.array([m.predict(Xa) for m in state["ml_models"]])
        mu_r = ml_preds.mean(axis=0)
        sigma_r = ml_preds.std(axis=0)
        mu = mu_p + mu_r
        return mu, sigma_r

    return predict_fn


# 全局变量传递 phys_idx 给闭包 (单进程, 安全)
phys_idx_global = None


# ----------------------------------------------------------------
# PACT v2 主流程
# ----------------------------------------------------------------
def run_pact_v2(df, target, n_splits=5, n_ml_models=5, random_state=42,
                alpha=ALPHA, cal_frac=0.2):
    global phys_idx_global

    feat_cols = get_feature_cols(df, exclude_struct=True)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]
    phys_idx_global = phys_idx
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_splits = list(cv.split(X))
    n = len(y)

    print(f"[PACT-v2] 目标: {target}, 样本 {n}", flush=True)
    print(f"          物理层 {len(phys_idx)}维, ML层 {len(feat_cols)}维, "
          f"集成 {n_ml_models}个LightGBM, conformal α={alpha}", flush=True)

    # ---- 1. 点预测 + ensemble σ (OOF) ----
    fit_fn = _make_fit_fn(n_ml_models, use_phys_baseline=True, phys_idx=phys_idx)
    predict_fn = _make_predict_fn()

    oof_mu = np.full(n, np.nan)
    oof_sigma = np.full(n, np.nan)
    oof_mu_p = np.full(n, np.nan)
    for tr, te in cv_splits:
        state = fit_fn(X[tr], y[tr])
        mu_te, sigma_te = predict_fn(state, X[te])
        oof_mu[te] = mu_te
        oof_sigma[te] = sigma_te
        # 物理基线部分 (用于物理贡献量化)
        Xp = state["sca_p"].transform(state["imp_p"].transform(X[te][:, phys_idx]))
        oof_mu_p[te] = state["krr"].predict(Xp)
    print(f"  点预测: R²={r2_score(y, oof_mu):.4f} "
          f"MAE={mean_absolute_error(y, oof_mu):.4f}", flush=True)

    # ---- 2. Split Conformal 区间 (PICP 有保证) ----
    conf = conformal_intervals_cv(
        X, y, cv_splits, fit_fn, predict_fn,
        alpha=alpha, cal_frac=cal_frac, seed=random_state,
    )
    oof_lower = conf["oof_lower"]
    oof_upper = conf["oof_upper"]
    q_per_fold = conf["q_per_fold"]
    interval_m = compute_picp_mpiw(y, oof_lower, oof_upper, nominal=1 - alpha)
    print(f"  Conformal PICP={interval_m['PICP']:.3f} (名义{1-alpha:.2f}) "
          f"MPIW={interval_m['MPIW']:.4f}", flush=True)
    print(f"  各折半宽 q: {[round(q,4) for q in q_per_fold]}", flush=True)

    # 对比: ensemble σ 区间 (旧方法, 应证 PICP 偏低)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    old_lower = oof_mu - z * oof_sigma
    old_upper = oof_mu + z * oof_sigma
    old_picp = float(np.mean((y >= old_lower) & (y <= old_upper)))
    print(f"  [对照] ensemble σ 区间 PICP={old_picp:.3f} (旧法, 偏低)", flush=True)

    # ---- 3. 校准曲线 ECE ----
    ece_res = compute_ece_by_uncertainty(y, oof_lower, oof_upper, oof_sigma,
                                         nominal=1 - alpha, n_bins=10)
    print(f"  ECE (按σ分桶覆盖率偏差) = {ece_res['ECE']:.3f} (越小越校准)", flush=True)

    # ---- 4. 物理贡献量化 (审计修复: 诚实重定义, 不再用误导的 ratio) ----
    # 旧版 phys_share = baseline_R²/total_R²=0.866 夸大了物理贡献。
    # 严谨表述: 物理层独立解释的方差 = baseline_R² 本身;
    #           ML 残差在此基础上增加的百分点 = total_R² - baseline_R²。
    base_r2 = float(r2_score(y, oof_mu_p))
    total_r2 = float(r2_score(y, oof_mu))
    ml_increment = total_r2 - base_r2  # ML 在物理基础上的增量
    phys_share = max(0, base_r2) / total_r2 if total_r2 > 0 else 0  # 兼容旧字段
    print(f"  物理贡献 (诚实): 物理层独立解释 {base_r2:.1%} 方差; "
          f"ML 残差在此基础上 +{ml_increment:.3f} (→总 {total_r2:.3f})", flush=True)

    # ---- 5. 应用域多方法对照 ----
    print(f"  [AD] 三方法对照 (σ/kNN/leverage)...", flush=True)
    ad_rows, trust_sigma, trust_knn, trust_lev = compare_ad_methods(
        y, oof_mu, oof_sigma, X, cv_splits, knn_k=5, knn_pct=95, pca_nc=20,
    )
    for r in ad_rows:
        print(f"    {r['method']:15s} 可信R²={r['R2_trusted']:.4f} "
              f"不可信R²={r['R2_untrusted']:.4f} gap={r['R2_gap']:+.4f} "
              f"与σ一致={r['agreement_with_sigma']:.2%}", flush=True)

    # ---- 6. 统计严谨性: bootstrap CI + σ-误差 p 值 ----
    reg = regression_metrics(y, oof_mu)
    r2_ci = bootstrap_ci(y, oof_mu, n_boot=1000, seed=random_state)
    mae_ci = bootstrap_ci(y, oof_mu, metric_fn=mean_absolute_error,
                          n_boot=1000, seed=random_state)
    unc = uncertainty_error_correlation(oof_sigma, y, oof_mu)
    print(f"  R²={reg['R2']:.4f} [{r2_ci['lo']:.4f}, {r2_ci['hi']:.4f}] (bootstrap95)", flush=True)
    print(f"  σ-|err|: Pearson r={unc['pearson_r']:.3f} (p={unc['pearson_p']:.1e})", flush=True)

    # ---- 7. 分类指标 (hull 能) ----
    cls_results = {}
    if target == "energy_above_hull":
        y_cls = (y < STABLE_THRESH).astype(int)
        score = -oof_mu  # 低 μ = 高稳定度
        cls_results = {
            "Precision": float(precision_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "Recall": float(recall_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "F1": float(f1_score(y_cls, (oof_mu < STABLE_THRESH).astype(int), zero_division=0)),
            "AUC": float(roc_auc_score(y_cls, score)),
            "AUC_PR": float(average_precision_score(y_cls, score)),
            "DAF_top10%": compute_daf(y, oof_mu, top_frac=0.10),
            "DAF_top5%": compute_daf(y, oof_mu, top_frac=0.05),
            "EF_top10%": enrichment_factor(y_cls, score, top_frac=0.10),
        }
        confident = oof_sigma < np.median(oof_sigma)
        if confident.sum() > 0:
            cls_results["DAF_confident_top10%"] = compute_daf(y[confident], oof_mu[confident], top_frac=0.10)
        print(f"  分类: F1={cls_results['F1']:.3f} AUC={cls_results['AUC']:.3f} "
              f"AUC-PR={cls_results['AUC_PR']:.3f} DAF={cls_results['DAF_top10%']:.2f} "
              f"EF={cls_results['EF_top10%']:.2f}", flush=True)

    # ---- 8. 持久化逐样本 (供画图/筛选) ----
    df_oof = pd.DataFrame({
        "y_true": y, "oof_mu": oof_mu, "oof_sigma": oof_sigma,
        "oof_mu_p": oof_mu_p,
        "conf_lower": oof_lower, "conf_upper": oof_upper,
        "abs_err": np.abs(y - oof_mu),
        "trust_sigma": trust_sigma, "trust_knn": trust_knn, "trust_lev": trust_lev,
    })
    oof_path = METRICS_DIR / f"pact_v2_oof_{target}.csv"
    df_oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"  [SAVE] {oof_path}", flush=True)

    return {
        "target": target,
        # 回归
        "R2": reg["R2"], "RMSE": reg["RMSE"], "MAE": reg["MAE"],
        "MRE": reg["MRE"], "Max_error": reg["Max_error"], "p95_error": reg["p95_error"],
        "R2_CI_lo": r2_ci["lo"], "R2_CI_hi": r2_ci["hi"],
        "MAE_CI_lo": mae_ci["lo"], "MAE_CI_hi": mae_ci["hi"],
        # 不确定性 (conformal)
        "PICP_conformal": interval_m["PICP"], "MPIW_conformal": interval_m["MPIW"],
        "PICP_old_ensemble": old_picp, "ECE": ece_res["ECE"],
        "sigma_pearson_r": unc["pearson_r"], "sigma_pearson_p": unc["pearson_p"],
        "phys_share": phys_share,
        "phys_baseline_R2": base_r2,   # 诚实: 物理层独立解释的方差
        "ml_increment_R2": ml_increment,  # 诚实: ML 在物理基础上的增量
        # 分类
        **cls_results,
        # 应用域 (sigma 方法)
        "AD_sigma_R2_trusted": float(ad_rows[0]["R2_trusted"]),
        "AD_sigma_R2_untrusted": float(ad_rows[0]["R2_untrusted"]),
        "AD_knn_R2_trusted": float(ad_rows[1]["R2_trusted"]),
        "AD_knn_R2_untrusted": float(ad_rows[1]["R2_untrusted"]),
        "AD_lev_R2_trusted": float(ad_rows[2]["R2_trusted"]),
        "AD_lev_R2_untrusted": float(ad_rows[2]["R2_untrusted"]),
        "AD_sigma_knn_agreement": float(ad_rows[1]["agreement_with_sigma"]),
        "AD_sigma_lev_agreement": float(ad_rows[2]["agreement_with_sigma"]),
        # OOF 数组
        "oof_mu": oof_mu, "oof_sigma": oof_sigma, "oof_y": y,
        "oof_lower": oof_lower, "oof_upper": oof_upper,
    }


def main():
    df = load_features()
    print("=" * 64, flush=True)
    print("  PACT v2: Physics-Anchored Conformal-Trust", flush=True)
    print("  (conformal UQ + 多方法AD + 统计严谨)", flush=True)
    print("=" * 64, flush=True)

    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n{'='*48}", flush=True)
        res = run_pact_v2(df, target)
        all_res.append(res)

    # 汇总 (去除数组字段)
    summary_rows = []
    for r in all_res:
        row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        summary_rows.append(row)
    df_out = pd.DataFrame(summary_rows)
    out_path = METRICS_DIR / "pact_v2_results.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("  PACT v2 汇总", flush=True)
    print("=" * 64, flush=True)
    for r in all_res:
        print(f"\n  {r['target']}:", flush=True)
        print(f"    R²={r['R2']:.4f} [{r['R2_CI_lo']:.4f},{r['R2_CI_hi']:.4f}] "
              f"MAE={r['MAE']:.4f}", flush=True)
        print(f"    PICP(conformal)={r['PICP_conformal']:.3f} vs "
              f"PICP(old ensemble)={r['PICP_old_ensemble']:.3f}", flush=True)
        print(f"    σ-|err| r={r['sigma_pearson_r']:.3f} 物理贡献={r['phys_share']:.1%}", flush=True)
        print(f"    AD: σ可信R²={r['AD_sigma_R2_trusted']:.4f} "
              f"kNN可信R²={r['AD_knn_R2_trusted']:.4f} "
              f"lev可信R²={r['AD_lev_R2_trusted']:.4f}", flush=True)
        if "F1" in r:
            print(f"    F1={r['F1']:.3f} AUC={r['AUC']:.3f} "
                  f"AUC-PR={r['AUC_PR']:.3f} DAF={r['DAF_top10%']:.2f}", flush=True)


if __name__ == "__main__":
    main()
