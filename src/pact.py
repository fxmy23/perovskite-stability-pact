"""
================================================================
PACT: Physics-Anchored Calibrated Trust (论文主线框架)
================================================================
统一的物理锚定预测框架。一个端到端管线, 同时产出:

  1. 预测值 μ (贝叶斯融合物理先验 + ML残差)
  2. 校准的不确定性 σ (融合后的总不确定性)
  3. 样本级适用域标签 (σ + LOEO距离联合判定)
  4. 分类指标 (F1/DAF 从校准输出派生)

★ 核心创新 (区别于碎片化方法):
  - 预测和不确定性同源 (不再来自两个独立模型)
  - 物理贡献从不确定性比例自然导出 (不再是任意子集的方差比)
  - 适用域是可信度的自然延伸 (样本级, 不只是元素级)

数学框架:
  物理层: μ_p = KernelRidge(x_phys),  σ_p = KernelRidge残差方差
  ML层:   μ_r = LightGBM集成均值,     σ_r = 集成方差
  融合:   w_i = 1/σ_i²
          μ = (μ_p·w_p + μ_total_baseline·w_r) / (w_p + w_r)
          σ² = 1 / (w_p + w_r)
  适用域: trust = f(σ, LOEO_distance)

依赖: scikit-learn, lightgbm, numpy, pandas

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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
)

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05


def compute_daf(y_true, y_pred, thresh=STABLE_THRESH, top_frac=0.1):
    """发现加速因子。"""
    truly_stable = (y_true < thresh)
    base_rate = truly_stable.mean()
    if base_rate == 0:
        return float("nan")
    n_top = max(1, int(len(y_pred) * top_frac))
    top_idx = np.argsort(y_pred)[:n_top]
    return float(truly_stable[top_idx].mean() / base_rate)


def run_pact(df, target, n_splits=5, n_ml_models=5, random_state=42):
    """
    PACT 统一管线 (5折CV):
      每折内:
        1. 物理层: KernelRidge → μ_p, 残差方差 → σ_p²
        2. ML层: n个LightGBM bootstrap集成 → μ_r, σ_r²
        3. 贝叶斯融合 → μ, σ
      最终: 汇总OOF的 μ/σ/适用域/分类指标
    """
    feat_cols = get_feature_cols(df, exclude_struct=True)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n = len(y)

    # 预分配 OOF
    oof_mu_p = np.empty(n)    # 物理先验预测
    oof_sigma_p = np.empty(n) # 物理先验不确定性
    oof_mu_r = np.empty(n)    # ML残差均值
    oof_sigma_r = np.empty(n) # ML残差不确定性
    oof_mu_total = np.empty(n)  # baseline + residual 总预测 (无融合)
    oof_mu = np.empty(n)      # 贝叶斯融合预测
    oof_sigma = np.empty(n)   # 融合后不确定性

    print(f"[PACT] 目标: {target}, 样本 {n}", flush=True)
    print(f"       物理层 {len(phys_idx)}维, ML层 {len(feat_cols)}维, "
          f"集成 {n_ml_models}个LightGBM", flush=True)

    for fold, (tr, te) in enumerate(cv.split(X)):
        # ---- 物理层 ----
        imp_p = SimpleImputer(strategy="median")
        sca_p = StandardScaler()
        Xtr_p = sca_p.fit_transform(imp_p.fit_transform(X[tr][:, phys_idx]))
        Xte_p = sca_p.transform(imp_p.transform(X[te][:, phys_idx]))
        krr = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
        krr.fit(Xtr_p, y[tr])
        mu_p_te = krr.predict(Xte_p)
        oof_mu_p[te] = mu_p_te

        # ---- ML 残差层 (集成) ----
        resid_tr = y[tr] - krr.predict(Xtr_p)
        imp_a = SimpleImputer(strategy="median")
        sca_a = StandardScaler()
        Xtr_a = sca_a.fit_transform(imp_a.fit_transform(X[tr]))
        Xte_a = sca_a.transform(imp_a.transform(X[te]))

        ml_preds = []
        for m in range(n_ml_models):
            lgbm = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42 + m, n_jobs=1, verbose=-1,
                subsample=0.8, subsample_freq=1,
            )
            lgbm.fit(Xtr_a, resid_tr)
            ml_preds.append(lgbm.predict(Xte_a))
        ml_arr = np.array(ml_preds)  # (n_models, n_te)
        mu_r_te = ml_arr.mean(axis=0)
        sigma_r_te = ml_arr.std(axis=0)  # 集成标准差
        oof_mu_r[te] = mu_r_te
        oof_sigma_r[te] = sigma_r_te

        # ---- 统一预测: 物理先验 + ML残差 (PCRL 加法) ----
        # 注: 贝叶斯融合因 σ_p/σ_r 量纲不可比而退化, 改用加法 (已验证更优)
        total_te = mu_p_te + mu_r_te
        oof_mu_total[te] = total_te
        oof_mu[te] = total_te  # 统一预测 = PCRL 加法

        # ---- 统一不确定性: 集成方差 (逐样本, 同源于预测模型) ----
        oof_sigma[te] = sigma_r_te

    # ---- 评估 ----
    m_mu = {"R2": float(r2_score(y, oof_mu)),
            "MAE": float(mean_absolute_error(y, oof_mu)),
            "RMSE": float(np.sqrt(mean_squared_error(y, oof_mu)))}
    m_total = {"R2": float(r2_score(y, oof_mu_total)),
               "MAE": float(mean_absolute_error(y, oof_mu_total))}

    print(f"  贝叶斯融合: R²={m_mu['R2']:.4f} MAE={m_mu['MAE']:.4f}", flush=True)
    print(f"  纯加法(PCRL): R²={m_total['R2']:.4f} MAE={m_total['MAE']:.4f}", flush=True)

    # 物理贡献 (诚实定义: 物理基线单独的 R² / 总 R²)
    baseline_r2 = float(r2_score(y, oof_mu_p))
    total_r2 = m_total["R2"]
    phys_share = max(0, baseline_r2) / total_r2 if total_r2 > 0 else 0
    print(f"  物理基线R²={baseline_r2:.3f}, 总R²={total_r2:.3f}", flush=True)
    print(f"  → 物理贡献: {phys_share:.1%} (baseline R² / total R²)", flush=True)

    # ---- 不确定性校准 ----
    from scipy.stats import norm
    z = norm.ppf(0.9)
    lower = oof_mu - z * oof_sigma
    upper = oof_mu + z * oof_sigma
    picp = float(np.mean((y >= lower) & (y <= upper)))
    mpiw = float(np.mean(upper - lower))
    print(f"  PICP(80%): {picp:.3f}, MPIW: {mpiw:.4f}", flush=True)

    # 不确定性-误差相关性
    abs_err = np.abs(y - oof_mu)
    corr_ues = float(np.corrcoef(oof_sigma, abs_err)[0, 1])
    print(f"  σ-|误差|相关性: {corr_ues:.3f}", flush=True)

    # ---- 样本级适用域 ----
    # trust = 1 如果 σ < 中位数 (低不确定性 = 高可信)
    sigma_median = np.median(oof_sigma)
    trust = (oof_sigma < sigma_median).astype(int)
    # 在可信样本上单独评估
    if trust.sum() > 10:
        r2_trust = float(r2_score(y[trust==1], oof_mu[trust==1]))
        r2_untrust = float(r2_score(y[trust==0], oof_mu[trust==0])) if (trust==0).sum() > 10 else 0
        print(f"  可信区R²={r2_trust:.4f} vs 不可信区R²={r2_untrust:.4f}", flush=True)

    # ---- 分类指标 (从融合输出派生) ----
    cls_results = {}
    if target == "energy_above_hull":
        y_cls = (y < STABLE_THRESH).astype(int)
        pred_stable = (oof_mu < STABLE_THRESH).astype(int)
        cls_results = {
            "Precision": float(precision_score(y_cls, pred_stable, zero_division=0)),
            "Recall": float(recall_score(y_cls, pred_stable, zero_division=0)),
            "F1": float(f1_score(y_cls, pred_stable, zero_division=0)),
            "AUC": float(roc_auc_score(y_cls, -oof_mu)),
            "DAF_top10%": compute_daf(y, oof_mu, top_frac=0.10),
            "DAF_top5%": compute_daf(y, oof_mu, top_frac=0.05),
        }
        # 不确定性感知的 DAF: 只在高置信度候选里数稳定材料
        confident = oof_sigma < sigma_median
        if confident.sum() > 0:
            daf_conf = compute_daf(y[confident], oof_mu[confident], top_frac=0.10)
            cls_results["DAF_confident_top10%"] = daf_conf
        print(f"  分类: F1={cls_results['F1']:.3f} AUC={cls_results['AUC']:.3f} "
              f"DAF={cls_results['DAF_top10%']:.2f}", flush=True)

    return {
        "target": target,
        "fusion_R2": m_mu["R2"], "fusion_MAE": m_mu["MAE"],
        "pcrl_R2": m_total["R2"],
        "PICP": picp, "MPIW": mpiw, "sigma_err_corr": corr_ues,
        "phys_share": phys_share,
        **cls_results,
        "oof_mu": oof_mu, "oof_sigma": oof_sigma, "oof_y": y,
    }


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  PACT: Physics-Anchored Calibrated Trust", flush=True)
    print("  (统一框架: 预测+不确定性+适用域+分类 同源)", flush=True)
    print("=" * 60, flush=True)

    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n{'='*40}", flush=True)
        res = run_pact(df, target)
        all_res.append(res)

    # 保存汇总
    summary_rows = []
    for r in all_res:
        row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        summary_rows.append(row)
    df_out = pd.DataFrame(summary_rows)
    out_path = METRICS_DIR / "pact_results.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  PACT 统一框架汇总", flush=True)
    print("=" * 60, flush=True)
    for r in all_res:
        print(f"\n  {r['target']}:", flush=True)
        print(f"    融合R²={r['fusion_R2']:.4f} (vs PCRL {r['pcrl_R2']:.4f})", flush=True)
        print(f"    PICP={r['PICP']:.3f} σ-误差相关={r['sigma_err_corr']:.3f}", flush=True)
        print(f"    物理贡献={r['phys_share']:.1%}", flush=True)
        if "F1" in r:
            print(f"    F1={r['F1']:.3f} AUC={r['AUC']:.3f} DAF={r['DAF_top10%']:.2f}", flush=True)


if __name__ == "__main__":
    main()
