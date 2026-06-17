"""
================================================================
条件保形预测: Conformalized Quantile Regression (CQR) — 理论创新
================================================================
★ 这是项目的核心理论升级 (基于 ICML 2025, ICLR 2025 前沿):

问题: 标准split conformal给所有样本相同宽度的区间 (±q), 不实用——
  外推样本(未见元素)需要更宽区间, 内插样本需要更窄。

CQR解决: 用分位数回归预测样本级区间 [q10(X), q90(X)], 再用conformal校准
  保证覆盖。结果: 高不确定样本区间宽, 低不确定样本区间窄 (异方差)。

理论保证 (Romano 2019, ICML 2025改进):
  - 边际覆盖: P(y ∈ C(X)) ≥ 1-α (与标准conformal相同)
  - 条件覆盖: 经验上更接近名义值 (标准conformal的条件覆盖差)

正收益:
  - 在保持PICP≥0.80下, 可信样本的MPIW降低 (更精确)
  - ECE (条件覆盖偏差) 降低

实现 (CV内, 无泄露):
  每折: 训练 q10/q90 分位数LightGBM → calibration集算CQR分数 → 校准区间
  对比: 标准conformal (均匀) vs CQR (异方差)

依赖: lightgbm, scikit-learn, numpy, pandas
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

ALPHA = 0.2  # 80% 名义覆盖
LOWER_Q = 0.1
UPPER_Q = 0.9
LGB_BASE = dict(n_estimators=450, num_leaves=39, learning_rate=0.078,
                max_depth=10, min_child_samples=17, subsample=0.902,
                colsample_bytree=0.728, reg_alpha=0.489, reg_lambda=0.278,
                n_jobs=1, verbose=-1, subsample_freq=1)


def cqr_cv(X, y, target_name, n_splits=5, alpha=ALPHA, cal_frac=0.2, seed=42):
    """
    CQR + 对比标准conformal。
    每折:
      1. 训练集切 proper_train / calibration
      2. proper_train 上训练 q10/q90/中位数 分位数LightGBM
      3. calibration 上算 CQR 分数: max(q10-y, y-q90)
      4. 取 (1-α) 分位数 d, 区间 = [q10-d, q90+d]
      5. 同时: 标准conformal (均匀 ±d') 对照
    """
    cv = KFold(n_splits, shuffle=True, random_state=seed)
    n = len(y)
    # CQR
    oof_lower_cqr = np.full(n, np.nan)
    oof_upper_cqr = np.full(n, np.nan)
    oof_point = np.full(n, np.nan)
    # 标准conformal (对照)
    oof_lower_std = np.full(n, np.nan)
    oof_upper_std = np.full(n, np.nan)

    for fold, (tr, te) in enumerate(cv.split(X)):
        rng = np.random.default_rng(seed + fold)
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))

        # 切 calibration
        perm = rng.permutation(len(tr))
        n_cal = max(20, int(len(tr) * cal_frac))
        cal_idx = perm[:n_cal]
        fit_idx = perm[n_cal:]
        Xf, yf = Xtr[fit_idx], y[tr][fit_idx]
        Xc, yc = Xtr[cal_idx], y[tr][cal_idx]

        # 训练分位数模型 (q10, q90, 中位数)
        q_lo = LGBMRegressor(objective="quantile", alpha=LOWER_Q, **LGB_BASE).fit(Xf, yf)
        q_hi = LGBMRegressor(objective="quantile", alpha=UPPER_Q, **LGB_BASE).fit(Xf, yf)
        q_mid = LGBMRegressor(objective="quantile", alpha=0.5, **LGB_BASE).fit(Xf, yf)

        # 测试折预测
        lo_te = q_lo.predict(Xte)
        hi_te = q_hi.predict(Xte)
        oof_point[te] = q_mid.predict(Xte)
        # 保证 lo <= hi
        lo_te = np.minimum(lo_te, hi_te)
        hi_te = np.maximum(lo_te, hi_te)

        # CQR 校准: calibration集上的分数
        lo_c = q_lo.predict(Xc); hi_c = q_hi.predict(Xc)
        lo_c = np.minimum(lo_c, hi_c); hi_c = np.maximum(lo_c, hi_c)
        cqr_scores = np.maximum(lo_c - yc, yc - hi_c)
        rank = int(np.ceil((1 - alpha) * (n_cal + 1)))
        rank = min(rank, n_cal)
        d = float(np.sort(cqr_scores)[rank - 1])

        oof_lower_cqr[te] = lo_te - d
        oof_upper_cqr[te] = hi_te + d

        # 标准conformal对照 (用中位数模型残差)
        mid_c = q_mid.predict(Xc)
        std_scores = np.abs(yc - mid_c)
        d_std = float(np.sort(std_scores)[rank - 1])
        mid_te = q_mid.predict(Xte)
        oof_lower_std[te] = mid_te - d_std
        oof_upper_std[te] = mid_te + d_std

        print(f"  fold{fold+1} done (CQR d={d:.3f}, std d={d_std:.3f})", flush=True)

    return oof_point, oof_lower_cqr, oof_upper_cqr, oof_lower_std, oof_upper_std


def evaluate_intervals(y, lower, upper, name, nominal=1-ALPHA, sigma=None):
    """评估区间: PICP/MPIW/ECE (条件覆盖)."""
    picp = float(np.mean((y >= lower) & (y <= upper)))
    mpiw = float(np.mean(upper - lower))
    res = {"name": name, "PICP": picp, "MPIW": mpiw, "nominal": nominal}
    # 条件覆盖: 按sigma分桶, 看每桶实际覆盖是否接近nominal
    if sigma is not None:
        covered = ((y >= lower) & (y <= upper)).astype(float)
        order = np.argsort(sigma)
        bins = np.array_split(order, 10)
        cond_gaps = []
        for b in bins:
            if len(b) == 0: continue
            emp = covered[b].mean()
            cond_gaps.append(abs(emp - nominal))
        res["ECE"] = float(np.mean(cond_gaps))  # 条件覆盖偏差
        # 分层MPIW: 低sigma(可信)样本的区间宽
        low_sigma_idx = order[:len(order)//3]
        res["MPIW_trusted"] = float(np.mean(upper[low_sigma_idx] - lower[low_sigma_idx]))
        high_sigma_idx = order[-len(order)//3:]
        res["MPIW_untrusted"] = float(np.mean(upper[high_sigma_idx] - lower[high_sigma_idx]))
    return res


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    X = df[feat].values
    print("=" * 64, flush=True)
    print("  条件保形预测 CQR (理论创新)", flush=True)
    print("  对比: 标准conformal(均匀) vs CQR(异方差)", flush=True)
    print("=" * 64, flush=True)

    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        y = df[target].values
        print(f"\n### {target} ###", flush=True)
        point, lo_cqr, hi_cqr, lo_std, hi_std = cqr_cv(X, y, target)
        # 点预测
        r2 = r2_score(y, point)
        print(f"  点预测 R²={r2:.4f}", flush=True)
        # 用点预测残差作sigma代理 (分层用)
        sigma = np.abs(y - point)
        # 评估
        res_cqr = evaluate_intervals(y, lo_cqr, hi_cqr, "CQR", sigma=sigma)
        res_std = evaluate_intervals(y, lo_std, hi_std, "Standard", sigma=sigma)
        print(f"\n  [结果] {target}:", flush=True)
        print(f"    标准conformal: PICP={res_std['PICP']:.3f} MPIW={res_std['MPIW']:.4f} "
              f"ECE={res_std.get('ECE',0):.3f}", flush=True)
        print(f"    CQR:            PICP={res_cqr['PICP']:.3f} MPIW={res_cqr['MPIW']:.4f} "
              f"ECE={res_cqr.get('ECE',0):.3f}", flush=True)
        print(f"    CQR可信样本MPIW={res_cqr.get('MPIW_trusted',0):.4f} vs "
              f"不可信={res_cqr.get('MPIW_untrusted',0):.4f} (应可信<不可信)", flush=True)
        print(f"    标准可信样本MPIW={res_std.get('MPIW_trusted',0):.4f} vs "
              f"不可信={res_std.get('MPIW_untrusted',0):.4f}", flush=True)
        all_res.append({"target": target, "r2_point": r2,
                        "std_PICP": res_std["PICP"], "std_MPIW": res_std["MPIW"], "std_ECE": res_std.get("ECE",0),
                        "cqr_PICP": res_cqr["PICP"], "cqr_MPIW": res_cqr["MPIW"], "cqr_ECE": res_cqr.get("ECE",0),
                        "cqr_MPIW_trusted": res_cqr.get("MPIW_trusted",0),
                        "cqr_MPIW_untrusted": res_cqr.get("MPIW_untrusted",0),
                        "std_MPIW_trusted": res_std.get("MPIW_trusted",0),
                        "std_MPIW_untrusted": res_std.get("MPIW_untrusted",0)})

    pd.DataFrame(all_res).to_csv(METRICS_DIR / "cqr_results.csv",
                                  index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] cqr_results.csv", flush=True)

    # 诚实判定
    print("\n" + "=" * 64, flush=True)
    print("  CQR vs 标准conformal 判定", flush=True)
    print("=" * 64, flush=True)
    for r in all_res:
        cqr_better_mpiw = r["cqr_MPIW_trusted"] < r["std_MPIW_trusted"]
        picp_ok = r["cqr_PICP"] >= 0.78
        ece_better = r["cqr_ECE"] < r["std_ECE"]
        print(f"  {r['target']}:", flush=True)
        print(f"    可信样本MPIW: CQR {r['cqr_MPIW_trusted']:.4f} vs 标准 {r['std_MPIW_trusted']:.4f} "
              f"{'✓CQR更精确' if cqr_better_mpiw else '⚠CQR未更优'}", flush=True)
        print(f"    PICP达标(≥0.78): {'✓' if picp_ok else '⚠'} (CQR={r['cqr_PICP']:.3f})", flush=True)
        print(f"    ECE(条件覆盖): CQR {r['cqr_ECE']:.3f} vs 标准 {r['std_ECE']:.3f} "
              f"{'✓CQR更校准' if ece_better else '⚠'}", flush=True)


if __name__ == "__main__":
    main()
