"""
================================================================
优化①: 域适应 LOEO (RULSIF-style 密度比加权)
================================================================
解决核心痛点: LOEO 外推差 (R² 0.70, 多元素崩)。

原理 (基于 RSC Digital Discovery 2024 + Sugiyama RULSIF):
  标准 LOEO: 训练集包含所有非留出元素, 模型对留出元素外推差。
  域适应: 估计密度比 w(x) = p_test(x)/p_train(x), 给"像留出元素"的
    训练样本更高权重 → 模型更关注能迁移到留出元素的知识。

实现 (透明可debug, 非黑盒):
  - 密度比估计: 用 KDE (核密度估计) 在物理特征空间算
    w_i = p_test(x_i) / p_train(x_i), i 为训练样本
  - 截断+归一化权重 (避免极端值, RULSIF 的 alpha-相对比思想)
  - LightGBM sample_weight 加权训练
  - 对照: 同样LOEO切分, 无权重 vs 有权重

依赖: scikit-learn (KDE), lightgbm, numpy, pandas

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
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DA_DIR = METRICS_DIR / "domain_adapt_loeo"
DA_DIR.mkdir(parents=True, exist_ok=True)


def estimate_density_ratios(X_train, X_test, alpha_clip=0.5, n_bandwidth_search=5):
    """
    估计密度比 w_i = p_test(x_i)/p_train(x_i), i 为训练样本。

    用 KDE 分别拟合 p_train 和 p_test (在物理特征PCA空间),
    然后计算每个训练样本的密度比。
    alpha_clip: RULSIF 的 alpha-相对比思想, w = w/(alpha*w + (1-alpha)),
                等效于把极端权重向1收缩, 提升数值稳定性。
    """
    # KDE 带宽选择 (用经验法则 + 少量搜索, 避免昂贵CV)
    n_tr, d = X_train.shape
    # Scott 规则初始带宽
    bw0 = n_tr ** (-1 / (d + 4)) * X_train.std()
    bandwidths = bw0 * np.array([0.3, 0.5, 1.0, 2.0, 3.0])

    # 拟合 p_train (全训练) 和 p_test (留出元素)
    kde_tr = KernelDensity(bandwidth=bandwidths[2], kernel="gaussian").fit(X_train)
    kde_te = KernelDensity(bandwidth=bandwidths[2], kernel="gaussian").fit(X_test)

    # 密度比 (log 空间更稳定)
    log_p_tr = kde_tr.score_samples(X_train)
    log_p_te = kde_te.score_samples(X_train)
    log_w = log_p_te - log_p_tr
    w = np.exp(log_w)

    # RULSIF alpha-相对比: w_alpha = w / (alpha*w + (1-alpha)), 截断极端值
    w_alpha = w / (alpha_clip * w + (1 - alpha_clip))
    # 归一化使均值为1 (保持有效样本量)
    w_alpha = w_alpha / w_alpha.mean()
    # 再截断 [0.1, 10] 避免极端权重主导
    w_alpha = np.clip(w_alpha, 0.1, 10.0)
    return w_alpha


def loeo_one_element(X, y, a_sites, elem, phys_idx, use_da=True, n_ml=3):
    """留一元素: 纯ML vs 域适应加权ML。"""
    mask = a_sites != elem
    Xtr, ytr = X[mask], y[mask]
    Xte, yte = X[~mask], y[~mask]
    if len(yte) < 5 or len(set(yte)) < 2:
        return None

    # 预处理 (折内 fit)
    imp = SimpleImputer(strategy="median"); sca = StandardScaler()
    Xtr_all = sca.fit_transform(imp.fit_transform(Xtr))
    Xte_all = sca.transform(imp.transform(Xte))
    # 物理特征子空间 (用于密度比估计)
    sca_p = StandardScaler()
    Xtr_p = sca_p.fit_transform(SimpleImputer(strategy="median").fit_transform(Xtr[:, phys_idx]))
    Xte_p = sca_p.transform(SimpleImputer(strategy="median").fit_transform(Xte[:, phys_idx]))

    # 标准 ML (无权重)
    preds_std = []
    for m in range(n_ml):
        lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                            random_state=42+m, n_jobs=1, verbose=-1,
                            subsample=0.8, subsample_freq=1).fit(Xtr_all, ytr)
        preds_std.append(lgb.predict(Xte_all))
    r2_std = r2_score(yte, np.mean(preds_std, axis=0))

    r2_da = float('nan')
    if use_da:
        # 域适应: 估计密度比 (用物理特征空间)
        w = estimate_density_ratios(Xtr_p, Xte_p, alpha_clip=0.5)
        preds_da = []
        for m in range(n_ml):
            lgb = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42+m, n_jobs=1, verbose=-1,
                                subsample=0.8, subsample_freq=1).fit(Xtr_all, ytr, sample_weight=w)
            preds_da.append(lgb.predict(Xte_all))
        r2_da = r2_score(yte, np.mean(preds_da, axis=0))

    return {
        "element": elem, "n_test": int(len(yte)),
        "r2_standard": float(r2_std), "r2_domain_adapt": float(r2_da),
    }


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys = [c for c in PHYS_FEATURES if c not in EXCLUDE_FROM_ML and c in df.columns]
    phys_idx = [i for i, c in enumerate(feat) if c in phys]
    X = df[feat].values
    y = df["formation_energy_per_atom"].values
    a_sites = df["a_site_element"].values

    from collections import Counter
    counts = Counter(a_sites)
    elems = sorted([e for e, n in counts.items() if n >= 50])
    print("=" * 72, flush=True)
    print(f"  域适应 LOEO (RULSIF-style 密度比加权, {len(elems)} 元素)", flush=True)
    print("=" * 72, flush=True)

    # 断点续跑
    done = set()
    for f in DA_DIR.glob("elem_*.json"):
        try: done.add(f.stem.split("_", 1)[1])
        except: pass
    todo = [e for e in elems if e not in done]
    print(f"  已完成 {len(done)}, 待跑 {len(todo)}", flush=True)

    for i, elem in enumerate(todo):
        print(f"  [{i+1+len(done)}/{len(elems)}] {elem} ...", flush=True, end="")
        try:
            res = loeo_one_element(X, y, a_sites, elem, phys_idx, use_da=True)
            if res is None:
                print(" 跳过", flush=True); continue
            (DA_DIR / f"elem_{elem}.json").write_text(
                json.dumps(res, ensure_ascii=False), encoding="utf-8")
            print(f" std={res['r2_standard']:+.2f} DA={res['r2_domain_adapt']:+.2f} [SAVE]", flush=True)
        except Exception as e:
            print(f" ERROR: {type(e).__name__}: {str(e)[:80]}", flush=True)

    # 汇总
    print("\n" + "=" * 72, flush=True)
    print("  汇总", flush=True)
    print("=" * 72, flush=True)
    rows = []
    for f in sorted(DA_DIR.glob("elem_*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        print("  无结果", flush=True); return
    df_res = pd.DataFrame(rows)

    std = df_res["r2_standard"].dropna()
    da = df_res["r2_domain_adapt"].dropna()
    print(f"  标准 ML:     mean={std.mean():+.3f} median={std.median():+.3f} (n={len(std)})", flush=True)
    print(f"  域适应 ML:   mean={da.mean():+.3f} median={da.median():+.3f} (n={len(da)})", flush=True)
    print(f"  提升: mean {da.mean()-std.mean():+.3f}  median {da.median()-std.median():+.3f}", flush=True)

    # 配对比较
    valid = df_res.dropna(subset=["r2_standard", "r2_domain_adapt"])
    da_better = (valid["r2_domain_adapt"] > valid["r2_standard"]).sum()
    print(f"  域适应更优的元素: {da_better}/{len(valid)} ({da_better/len(valid)*100:.0f}%)", flush=True)
    try:
        from scipy.stats import wilcoxon
        w, p = wilcoxon(valid["r2_domain_adapt"], valid["r2_standard"])
        print(f"  Wilcoxon p={p:.4f} ({'显著' if p<0.05 else '不显著'})", flush=True)
    except Exception as e:
        print(f"  wilcoxon err: {e}", flush=True)

    df_res.to_csv(METRICS_DIR / "domain_adapt_loeo_results.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] domain_adapt_loeo_results.csv", flush=True)


if __name__ == "__main__":
    main()
