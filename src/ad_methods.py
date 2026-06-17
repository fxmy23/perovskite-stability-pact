"""
================================================================
应用域 (Applicability Domain) 多方法对照 (P1 修订)
================================================================
背景: 原 PACT 用 "σ < 中位数" 做 AD 判定, 是单一启发式, 审稿人会质疑
  "为什么是这个阈值 / 为什么用 σ 不用别的"。

本模块实现 3 种标准 AD 方法的对照, 证明 σ 阈值不是任意选择:
  1. k-NN distance: 样本到训练集 k 近邻的平均距离, 阈值=训练距离 95 分位
  2. Leverage (Williams plot): h_i = x_iᵀ(XᵀX)⁻¹x_i, 阈值 h*=3p/n (PCA 降维后)
  3. Ensemble σ (主线): 集成标准差, 阈值=中位数

输出: 每种方法在"可信/不可信"分区上的 R² 差异, 以及方法间一致性。
若三种方法结论一致 (可信区 R² 都显著高于不可信区), 即多方法一致性证据。

依赖: scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------
# 方法 1: k-NN distance (在 PCA 降维空间上, 加速)
# ----------------------------------------------------------------
def knn_ad_oof(X, y, cv_splits, k=5, pct=95, pca_nc=20):
    """
    每折: PCA 降到 pca_nc 维, 算测试样本到训练集 k 近邻平均距离 d_te,
          阈值 = 训练集自身 kNN 距离的 pct 分位。
    PCA 降维是必须的: 113 维欧氏距离受维度灾难影响 + kNN 在高维慢。
    """
    n = len(y)
    oof_dist = np.full(n, np.nan)
    for tr, te in cv_splits:
        scaler = StandardScaler()
        imp = SimpleImputer(strategy="median")
        Xtr_raw = scaler.fit_transform(imp.fit_transform(X[tr]))
        Xte_raw = scaler.transform(imp.transform(X[te]))
        nc = min(pca_nc, Xtr_raw.shape[0] - 1, Xtr_raw.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        Xtr = pca.fit_transform(Xtr_raw)
        Xte = pca.transform(Xte_raw)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(Xtr)
        d_te, _ = nn.kneighbors(Xte)
        oof_dist[te] = d_te[:, 1:].mean(axis=1)
    return oof_dist


def knn_trust_per_fold(X, y, cv_splits, k=5, pct=95, pca_nc=20):
    """返回逐折判定后的 trust 标签 (n,)。"""
    n = len(y)
    trust = np.zeros(n, dtype=int)
    for tr, te in cv_splits:
        scaler = StandardScaler()
        imp = SimpleImputer(strategy="median")
        Xtr_raw = scaler.fit_transform(imp.fit_transform(X[tr]))
        Xte_raw = scaler.transform(imp.transform(X[te]))
        nc = min(pca_nc, Xtr_raw.shape[0] - 1, Xtr_raw.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        Xtr = pca.fit_transform(Xtr_raw)
        Xte = pca.transform(Xte_raw)
        nn = NearestNeighbors(n_neighbors=k + 1, n_jobs=1).fit(Xtr)
        d_tr, _ = nn.kneighbors(Xtr)
        thresh = np.percentile(d_tr[:, 1:].mean(axis=1), pct)
        d_te, _ = nn.kneighbors(Xte)
        trust[te] = (d_te[:, 1:].mean(axis=1) <= thresh).astype(int)
    return trust


# ----------------------------------------------------------------
# 方法 2: Leverage (Williams plot) on PCA
# ----------------------------------------------------------------
def leverage_ad_oof(X, y, cv_splits, n_components=20, h_star_factor=3.0):
    """
    每折: PCA 降到 n_components, 算 leverage h_i = x_iᵀ(XᵀX)⁻¹x_i,
          阈值 h* = h_star_factor * n_components / n_train。
    高维时必须先 PCA, 否则 (XᵀX)⁻¹ 病态。
    """
    n = len(y)
    oof_lev = np.full(n, np.nan)
    trust = np.zeros(n, dtype=int)
    for tr, te in cv_splits:
        scaler = StandardScaler()
        imp = SimpleImputer(strategy="median")
        Xtr_raw = scaler.fit_transform(imp.fit_transform(X[tr]))
        Xte_raw = scaler.transform(imp.transform(X[te]))
        nc = min(n_components, Xtr_raw.shape[0] - 1, Xtr_raw.shape[1])
        pca = PCA(n_components=nc, random_state=42)
        Xtr = pca.fit_transform(Xtr_raw)
        Xte = pca.transform(Xte_raw)
        # leverage: h = diag(X (XᵀX)⁻¹ Xᵀ)
        # 用 SVD 数值稳定: X = U S Vᵀ, (XᵀX)⁻¹ = V S⁻² Vᵀ
        U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
        S2_inv = 1.0 / (S ** 2 + 1e-10)
        # h_te_j = sum_k Xte[j,k] * (V S⁻² Vᵀ Xte[j]ᵀ)_k
        Xt_Xinv = (Vt.T * S2_inv) @ Vt  # (nc, nc)
        h_te = np.einsum("ij,jk,ik->i", Xte, Xt_Xinv, Xte)
        oof_lev[te] = h_te
        h_star = h_star_factor * nc / len(tr)
        trust[te] = (h_te <= h_star).astype(int)
    return oof_lev, trust


# ----------------------------------------------------------------
# 方法 3: Ensemble σ (主线已有, 这里统一接口)
# ----------------------------------------------------------------
def sigma_trust(sigma, cv_splits=None):
    """σ < 中位数 → 可信。无需 CV (σ 是 OOF 量)。"""
    return (sigma < np.median(sigma)).astype(int)


# ----------------------------------------------------------------
# 多方法一致性 + 分区 R²
# ----------------------------------------------------------------
def compare_ad_methods(y_true, y_pred, sigma, X, cv_splits,
                       knn_k=5, knn_pct=95, pca_nc=20):
    """
    返回 (rows, trust_sigma, trust_knn, trust_lev): 每种 AD 方法的
    可信区R² / 不可信区R² / 可信比例 / 与σ一致性。
    ★ 性能: kNN 与 leverage 都在 PCA(pca_nc) 空间上算, 避免高维慢。
    """
    from sklearn.metrics import r2_score
    trust_sigma = sigma_trust(sigma)
    trust_knn = knn_trust_per_fold(X, y_true, cv_splits, k=knn_k, pct=knn_pct, pca_nc=pca_nc)
    _, trust_lev = leverage_ad_oof(X, y_true, cv_splits, n_components=pca_nc)

    rows = []
    for name, trust in [("ensemble_sigma", trust_sigma),
                        ("knn_distance", trust_knn),
                        ("leverage_pca", trust_lev)]:
        t_mask = trust == 1
        u_mask = trust == 0
        r2_t = r2_score(y_true[t_mask], y_pred[t_mask]) if t_mask.sum() > 10 else float("nan")
        r2_u = r2_score(y_true[u_mask], y_pred[u_mask]) if u_mask.sum() > 10 else float("nan")
        # 与 σ 方法的一致率
        agree = float(np.mean(trust == trust_sigma))
        rows.append({
            "method": name,
            "n_trusted": int(t_mask.sum()),
            "frac_trusted": float(t_mask.mean()),
            "R2_trusted": float(r2_t),
            "R2_untrusted": float(r2_u),
            "R2_gap": float(r2_t - r2_u),
            "agreement_with_sigma": agree,
        })
    return np.array(rows, dtype=object), trust_sigma, trust_knn, trust_lev
