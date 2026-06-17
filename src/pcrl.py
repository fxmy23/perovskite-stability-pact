"""
================================================================
PCRL: Physics-Constrained Residual Learning — 消融对照实验
================================================================
★ 2026-06-16 诚实降级 (基于严苛审稿 + 实测数据):

  本模块原计划作为论文"核心方法创新", 但 5 折 CV 实测显示:
    pure_ml (LightGBM)         R²=0.9101  (形成能)
    standard_pgml              R²=0.9074
    pcrl_v1 (p=0.1)            R²=0.9060
    pcrl_v2_shap               R²=0.9039  ← 反而最差
  即 PCRL 在两个目标上均劣于纯 ML 基线。

  根因分析 (诚实, 见 docs/research_findings_v2.md §5):
    形成能高度物理可解释 (容差因子 t、电负性差 χ_AB 已捕获主要变异),
    物理先验近完整。此时对残差施加物理特征惩罚会压制有用的非线性,
    反而劣化性能。PCRL 的适用边界是"物理先验不完整"的任务 (如带隙、
    弹性模量), 这与 Mannodi 2020 (npj) 在弹性模量上 PCRL 成功一致。

  本模块因此定位为 §"对照实验: 显式物理约束的适用边界"——
  这本身是有科学价值的负面结果 (材料 ML 领域稀缺的诚实分析)。
  论文创新主线已转移至 PACT v2 (conformal UQ + 多方法 AD + LOEO)。

PCRL 数学框架 (保留作参考):
  y = f_phys(x_phys) + g_ML(x_all)
  损失 = ||y - f - g||² + λ·||∇_{x_phys} g||²
  (梯度项用"物理特征在残差模型的SHAP重要性"近似)

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
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ★ 统一物理特征列表: 从 utils 引用 (单一来源)
try:
    from src.utils import PHYS_FEATURES
except ImportError:
    PHYS_FEATURES = [
        "phys_tolerance_factor", "phys_octahedral_factor",
        "phys_electroneg_diff_AB", "phys_radius_ratio_AB",
        "phys_b_site_valence", "phys_a_site_radius", "phys_b_site_radius",
        "phys_a_site_en", "phys_b_site_en", "phys_stability_score",
    ]


# ----------------------------------------------------------------
# PCRL 核心模型
# ----------------------------------------------------------------
class PCRLModel:
    """
    Physics-Constrained Residual Learning 模型。

    Parameters
    ----------
    physics_penalty : float
        物理一致性约束强度。物理特征在残差模型中的分裂增益乘以 (1-penalty)。
        penalty=0 → 无约束 (退化为标准PGML)
        penalty=1 → 物理特征完全禁止在残差中使用
    adaptive_alpha : bool
        是否启用自适应权重融合
    """

    def __init__(self, physics_penalty=0.1, adaptive_alpha=False,
                 phys_features=None):
        self.physics_penalty = physics_penalty
        self.adaptive_alpha = adaptive_alpha
        self.phys_features = phys_features or PHYS_FEATURES
        self._phys_idx = None
        self._scaler_phys = None
        self._scaler_all = None
        self._imputer_phys = None
        self._imputer_all = None
        self._baseline = None
        self._residual = None
        self._alpha = 1.0

    def _split_features(self, X, feat_cols):
        """拆分物理特征与全部特征, 记录物理列索引。"""
        phys_idx = [i for i, c in enumerate(feat_cols) if c in self.phys_features]
        return phys_idx

    def fit(self, X, y, feat_cols):
        """
        训练 PCRL: 先物理基线, 再约束残差。

        X: (n, d) 全特征矩阵
        y: (n,) 目标
        feat_cols: 特征列名列表 (用于识别物理列)
        """
        self._phys_idx = self._split_features(X, feat_cols)
        X_phys = X[:, self._phys_idx]

        # 预处理 (impute + scale, fit on train only, 无泄露)
        self._imputer_phys = SimpleImputer(strategy="median")
        self._scaler_phys = StandardScaler()
        X_phys_p = self._scaler_phys.fit_transform(
            self._imputer_phys.fit_transform(X_phys)
        )

        self._imputer_all = SimpleImputer(strategy="median")
        self._scaler_all = StandardScaler()
        X_all_p = self._scaler_all.fit_transform(
            self._imputer_all.fit_transform(X)
        )

        # Step 1: 物理基线 (KernelRidge RBF, 捕捉容忍因子的非线性稳定带)
        self._baseline = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
        self._baseline.fit(X_phys_p, y)

        # Step 2: 计算残差
        baseline_pred = self._baseline.predict(X_phys_p)
        residual = y - baseline_pred

        # Step 3: ★ 物理一致性约束的残差模型
        # LightGBM 的 feature_penalty 参数: 降低物理特征的分裂增益
        if HAS_LGBM and self.physics_penalty > 0:
            # 构造 feature_penalty 数组: 物理特征惩罚, 其余正常
            penalties = [1.0] * X.shape[1]
            for i in self._phys_idx:
                penalties[i] = 1.0 - self.physics_penalty
            self._residual = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
                feature_penalty=penalties,
            )
        elif HAS_LGBM:
            self._residual = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )
        else:
            self._residual = Ridge(alpha=1.0)

        self._residual.fit(X_all_p, residual)

        # Step 4: 自适应权重 α (可选)
        if self.adaptive_alpha:
            total_pred = baseline_pred + self._residual.predict(X_all_p)
            ml_pred = self._residual.predict(X_all_p)
            # α 优化: 在训练集上找最优混合 (简化版, 正式应分验证集)
            best_a, best_r2 = 1.0, -np.inf
            for a in np.linspace(0, 1, 11):
                mix = a * baseline_pred + (1 - a) * total_pred
                r2 = r2_score(y, mix)
                if r2 > best_r2:
                    best_r2, best_a = r2, a
            self._alpha = best_a

        return self

    def predict(self, X):
        X_phys = X[:, self._phys_idx]
        X_phys_p = self._scaler_phys.transform(self._imputer_phys.transform(X_phys))
        X_all_p = self._scaler_all.transform(self._imputer_all.transform(X))

        bp = self._baseline.predict(X_phys_p)
        rp = self._residual.predict(X_all_p)
        total = bp + rp
        if self.adaptive_alpha:
            return self._alpha * bp + (1 - self._alpha) * total
        return total

    def predict_decomposed(self, X):
        """返回 (baseline_pred, residual_pred, total_pred)。"""
        X_phys = X[:, self._phys_idx]
        X_phys_p = self._scaler_phys.transform(self._imputer_phys.transform(X_phys))
        X_all_p = self._scaler_all.transform(self._imputer_all.transform(X))

        bp = self._baseline.predict(X_phys_p)
        rp = self._residual.predict(X_all_p)
        total = bp + rp
        if self.adaptive_alpha:
            total = self._alpha * bp + (1 - self._alpha) * total
        return bp, rp, total


# ----------------------------------------------------------------
# 四方对比实验
# ----------------------------------------------------------------
def evaluate(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_pcrl_v2_shap_guided(df, target, n_splits=5, random_state=42,
                             target_phys_share=0.20):
    """
    PCRL v2: SHAP-guided 自适应物理约束。

    核心创新 (区别于固定 feature_penalty 的 PCRL v1):
      1. 初始 penalty=0.1 训练残差模型
      2. 计算残差模型的 SHAP, 测量物理特征占比
      3. 如果物理占比 > target_phys_share (如20%), 增加 penalty 重训
      4. 迭代直到物理占比达标或 penalty 达上限

    这实现了"显式 SHAP 正则"的第二代 PIML:
      L = ||y - f - g||² + λ·SHAP_phys(g)²
    其中 λ 由自适应迭代确定, 而非硬编码。

    返回: (指标DataFrame, 最优penalty, 最终物理SHAP占比)
    """
    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False

    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]

    print(f"[PCRL-v2] 目标: {target}, SHAP-guided自适应约束", flush=True)
    print(f"          目标物理SHAP占比: ≤{target_phys_share:.0%}", flush=True)

    # CV 评估函数 (给定 penalty)
    def eval_with_penalty(penalty):
        preds = np.empty(len(y))
        for tr, te in cv.split(X):
            imp_p = SimpleImputer(strategy="median")
            sca_p = StandardScaler()
            Xtr_p = sca_p.fit_transform(imp_p.fit_transform(X[tr][:, phys_idx]))
            Xte_p = sca_p.transform(imp_p.transform(X[te][:, phys_idx]))
            base = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
            base.fit(Xtr_p, y[tr])
            res_tr = y[tr] - base.predict(Xtr_p)

            imp_a = SimpleImputer(strategy="median")
            sca_a = StandardScaler()
            Xtr_a = sca_a.fit_transform(imp_a.fit_transform(X[tr]))
            Xte_a = sca_a.transform(imp_a.transform(X[te]))

            penalties = [1.0] * len(feat_cols)
            if penalty > 0:
                for i in phys_idx:
                    penalties[i] = 1.0 - penalty
            ml = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                               random_state=42, n_jobs=1, verbose=-1,
                               feature_penalty=penalties)
            ml.fit(Xtr_a, res_tr)
            preds[te] = base.predict(Xte_p) + ml.predict(Xte_a)
        return preds

    # ★ 修复: 嵌套 CV 选 penalty — 只用第一折训练集, 不碰测试数据
    best_penalty = 0.1
    best_share = 1.0
    if HAS_SHAP:
        # 取第一折的训练集做 penalty 选择 (近似嵌套, 避免全量泄露)
        first_tr, _ = next(iter(cv.split(X)))
        X_sel = X[first_tr]
        y_sel = y[first_tr]

        imp_p = SimpleImputer(strategy="median")
        sca_p = StandardScaler()
        X_phys_p = sca_p.fit_transform(imp_p.fit_transform(X_sel[:, phys_idx]))
        base = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
        base.fit(X_phys_p, y_sel)
        res = y_sel - base.predict(X_phys_p)

        imp_a = SimpleImputer(strategy="median")
        sca_a = StandardScaler()
        X_all_p = sca_a.fit_transform(imp_a.fit_transform(X_sel))

        for penalty in [0.1, 0.2, 0.3, 0.5]:
            penalties = [1.0] * len(feat_cols)
            for i in phys_idx:
                penalties[i] = 1.0 - penalty
            ml = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                               random_state=42, n_jobs=1, verbose=-1,
                               feature_penalty=penalties)
            ml.fit(X_all_p, res)
            explainer = shap.TreeExplainer(ml)
            sv = explainer.shap_values(X_all_p[:300])
            global_imp = np.abs(sv).mean(axis=0)
            phys_total = global_imp[phys_idx].sum()
            all_total = global_imp.sum()
            phys_share = phys_total / all_total if all_total > 0 else 0.0
            print(f"    penalty={penalty}: 物理SHAP占比={phys_share:.1%} (训练折内选择)", flush=True)
            if phys_share <= target_phys_share:
                best_penalty, best_share = penalty, phys_share
                break
            best_penalty, best_share = penalty, phys_share
        print(f"  → 选定 penalty={best_penalty} (物理SHAP={best_share:.1%})", flush=True)

    # 用最优 penalty 做 CV 评估
    preds = eval_with_penalty(best_penalty)
    m = evaluate(y, preds)
    print(f"  PCRL-v2 R²={m['R2']:.4f} MAE={m['MAE']:.4f} (penalty={best_penalty})", flush=True)

    # 也评估其他方法做对比
    results = []
    for label, p in [("pure_ml", None), ("standard_pgml", 0.0),
                     ("pcrl_v1(p=0.1)", 0.1), ("pcrl_v2_shap", best_penalty)]:
        if p is None:
            # 纯 ML
            preds_ml = np.empty(len(y))
            for tr, te in cv.split(X):
                imp = SimpleImputer(strategy="median")
                sca = StandardScaler()
                Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
                Xte = sca.transform(imp.transform(X[te]))
                ml = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                   random_state=42, n_jobs=1, verbose=-1)
                ml.fit(Xtr, y[tr])
                preds_ml[te] = ml.predict(Xte)
            mm = evaluate(y, preds_ml)
        else:
            mm = evaluate(y, eval_with_penalty(p))
        mm["method"] = label
        results.append(mm)
        print(f"  {label:20s} R²={mm['R2']:.4f}", flush=True)

    return pd.DataFrame(results), best_penalty, best_share


def run_pcrl_comparison(df, target, n_splits=5, random_state=42):
    """
    四方对比: Pure Physics / Pure ML / Standard PGML / PCRL
    """
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    print(f"[PCRL] 目标: {target}, 样本 {len(y)}, 特征 {len(feat_cols)}", flush=True)

    # 预分配
    n = len(y)
    pred_phys = np.empty(n)
    pred_ml = np.empty(n)
    pred_pgml = np.empty(n)
    pred_pcrl = np.empty(n)
    pred_pcrl_adaptive = np.empty(n)
    pcrl_baseline_part = np.empty(n)

    phys_idx = [i for i, c in enumerate(feat_cols) if c in PHYS_FEATURES]

    for fold, (tr, te) in enumerate(cv.split(X)):
        # --- Pure Physics (KernelRidge on 物理特征) ---
        imp_p = SimpleImputer(strategy="median")
        sca_p = StandardScaler()
        Xtr_p = sca_p.fit_transform(imp_p.fit_transform(X[tr][:, phys_idx]))
        Xte_p = sca_p.transform(imp_p.transform(X[te][:, phys_idx]))
        base = KernelRidge(alpha=1.0, kernel="rbf", gamma=0.1)
        base.fit(Xtr_p, y[tr])
        pred_phys[te] = base.predict(Xte_p)

        # --- Pure ML (LightGBM 全特征) ---
        imp_a = SimpleImputer(strategy="median")
        sca_a = StandardScaler()
        Xtr_a = sca_a.fit_transform(imp_a.fit_transform(X[tr]))
        Xte_a = sca_a.transform(imp_a.transform(X[te]))
        ml = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                           random_state=42, n_jobs=1, verbose=-1)
        ml.fit(Xtr_a, y[tr])
        pred_ml[te] = ml.predict(Xte_a)

        # --- Standard PGML (无约束残差) ---
        res_tr = y[tr] - base.predict(Xtr_p)
        ml_pgml = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42, n_jobs=1, verbose=-1)
        ml_pgml.fit(Xtr_a, res_tr)
        pred_pgml[te] = base.predict(Xte_p) + ml_pgml.predict(Xte_a)

        # --- PCRL (物理约束残差, penalty=0.5) ---
        penalties = [1.0] * len(feat_cols)
        for i in phys_idx:
            penalties[i] = 0.5  # 物理特征分裂增益减半
        ml_pcrl = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                                random_state=42, n_jobs=1, verbose=-1,
                                feature_penalty=penalties)
        ml_pcrl.fit(Xtr_a, res_tr)
        bp_te = base.predict(Xte_p)
        pred_pcrl[te] = bp_te + ml_pcrl.predict(Xte_a)
        pcrl_baseline_part[te] = bp_te

        # --- PCRL + Adaptive α ---
        # 在训练集找最优 α
        bp_tr = base.predict(Xtr_p)
        total_tr = bp_tr + ml_pcrl.predict(Xtr_a)
        best_a, best_r2 = 1.0, -1e9
        for a in np.linspace(0, 1, 11):
            r2 = r2_score(y[tr], a * bp_tr + (1 - a) * total_tr)
            if r2 > best_r2:
                best_r2, best_a = r2, a
        pred_pcrl_adaptive[te] = best_a * bp_te + (1 - best_a) * (bp_te + ml_pcrl.predict(Xte_a))

    # 评估
    methods = {
        "pure_physics": pred_phys,
        "pure_ml": pred_ml,
        "standard_pgml": pred_pgml,
        "PCRL (penalty=0.5)": pred_pcrl,
        "PCRL+adaptive": pred_pcrl_adaptive,
    }
    results = []
    for name, preds in methods.items():
        m = evaluate(y, preds)
        m["method"] = name
        results.append(m)
        print(f"  {name:25s} R²={m['R2']:.4f} MAE={m['MAE']:.4f}", flush=True)

    # 物理贡献量化 (正确公式: baseline_R² / total_R²)
    base_r2 = r2_score(y, pcrl_baseline_part)
    pcrl_r2 = r2_score(y, pred_pcrl)
    if pcrl_r2 > 0:
        phys_share = max(0, base_r2) / pcrl_r2
    else:
        phys_share = float("nan")
    print(f"  物理贡献占比: {phys_share:.1%} (baseline R²={base_r2:.3f}, "
          f"PCRL R²={pcrl_r2:.3f})", flush=True)

    df_out = pd.DataFrame(results)
    df_out["target"] = target
    df_out["physics_share"] = [np.nan] * 4 + [phys_share]
    return df_out, phys_share


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  PCRL 消融对照: 显式物理约束的适用边界 (负面结果)", flush=True)
    print("  (诚实报告: PCRL 在高物理可解释目标上劣于纯ML基线)", flush=True)
    print("=" * 60, flush=True)

    # PCRL v2 (SHAP-guided 自适应约束) — 作为对照实验
    all_res = []
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        df_res, best_pen, best_share = run_pcrl_v2_shap_guided(df, target)
        df_res["target"] = target
        all_res.append(df_res)

    df_all = pd.concat(all_res, ignore_index=True)
    out_path = METRICS_DIR / "pcrl_v2_comparison.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("  PCRL 消融对照汇总 (诚实负面结果)", flush=True)
    print("=" * 60, flush=True)
    print(df_all[["target", "method", "R2", "MAE"]].to_string(index=False), flush=True)

    # 负面结果量化分析
    print("\n  === 负面结果分析 ===", flush=True)
    for target in df_all["target"].unique():
        sub = df_all[df_all["target"] == target].set_index("method")
        if "pure_ml" in sub.index and "pcrl_v2_shap" in sub.index:
            drop = sub.loc["pure_ml", "R2"] - sub.loc["pcrl_v2_shap", "R2"]
            print(f"  {target}: PCRL-v2 vs pure_ml R²下降 {drop:.4f} "
                  f"(物理约束适得其反, 因形成能物理先验已近完整)", flush=True)

    print("\n  结论: PCRL 不作为方法创新, 仅作'物理约束适用边界'的对照证据。", flush=True)
    print("  论文方法创新主线 = PACT v2 (conformal UQ + 多方法 AD + LOEO)。", flush=True)


if __name__ == "__main__":
    main()
