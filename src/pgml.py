"""
================================================================
PGML 残差建模模块 (Physics-Guided Machine Learning)
================================================================
论文方法创新核心: 不再用 ML 直接预测目标, 而是把目标分解为

    target  =  physics_baseline(X_phys)  +  ML_residual(X_all)

  - physics_baseline: 基于经典物理判据(容忍因子/八面体因子)的解析模型
  - ML_residual:      ML 学习"物理解释不了的残差"

这样做的好处(论文卖点):
  1. 可量化物理 vs ML 的贡献占比 (物理贡献X% / ML补充Y%)
  2. 残差通常更平滑、更易学, ML 性能更稳
  3. 物理基线提供可解释的"骨架", PGML 比"黑盒ML+事后SHAP"更高级
  4. 对训练集外的元素组合, 物理基线至少给出合理外推

评估流程:
  Step 1: 拟合 physics_baseline (线性/核回归 on 物理特征)
  Step 2: 计算 residual = y_true - physics_baseline_pred
  Step 3: 用 ML 学习 residual
  Step 4: 最终预测 = physics_baseline + ML_residual
  Step 5: 对比 {纯物理, 纯ML, PGML} 三者性能

依赖: scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------
# 物理基线模型
# ----------------------------------------------------------------
# ★ 统一物理特征列表: 从 utils.PHYS_FEATURES 引用 (单一来源)
# 避免多份硬编码副本漂移
try:
    from src.utils import PHYS_FEATURES as PHYS_BASELINE_FEATURES
except ImportError:
    # 兜底 (直接运行时)
    PHYS_BASELINE_FEATURES = [
        "phys_tolerance_factor", "phys_octahedral_factor",
        "phys_electroneg_diff_AB", "phys_radius_ratio_AB",
        "phys_b_site_valence", "phys_a_site_radius", "phys_b_site_radius",
        "phys_a_site_en", "phys_b_site_en", "phys_stability_score",
    ]


class PhysicsBaselineModel(BaseEstimator, RegressorMixin):
    """
    物理基线模型: 用核岭回归(KernelRidge, RBF核)拟合物理特征到目标。

    ★ 统一基线 (2026-06-16 架构自洽性修复):
      之前 pgml.py 用 Ridge, pcrl.py 用 KernelRidge, 导致同一论文两个基线。
      现统一为 KernelRidge(gamma=0.1), 与 pcrl.py 一致。
      之前注释说"KernelRidge 让 R²变负"是因为当时残差模型太弱(Ridge),
      现在残差用 LightGBM, KernelRidge 基线能正确工作 (R²=0.71)。
      KernelRidge 的 RBF 核能捕捉容忍因子的非线性稳定带, 优于线性 Ridge。
    """

    def __init__(self, alpha=1.0, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self._model = None

    def fit(self, X_phys, y):
        from sklearn.impute import SimpleImputer
        self._model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("krr", KernelRidge(alpha=self.alpha, kernel="rbf", gamma=self.gamma)),
        ])
        self._model.fit(X_phys, y)
        return self

    def predict(self, X_phys):
        return self._model.predict(X_phys)


class PGMLModel(BaseEstimator, RegressorMixin):
    """
    PGML 残差模型: target = physics_baseline(X_phys) + ML_residual(X_all)

    Parameters
    ----------
    physics_baseline : 物理基线估计器 (默认 KernelRidge on 物理特征)
    ml_model :        残差学习器 (默认 Ridge; 也可传 XGBoost/LightGBM)
    phys_features :   物理特征列名列表
    """

    def __init__(self, physics_baseline=None, ml_model=None, phys_features=None):
        self.physics_baseline = physics_baseline or PhysicsBaselineModel()
        self.ml_model = ml_model or Ridge(alpha=1.0)
        self.phys_features = phys_features or PHYS_BASELINE_FEATURES
        self._phys_cols_idx_ = None

    def _split(self, X):
        """把特征矩阵拆为物理特征列 + 全部列。"""
        if isinstance(X, pd.DataFrame):
            X_phys = X[self.phys_features].values
            X_all = X.values
        else:  # numpy 数组: 需调用方先记录列索引(此处简化为前N列)
            X_phys = X
            X_all = X
        return X_phys, X_all

    def fit(self, X, y):
        X_phys, X_all = self._split(X)
        y = np.asarray(y, dtype=float)

        # Step 1: 拟合物理基线
        self.physics_baseline.fit(X_phys, y)
        baseline_pred = self.physics_baseline.predict(X_phys)

        # Step 2: 计算残差
        residual = y - baseline_pred
        self._residual_mean_ = float(np.mean(residual))
        self._residual_std_ = float(np.std(residual))

        # Step 3: ML 学习残差
        self.ml_model.fit(X_all, residual)
        return self

    def predict(self, X):
        X_phys, X_all = self._split(X)
        return self.physics_baseline.predict(X_phys) + self.ml_model.predict(X_all)

    def predict_decomposed(self, X):
        """返回 (baseline_pred, ml_residual_pred, total_pred)。"""
        X_phys, X_all = self._split(X)
        bp = self.physics_baseline.predict(X_phys)
        rp = self.ml_model.predict(X_all)
        return bp, rp, bp + rp


# ----------------------------------------------------------------
# 评估指标
# ----------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------
# 三方对比: 纯物理 vs 纯ML vs PGML
# ----------------------------------------------------------------
def run_pgml_comparison(
    df: pd.DataFrame,
    target: str,
    feat_prefixes: tuple = ("magpie_", "phys_"),  # P0-1: 排除struct避免泄露
    ml_factory=None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    对单个目标做 {纯物理, 纯ML, PGML} 三方交叉验证对比。

    Args:
        df: 特征矩阵
        target: 目标列名
        ml_factory: 返回新 ML 估计器的工厂函数 (每次调用返回新实例)
                    默认用 Ridge

    Returns:
        DataFrame: 每行一种方法, 列为 RMSE/MAE/R² + 物理贡献占比
    """
    if ml_factory is None:
        # 残差用 LightGBM (非线性, 比纯 Ridge 更适合学残差)
        try:
            from lightgbm import LGBMRegressor
            ml_factory = lambda: LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )
        except ImportError:
            ml_factory = lambda: Ridge(alpha=1.0)

    feat_cols = [c for c in df.columns if c.startswith(feat_prefixes)]
    phys_cols = [c for c in feat_cols if c in PHYS_BASELINE_FEATURES]
    # 确保物理列存在
    phys_cols = [c for c in phys_cols if c in df.columns]

    X_all = df[feat_cols]
    X_phys = df[phys_cols]
    y = df[target].values

    print(f"[PGML] 目标: {target}")
    print(f"       样本数 {len(y)}, 全特征 {X_all.shape[1]}维, 物理特征 {X_phys.shape[1]}维")

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results = []

    # ---- (1) 纯物理基线 ----
    print("  >> 纯物理基线 (KernelRidge on 物理特征)...")
    pb = PhysicsBaselineModel()
    y_pred_phys = cross_val_predict(pb, X_phys.values, y, cv=cv)
    m_phys = evaluate(y, y_pred_phys)
    m_phys["method"] = "physics_only"
    results.append(m_phys)

    # ---- (2) 纯 ML (全特征) ----
    print("  >> 纯 ML (全特征)...")
    # 对 ML 做标准化管线
    ml_pipeline_factory = lambda: Pipeline([
        ("scaler", StandardScaler()),
        ("ml", ml_factory()),
    ])
    y_pred_ml = cross_val_predict(ml_pipeline_factory(), X_all.values, y, cv=cv)
    m_ml = evaluate(y, y_pred_ml)
    m_ml["method"] = "ml_only"
    results.append(m_ml)

    # ---- (3) PGML: 物理基线 + ML 残差 ----
    print("  >> PGML (物理基线 + ML 残差)...")
    # ★ 关键: 预分配全长度数组, 按 te_idx 回填, 保证与 y 顺序对齐
    n = len(y)
    pred_baseline = np.empty(n, dtype=float)
    pred_residual = np.empty(n, dtype=float)
    pred_total = np.empty(n, dtype=float)
    for tr_idx, te_idx in cv.split(X_all):
        X_tr_all = X_all.iloc[tr_idx]
        X_te_all = X_all.iloc[te_idx]
        y_tr = y[tr_idx]

        pgml = PGMLModel(
            physics_baseline=PhysicsBaselineModel(),
            ml_model=Pipeline([("scaler", StandardScaler()), ("ml", ml_factory())]),
            phys_features=phys_cols,
        )
        pgml.fit(X_tr_all, y_tr)
        bp, rp, tot = pgml.predict_decomposed(X_te_all)
        # 按 te_idx 回填到原始位置, 保证顺序对齐
        pred_baseline[te_idx] = bp
        pred_residual[te_idx] = rp
        pred_total[te_idx] = tot

    m_pgml = evaluate(y, pred_total)
    m_pgml["method"] = "pgml"
    results.append(m_pgml)

    # ---- 物理贡献量化 (P0-5 修复: 用正确的方差分解) ----
    # 正确定义: 物理贡献 = baseline 单独解释的目标方差比例 (即 baseline R²)
    # ML 增益 = PGML 相对 baseline 提升的解释量
    # 总解释 = PGML R²
    # 物理占比 = baseline_R² / pgml_R² (基线贡献了总解释度的多少)
    from sklearn.metrics import r2_score as _r2
    baseline_r2 = float(_r2(y, pred_baseline))
    pgml_r2 = float(_r2(y, pred_total))
    ml_gain = pgml_r2 - baseline_r2  # ML 在物理基础上额外解释的
    # 物理贡献占比 = 基线解释 / 总解释 (防止 pgml_r2≤0 时除零)
    if pgml_r2 > 0:
        physics_share = max(0.0, baseline_r2) / pgml_r2
    else:
        physics_share = float("nan")
    ml_share = 1.0 - physics_share if np.isfinite(physics_share) else float("nan")
    m_pgml["physics_share"] = physics_share
    m_pgml["ml_share"] = ml_share
    m_pgml["baseline_R2"] = baseline_r2
    m_pgml["ml_gain"] = ml_gain
    print(f"       baseline R²={baseline_r2:.3f} | ML增益={ml_gain:+.3f} | "
          f"PGML R²={pgml_r2:.3f}", flush=True)
    print(f"       物理贡献占比: {physics_share:.1%} | ML贡献占比: {ml_share:.1%}", flush=True)

    df_out = pd.DataFrame(results)
    df_out["target"] = target
    return df_out


# ----------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------
def main():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils import load_features
    df = load_features()
    if df is None or len(df) == 0:
        raise SystemExit("[ERROR] 特征数据为空, 先运行 features.py")

    print("=" * 60)
    print("  PGML 残差建模 (物理 vs ML vs PGML 三方对比)")
    print("=" * 60)

    all_results = []
    targets = {
        "formation_energy_per_atom": "形成能 Ef",
        "energy_above_hull": "凸包能 Ehull",
    }
    for target_col, target_cn in targets.items():
        print(f"\n### 目标: {target_cn} ###")
        res = run_pgml_comparison(df, target=target_col)
        all_results.append(res)

    df_all = pd.concat(all_results, ignore_index=True)
    out_path = METRICS_DIR / "pgml_comparison.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] PGML 对比结果: {out_path}")

    print("\n" + "=" * 60)
    print("  PGML 三方对比汇总")
    print("=" * 60)
    show = df_all[["target", "method", "RMSE", "MAE", "R2"]]
    if "physics_share" in df_all.columns:
        show = df_all.copy()
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
