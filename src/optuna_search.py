"""
================================================================
优化②: Optuna 超参搜索 (提升 CV R²)
================================================================
文献支撑: MDPI Materials 2025 R²=0.928 就靠系统超参搜索。
当前我们用 LightGBM 默认/手调超参, R²=0.910。

本模块: Optuna 贝叶斯优化 LightGBM 超参, 嵌套 CV 评估 (无泄露):
  - 外层 5 折 CV 评估 (固定, 与主线一致)
  - 每个外层折内, Optuna 在训练折上做 3 折内部 CV 选超参
  - 用选出的超参在训练折 fit, 在测试折 predict
  → OOF R² 是无偏估计 (超参选择不接触测试折)

搜索空间 (LightGBM 关键超参):
  - n_estimators: 100-500
  - num_leaves: 15-127
  - learning_rate: 0.01-0.3 (log)
  - max_depth: -1, 4-12
  - min_child_samples: 5-50
  - subsample: 0.5-1.0
  - colsample_bytree: 0.5-1.0
  - reg_alpha: 1e-3 to 10 (log)
  - reg_lambda: 1e-3 to 10 (log)

依赖: optuna, lightgbm, scikit-learn, numpy, pandas

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def sample_params(trial):
    """LightGBM 超参搜索空间。"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "n_jobs": 1, "verbose": -1,
    }


def make_model(params, seed=42):
    return LGBMRegressor(random_state=seed, **params)


def nested_cv_optuna(X, y, target_name, n_outer=5, n_trials=30, seed=42):
    """
    嵌套 CV: 外层评估, 内层 Optuna 选超参。
    每个外层折独立 Optuna 搜索 → 超参不接触该折测试集 (无泄露)。
    """
    print(f"\n[Optuna-嵌套CV] {target_name}, {n_trials} trials/折, {n_outer}外折", flush=True)
    outer_cv = KFold(n_outer, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    best_params_per_fold = []
    baseline_oof = np.full(len(y), np.nan)  # 默认超参对照

    for fold, (tr, te) in enumerate(outer_cv.split(X)):
        print(f"  外折 {fold+1}/{n_outer} ...", flush=True)
        # 预处理 (折内)
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr = sca.fit_transform(imp.fit_transform(X[tr]))
        Xte = sca.transform(imp.transform(X[te]))

        # 默认超参对照
        base = LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.1,
                             random_state=seed, n_jobs=1, verbose=-1,
                             subsample=0.8, subsample_freq=1).fit(Xtr, y[tr])
        baseline_oof[te] = base.predict(Xte)

        # Optuna 内层 CV
        inner_cv = KFold(3, shuffle=True, random_state=seed + fold)

        def objective(trial):
            params = sample_params(trial)
            inner_scores = []
            for itr, ite in inner_cv.split(Xtr):
                m = make_model(params, seed).fit(Xtr[itr], y[tr][itr])
                inner_scores.append(r2_score(y[tr][ite], m.predict(Xtr[ite])))
            return np.mean(inner_scores)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed + fold))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params_per_fold.append({"fold": fold + 1, **best_params,
                                      "inner_r2": study.best_value})
        print(f"    最佳内层R²={study.best_value:.4f}, "
              f"n_est={best_params['n_estimators']}, "
              f"lr={best_params['learning_rate']:.3f}, "
              f"leaves={best_params['num_leaves']}", flush=True)

        # 用最佳超参在外层训练折 fit, 测试折 predict
        best_model = make_model(best_params, seed).fit(Xtr, y[tr])
        oof[te] = best_model.predict(Xte)

    r2_opt = r2_score(y, oof)
    r2_base = r2_score(y, baseline_oof)
    print(f"\n  结果:", flush=True)
    print(f"    默认超参 R²={r2_base:.4f}", flush=True)
    print(f"    Optuna  R²={r2_opt:.4f}", flush=True)
    print(f"    提升: {r2_opt-r2_base:+.4f}", flush=True)
    return {"target": target_name, "r2_default": r2_base, "r2_optuna": r2_opt,
            "improvement": r2_opt - r2_base,
            "best_params_per_fold": best_params_per_fold,
            "mae_default": mean_absolute_error(y, baseline_oof),
            "mae_optuna": mean_absolute_error(y, oof)}


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    X = df[feat].values

    print("=" * 64, flush=True)
    print("  优化②: Optuna 超参搜索 (嵌套CV, 无泄露)", flush=True)
    print("=" * 64, flush=True)

    all_res = []
    # n_trials=15 (实测每trial~3s, 5foldx15=可接受)
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        y = df[target].values
        res = nested_cv_optuna(X, y, target, n_trials=15)
        all_res.append(res)

    # 持久化
    summary = []
    for r in all_res:
        summary.append({"target": r["target"], "r2_default": r["r2_default"],
                        "r2_optuna": r["r2_optuna"], "improvement": r["improvement"],
                        "mae_default": r["mae_default"], "mae_optuna": r["mae_optuna"]})
    pd.DataFrame(summary).to_csv(METRICS_DIR / "optuna_results.csv",
                                  index=False, encoding="utf-8-sig")
    # 超参详情
    for r in all_res:
        pd.DataFrame(r["best_params_per_fold"]).to_csv(
            METRICS_DIR / f"optuna_best_params_{r['target']}.csv",
            index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] optuna_results.csv + 超参详情", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("  汇总", flush=True)
    print("=" * 64, flush=True)
    for r in all_res:
        print(f"  {r['target']}: R² {r['r2_default']:.4f} → {r['r2_optuna']:.4f} "
              f"({r['improvement']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
