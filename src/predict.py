"""
================================================================
钙钛矿性质预测工具 (用户交互接口)
================================================================
论文 P1-a: 提供可用的预测工具, 提升实用性。

用法:
    # 单个化学式
    python src/predict.py LaTiO3

    # 多个化学式
    python src/predict.py LaTiO3 BaTiO3 SrTiO3

    # 交互模式
    python src/predict.py --interactive

输出:
    化学式 → 形成能 Ef / 凸包能 Ehull / 稳定性判断 / 物理特征

依赖: 训练好的 LightGBM 模型 (首次运行自动训练并缓存)

作者: 清华大学材料学院本科生
================================================================
"""

from __future__ import annotations

import sys
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols
from src.features import generate_magpie_features, generate_physical_features

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "results" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STABLE_THRESH = 0.05
TARGETS = ["formation_energy_per_atom", "energy_above_hull"]


def _train_or_load_model(target):
    """训练或加载缓存模型。"""
    cache = MODELS_DIR / f"predict_model_{target}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    print(f"[MODEL] 训练 {target} 模型 (首次运行, 后续缓存)...", flush=True)
    df = load_features()
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y = df[target].values

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lgbm", LGBMRegressor(
            n_estimators=200, num_leaves=31, learning_rate=0.1,
            random_state=42, n_jobs=1, verbose=-1,
        )),
    ])
    model.fit(X, y)

    with open(cache, "wb") as f:
        pickle.dump({"model": model, "feat_cols": feat_cols}, f)
    print(f"[MODEL] 已缓存: {cache}", flush=True)
    return {"model": model, "feat_cols": feat_cols}


def predict_formula(formula, a_site=None, b_site=None):
    """对单个化学式做预测。"""
    # 加载模型
    pkg_ef = _train_or_load_model("formation_energy_per_atom")
    pkg_ehull = _train_or_load_model("energy_above_hull")
    feat_cols = pkg_ef["feat_cols"]

    # 生成特征
    mag = generate_magpie_features([formula])
    phys = generate_physical_features([formula], [a_site or ""], [b_site or ""])
    df_feat = pd.concat([mag.reset_index(drop=True),
                         phys.reset_index(drop=True)], axis=1)
    # 对齐特征列
    for c in feat_cols:
        if c not in df_feat.columns:
            df_feat[c] = np.nan
    X = df_feat[feat_cols].values

    # 预测
    eform = float(pkg_ef["model"].predict(X)[0])
    ehull = float(pkg_ehull["model"].predict(X)[0])

    # 物理特征
    t = float(phys["phys_tolerance_factor"].iloc[0]) if "phys_tolerance_factor" in phys.columns else np.nan
    mu = float(phys["phys_octahedral_factor"].iloc[0]) if "phys_octahedral_factor" in phys.columns else np.nan
    in_zone = int(phys["phys_in_stable_zone"].iloc[0]) if "phys_in_stable_zone" in phys.columns else 0

    # 稳定性判断
    is_stable = ehull < STABLE_THRESH

    return {
        "formula": formula,
        "Eform_eV": eform,
        "Ehull_eV": ehull,
        "stable": is_stable,
        "tolerance_factor_t": t,
        "octahedral_factor_mu": mu,
        "in_classic_zone": bool(in_zone),
    }


def format_result(r):
    """格式化输出。"""
    stable_str = "✓ 稳定" if r["stable"] else "✗ 不稳定"
    zone_str = "在经典区" if r["in_classic_zone"] else "外经典区"
    lines = [
        f"\n{'='*50}",
        f"  化学式: {r['formula']}",
        f"{'='*50}",
        f"  形成能 Ef:      {r['Eform_eV']:+.3f} eV/atom",
        f"  凸包能 Ehull:    {r['Ehull_eV']:.3f} eV/atom  (< 0.05 = 稳定)",
        f"  稳定性判断:      {stable_str}",
        f"  容忍因子 t:      {r['tolerance_factor_t']:.3f}" if np.isfinite(r["tolerance_factor_t"]) else "  容忍因子 t:      N/A",
        f"  八面体因子 μ:    {r['octahedral_factor_mu']:.3f}" if np.isfinite(r["octahedral_factor_mu"]) else "  八面体因子 μ:    N/A",
        f"  Goldschmidt区:   {zone_str} (t∈[0.8,1.0] & μ∈[0.414,0.732])",
        f"{'='*50}",
    ]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python src/predict.py <化学式> [化学式2 ...]")
        print("示例: python src/predict.py LaTiO3 BaTiO3")
        print("交互: python src/predict.py --interactive")
        return

    if sys.argv[1] == "--interactive":
        print("交互模式 (输入化学式, q退出):")
        while True:
            f = input("\n化学式> ").strip()
            if f.lower() in ("q", "quit", "exit"):
                break
            if not f:
                continue
            try:
                r = predict_formula(f)
                print(format_result(r))
            except Exception as e:
                print(f"  错误: {e}")
        return

    for formula in sys.argv[1:]:
        try:
            r = predict_formula(formula)
            print(format_result(r))
        except Exception as e:
            print(f"\n{formula}: 错误 - {e}")


if __name__ == "__main__":
    main()
