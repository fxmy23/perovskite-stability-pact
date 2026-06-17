"""
================================================================
外推性测试模块 (Leave-One-Element-Out, LOEO)
================================================================
论文 P0-2 + 创新点 A: 评估模型对"训练集未见元素"的外推预测能力。

为什么需要 LOEO:
  随机 K 折 CV 测的是"插值"(测试集元素在训练集出现过),
  但真实材料发现场景是"外推"(预测含全新元素的材料)。
  Nature Comm. Mater. 2024 指出这是材料 ML 普遍被高估的环节。

LOEO 协议:
  对每种元素 E (如 La, Ti):
    1. 排除所有含 E 的化合物 → 训练集
    2. 在含 E 的化合物上测试 → 测试集
    3. 报告该元素的外推性能 (RMSE/MAE)
  最终得到"每种元素的外推难度" → 揭示模型适用域 (applicability domain)

预期发现:
  - 常见元素 (La, Ti, O...): 外推好 (训练集有大量同类)
  - 稀有元素 (Pa, Ac, 锕系...): 外推差 (化学空间稀疏)
  - 这本身就是论文的物理发现: 模型适用域与元素丰度相关

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def get_elements_in_compound(elements_str: str) -> set:
    """解析 elements 列 (如 'Ac-Ag-O') 为元素集合。"""
    return set(e.strip() for e in str(elements_str).split("-") if e.strip())


def run_loeo(
    df: pd.DataFrame,
    target: str,
    model_factory=None,
    min_test_samples: int = 20,
) -> pd.DataFrame:
    """
    Leave-One-Element-Out 外推测试。

    Args:
        df: 特征 DataFrame
        target: 目标列
        model_factory: 返回新模型的工厂函数 (默认 LightGBM)
        min_test_samples: 元素测试集最少样本数 (太少则跳过, 避免噪声)

    Returns:
        DataFrame: 每行一种元素, 列为该元素的测试样本数/RMSE/MAE/R²
    """
    if model_factory is None:
        if HAS_LGBM:
            model_factory = lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("lgbm", LGBMRegressor(
                    n_estimators=200, num_leaves=31, learning_rate=0.1,
                    random_state=42, n_jobs=1, verbose=-1,
                )),
            ])
        else:
            model_factory = lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestRegressor(
                    n_estimators=100, random_state=42, n_jobs=1)),
            ])

    feat_cols = get_feature_cols(df, exclude_struct=True)
    X_all = df[feat_cols].values
    y_all = df[target].values

    # 解析每个样本的元素集合
    elements_col = df["elements"] if "elements" in df.columns else None
    if elements_col is None:
        # 从 a_site/b_site 构造
        elements_col = df["a_site_element"] + "-" + df["b_site_element"] + "-O"
    sample_elems = [get_elements_in_compound(e) for e in elements_col]

    # 统计每种元素出现次数
    from collections import Counter
    elem_counts = Counter()
    for s in sample_elems:
        for e in s:
            elem_counts[e] += 1

    # 只测试出现次数 >= min_test_samples 的元素 (统计可靠)
    test_elements = [e for e, c in elem_counts.items() if c >= min_test_samples]
    print(f"[LOEO] 目标: {target}", flush=True)
    print(f"       测试 {len(test_elements)} 种元素 (出现>={min_test_samples}次)", flush=True)

    results = []
    for i, elem in enumerate(sorted(test_elements)):
        # 划分: 不含 elem 的训练, 含 elem 的测试
        test_mask = np.array([elem in s for s in sample_elems])
        train_mask = ~test_mask

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_train < 100 or n_test < min_test_samples:
            continue

        X_tr, X_te = X_all[train_mask], X_all[test_mask]
        y_tr, y_te = y_all[train_mask], y_all[test_mask]

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae = float(mean_absolute_error(y_te, y_pred))
        r2 = float(r2_score(y_te, y_pred))

        results.append({
            "element": elem,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "test_frac": float(n_test / len(df)),
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        })
        if (i + 1) % 10 == 0:
            print(f"       [{i+1}/{len(test_elements)}] 完成", flush=True)

    df_out = pd.DataFrame(results).sort_values("R2", ascending=False)
    return df_out


def main():
    df = load_features()
    if "elements" not in df.columns:
        # wolverton 数据没有 elements 列, 用 a/b 位构造
        df["elements"] = df["a_site_element"].astype(str) + "-" + \
                         df["b_site_element"].astype(str) + "-O"

    print("=" * 60, flush=True)
    print("  外推性测试 (Leave-One-Element-Out)", flush=True)
    print("=" * 60, flush=True)

    all_results = {}
    for target in ["formation_energy_per_atom", "energy_above_hull"]:
        print(f"\n### {target} ###", flush=True)
        df_loeo = run_loeo(df, target=target, min_test_samples=20)
        all_results[target] = df_loeo

        out_path = METRICS_DIR / f"loeo_{target}.csv"
        df_loeo.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [SAVE] {out_path}  ({len(df_loeo)} 种元素)", flush=True)

        # 汇总统计
        print(f"\n  === {target} 外推性能汇总 ===", flush=True)
        print(f"  元素数: {len(df_loeo)}", flush=True)
        print(f"  R² 中位数: {df_loeo['R2'].median():.3f}", flush=True)
        print(f"  R² 范围: [{df_loeo['R2'].min():.3f}, {df_loeo['R2'].max():.3f}]", flush=True)
        print(f"  外推最好 Top5 元素:", flush=True)
        print(df_loeo.head(5)[["element", "n_test", "R2", "MAE"]].to_string(index=False), flush=True)
        print(f"  外推最差 Top5 元素:", flush=True)
        print(df_loeo.tail(5)[["element", "n_test", "R2", "MAE"]].to_string(index=False), flush=True)

    # 对比: 随机 CV vs LOEO
    print("\n" + "=" * 60, flush=True)
    print("  插值(随机CV) vs 外推(LOEO) 对比", flush=True)
    print("=" * 60, flush=True)
    # 读取随机 CV 结果
    baseline_path = METRICS_DIR / "baseline_metrics.csv"
    if baseline_path.exists():
        df_base = pd.read_csv(baseline_path)
        for target in ["formation_energy_per_atom", "energy_above_hull"]:
            rand_r2 = df_base[df_base["target"] == target]["R2"].max()
            loeo_r2 = all_results[target]["R2"].median()
            gap = rand_r2 - loeo_r2
            print(f"  {target}:", flush=True)
            print(f"    随机CV R²(插值) = {rand_r2:.3f}", flush=True)
            print(f"    LOEO R²中位数(外推) = {loeo_r2:.3f}", flush=True)
            print(f"    插值-外推差距 = {gap:.3f} (差距越大, 外推越难)", flush=True)


if __name__ == "__main__":
    main()
