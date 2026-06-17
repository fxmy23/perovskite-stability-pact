"""
================================================================
稳定性分类评估模块 (Matbench Discovery 标准指标)
================================================================
论文 P0-a: 把稳定性预测从纯回归重构为"回归+分类"双视角。

为什么需要分类指标:
  材料发现的本质是分类(稳定/不稳定), 不只是回归(E_hull值)。
  Matbench Discovery 官方标准包含 F1/Precision/Recall/DAF,
  审稿人会直接问"你的模型判断稳定的准确率是多少"。

指标定义 (对齐 Matbench Discovery):
  - Precision = TP / (TP+FP): 预测稳定中真稳定比例
  - Recall(TPR) = TP / (TP+FN): 真稳定中被找出比例
  - F1 = 2·P·R/(P+R): 精确率与召回的调和
  - FPR = FP / (FP+TN): 假阳性率
  - AUC-ROC: ROC曲线下面积 (分类判别力)
  - DAF (Discovery Acceleration Factor): 发现加速因子
    = (top-k中稳定材料密度) / (全空间稳定材料密度)
    DAF>1 表示比随机好, DAF=2.70 是 M3GNet 标杆

阈值: E_hull < 0.05 eV/atom 视为稳定 (文献常用)

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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)

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

STABLE_THRESH = 0.05  # E_hull < 0.05 eV/atom 视为稳定


def compute_daf(y_true, y_pred, thresh=STABLE_THRESH, top_frac=0.1):
    """
    Discovery Acceleration Factor (发现加速因子)。
    DAF = (top-k%预测中真稳定比例) / (全空间真稳定比例)

    含义: 用模型筛选 top-k% 候选, 比随机挑选能多发现几倍稳定材料。
    """
    truly_stable = (y_true < thresh)
    base_rate = truly_stable.mean()  # 全空间稳定率
    if base_rate == 0:
        return float("nan")

    # 按预测 E_hull 升序(最可能稳定的在前), 取 top-k%
    n_top = max(1, int(len(y_pred) * top_frac))
    top_idx = np.argsort(y_pred)[:n_top]
    top_stable_rate = truly_stable[top_idx].mean()
    return float(top_stable_rate / base_rate)


def run_classification_eval(df, target="energy_above_hull", n_splits=5,
                            random_state=42):
    """
    用回归模型预测 E_hull, 然后用阈值分类, 计算全套分类指标。
    """
    feat_cols = get_feature_cols(df, exclude_struct=True)
    X = df[feat_cols].values
    y_reg = df[target].values
    y_cls = (y_reg < STABLE_THRESH).astype(int)  # 二分类标签

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n = len(y_reg)

    # 预分配 OOF 预测
    oof_pred = np.empty(n)
    base_rate = y_cls.mean()
    print(f"[CLASSIFY] 稳定阈值 E_hull<{STABLE_THRESH}", flush=True)
    print(f"           全空间稳定率: {base_rate:.2%} ({y_cls.sum()}/{n})", flush=True)

    for fold, (tr, te) in enumerate(cv.split(X)):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                random_state=42, n_jobs=1, verbose=-1,
            )),
        ])
        model.fit(X[tr], y_reg[tr])
        oof_pred[te] = model.predict(X[te])

    # 用预测值做分类
    pred_stable = (oof_pred < STABLE_THRESH).astype(int)

    # 分类指标
    precision = precision_score(y_cls, pred_stable, zero_division=0)
    recall = recall_score(y_cls, pred_stable, zero_division=0)
    f1 = f1_score(y_cls, pred_stable, zero_division=0)

    # 混淆矩阵元素
    TP = int(((pred_stable == 1) & (y_cls == 1)).sum())
    FP = int(((pred_stable == 1) & (y_cls == 0)).sum())
    FN = int(((pred_stable == 0) & (y_cls == 1)).sum())
    TN = int(((pred_stable == 0) & (y_cls == 0)).sum())
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    # AUC-ROC (用连续预测值, 不用阈值化)
    auc = roc_auc_score(y_cls, -oof_pred)  # 负号: E_hull越低越可能稳定

    # DAF (多个 top-fraction)
    daf_10 = compute_daf(y_reg, oof_pred, top_frac=0.10)
    daf_05 = compute_daf(y_reg, oof_pred, top_frac=0.05)
    daf_01 = compute_daf(y_reg, oof_pred, top_frac=0.01)

    results = {
        "stable_rate": base_rate,
        "Precision": precision,
        "Recall_TPR": recall,
        "F1": f1,
        "FPR": fpr,
        "AUC_ROC": auc,
        "DAF_top10%": daf_10,
        "DAF_top5%": daf_05,
        "DAF_top1%": daf_01,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    }

    print(f"\n  === 分类指标 (稳定性预测) ===", flush=True)
    print(f"  Precision:  {precision:.3f}  (预测稳定中真稳定比例)", flush=True)
    print(f"  Recall:     {recall:.3f}  (真稳定中被找出比例)", flush=True)
    print(f"  F1 Score:   {f1:.3f}", flush=True)
    print(f"  FPR:        {fpr:.3f}  (假阳性率)", flush=True)
    print(f"  AUC-ROC:    {auc:.3f}  (分类判别力, 0.5=随机, 1=完美)", flush=True)
    print(f"  混淆矩阵: TP={TP} FP={FP} FN={FN} TN={TN}", flush=True)
    print(f"\n  === Discovery Acceleration Factor ===", flush=True)
    print(f"  DAF(top 10%): {daf_10:.2f}x  (比随机快{daf_10:.1f}倍)", flush=True)
    print(f"  DAF(top 5%):  {daf_05:.2f}x", flush=True)
    print(f"  DAF(top 1%):  {daf_01:.2f}x", flush=True)
    print(f"  注: DAF 对基础率敏感(本数据集稳定率{base_rate:.1%})", flush=True)
    print(f"  不可与不同数据集的 DAF 直接比较 (如 M3GNet 在异构数据集上的 2.70)", flush=True)

    return results, oof_pred, y_cls


def main():
    df = load_features()
    print("=" * 60, flush=True)
    print("  稳定性分类评估 (Matbench Discovery 标准指标)", flush=True)
    print("=" * 60, flush=True)

    results, oof_pred, y_cls = run_classification_eval(df)

    # 保存指标
    df_out = pd.DataFrame([results])
    out_path = METRICS_DIR / "classification_metrics.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}", flush=True)

    # 保存 OOF 预测 (供 PR/ROC 曲线绘制)
    df_oof = pd.DataFrame({
        "y_true_cls": y_cls,
        "y_pred_ehull": oof_pred,
    })
    oof_path = METRICS_DIR / "classification_oof.csv"
    df_oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {oof_path} (供 PR/ROC 曲线)", flush=True)

    # 解读
    print("\n" + "=" * 60, flush=True)
    print("  解读 (审稿人视角)", flush=True)
    print("=" * 60, flush=True)
    if results["F1"] > 0.5:
        print(f"  F1={results['F1']:.2f} > 0.5: 稳定性分类有效 ✓", flush=True)
    else:
        print(f"  F1={results['F1']:.2f} < 0.5: 稳定性分类需改进 (类别不平衡影响)", flush=True)
    if results["AUC_ROC"] > 0.8:
        print(f"  AUC={results['AUC_ROC']:.2f} > 0.8: 判别力强 ✓", flush=True)
    elif results["AUC_ROC"] > 0.7:
        print(f"  AUC={results['AUC_ROC']:.2f}: 判别力中等", flush=True)
    if results["DAF_top10%"] > 2.0:
        print(f"  DAF={results['DAF_top10%']:.1f} > 2.0: 发现加速显著 ✓", flush=True)


if __name__ == "__main__":
    main()
