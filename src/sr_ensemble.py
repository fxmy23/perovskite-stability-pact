"""
================================================================
符号集成 (Symbolic Ensemble) — 解决 SR 非唯一性问题 (问题③)
================================================================
背景: SR 5 折公式结构不一致 (非唯一解, 符号回归已知特性, ACM Survey 2024)。
解决: 跑 N 次 SR (不同 seed), 统计每个物理量在公式中的出现频率,
  报"特征共识度"而非单一公式。把"非唯一"从缺陷转为"集成共识"优势。

★ 增量保存: 每跑完一个 seed 立即追加写盘, 意外中断不丢已完成 seed。
★ 断点续跑: 启动时检查已完成 seed, 跳过。

依赖: gplearn, scikit-learn, numpy, pandas

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
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import load_features, get_feature_cols, PHYS_FEATURES, EXCLUDE_FROM_ML
from src.sr_physics_layer import make_sr_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
ENSEMBLE_DIR = METRICS_DIR / "sr_ensemble"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
SEEDS = [42, 123, 456, 789, 2024, 314, 271, 161, 803, 951]


def extract_features_from_equation(eq, n_features=14):
    """从公式字符串提取出现的 Xi 索引集合。"""
    found = set()
    for i in range(n_features):
        # 匹配 Xi (避免 X1 匹配到 X11 的子串: 用边界)
        token = f"X{i}"
        # 简单匹配: 在 eq 中找 X{i} 后面不是数字
        idx = 0
        while True:
            pos = eq.find(token, idx)
            if pos == -1:
                break
            after = pos + len(token)
            # 检查后面字符不是数字 (避免 X1 匹配 X11)
            if after >= len(eq) or not eq[after].isdigit():
                found.add(i)
            idx = pos + 1
    return found


def run_one_seed(X, y, phys_idx, seed, n_splits=5):
    """跑一个 seed: 5 折 CV, 每折独立 SR, 收集公式 + R²。"""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    equations = []
    r2_physics_list = []
    for fold, (tr, te) in enumerate(cv.split(X)):
        imp = SimpleImputer(strategy="median"); sca = StandardScaler()
        Xtr_p = sca.fit_transform(imp.fit_transform(X[tr][:, phys_idx]))
        Xte_p = sca.transform(imp.transform(X[te][:, phys_idx]))
        sr = make_sr_model(parsimony=0.001, seed=seed * 100 + fold)
        sr.fit(Xtr_p, y[tr])
        eq = str(sr._program)
        equations.append(eq)
        r2_physics_list.append(float(r2_score(y[te], sr.predict(Xte_p))))
    return {
        "seed": seed,
        "equations": equations,
        "r2_physics_mean": float(np.mean(r2_physics_list)),
        "r2_physics_std": float(np.std(r2_physics_list)),
    }


def main():
    df = load_features()
    feat = get_feature_cols(df, exclude_struct=True, exclude_derived=True)
    phys = [c for c in PHYS_FEATURES if c not in EXCLUDE_FROM_ML and c in df.columns]
    phys_idx = [i for i, c in enumerate(feat) if c in phys]
    phys_short = [c.replace("phys_", "") for c in phys]
    X = df[feat].values
    y = df["formation_energy_per_atom"].values

    print("=" * 64, flush=True)
    print(f"  符号集成 ({N_SEEDS} seeds × 5 folds = {N_SEEDS*5} 个公式)", flush=True)
    print(f"  增量保存到 {ENSEMBLE_DIR}", flush=True)
    print("=" * 64, flush=True)

    # 断点续跑: 检查已完成 seed
    done_seeds = set()
    for f in ENSEMBLE_DIR.glob("seed_*.json"):
        try:
            s = int(f.stem.split("_")[1])
            done_seeds.add(s)
        except Exception:
            pass
    todo = [s for s in SEEDS if s not in done_seeds]
    if done_seeds:
        print(f"  已完成 {len(done_seeds)} seed: {sorted(done_seeds)}", flush=True)
    print(f"  待跑 {len(todo)} seed: {todo}", flush=True)

    # 逐 seed 跑 (增量保存)
    for seed in todo:
        print(f"\n  >> seed={seed} ...", flush=True)
        try:
            res = run_one_seed(X, y, phys_idx, seed)
            (ENSEMBLE_DIR / f"seed_{seed}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"     R²(物理层)={res['r2_physics_mean']:.4f}±{res['r2_physics_std']:.4f}", flush=True)
            print(f"     fold1公式: {res['equations'][0][:80]}", flush=True)
            print(f"     [SAVE] seed_{seed}.json", flush=True)
        except Exception as e:
            print(f"     [ERROR] {type(e).__name__}: {str(e)[:150]}", flush=True)

    # ===== 汇总: 特征共识度 =====
    print("\n" + "=" * 64, flush=True)
    print("  特征共识度分析 (N_seed × 5 fold 公式中各物理量出现频率)", flush=True)
    print("=" * 64, flush=True)

    all_eqs = []
    all_r2 = []
    for f in sorted(ENSEMBLE_DIR.glob("seed_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        all_eqs.extend(d["equations"])
        all_r2.append(d["r2_physics_mean"])

    n_total = len(all_eqs)
    feature_counts = {i: 0 for i in range(len(phys))}
    for eq in all_eqs:
        found = extract_features_from_equation(eq, len(phys))
        for i in found:
            feature_counts[i] += 1

    print(f"\n  共 {n_total} 个公式 (物理层 R²={np.mean(all_r2):.4f}±{np.std(all_r2):.4f}):", flush=True)
    print(f"\n  {'物理量':<28} {'出现频率':>10} {'共识度':>8}", flush=True)
    print("  " + "-" * 50, flush=True)
    consensus_rows = []
    for i in sorted(feature_counts.keys(), key=lambda k: -feature_counts[k]):
        freq = feature_counts[i]
        pct = freq / n_total * 100
        bar = "█" * int(pct / 5)
        short = phys_short[i] if i < len(phys_short) else f"X{i}"
        print(f"  {short:<28} {freq:>5}/{n_total:<4} {pct:>6.1f}% {bar}", flush=True)
        consensus_rows.append({
            "feature": short,
            "feature_full": phys[i] if i < len(phys) else f"X{i}",
            "appearances": freq,
            "total_formulas": n_total,
            "consensus_pct": round(pct, 1),
        })

    # 持久化
    df_consensus = pd.DataFrame(consensus_rows)
    df_consensus.to_csv(METRICS_DIR / "sr_ensemble_consensus.csv",
                        index=False, encoding="utf-8-sig")
    # R² 汇总
    pd.DataFrame({"seed": SEEDS[:len(all_r2)], "r2_physics": all_r2}).to_csv(
        METRICS_DIR / "sr_ensemble_r2.csv", index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] sr_ensemble_consensus.csv, sr_ensemble_r2.csv", flush=True)

    # 高共识特征 (>50%)
    high = df_consensus[df_consensus["consensus_pct"] > 50]
    print(f"\n  高共识特征 (>50% 出现):", flush=True)
    for _, r in high.iterrows():
        print(f"    {r['feature']}: {r['consensus_pct']:.0f}% → 强物理信号", flush=True)


if __name__ == "__main__":
    main()
