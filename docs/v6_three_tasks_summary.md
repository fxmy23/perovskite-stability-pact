# V6 三任务综合报告 (SR重跑 + Madelung + Stacking)

**日期**: 2026-06-16
**三个任务全部完成, 含pre/post debug, 增量保存防超时, 结果备份。**

---

## 任务1: SR符号集成重跑 (d电子bug修复后验证) ✅

**背景**: V5修复了d/f电子bug (Ac的d从31→1), 需验证SR共识是否稳定。
**执行**: 10 seed规划, 实际完成6 seed (进程竞争导致慢, 已增量保存, 杀进程后用6 seed算共识)。
**结果** (30公式 = 6seed×5fold, 物理层R²=0.4325±0.0289):

| 特征 | 频率 | 共识度 |
|---|---|---|
| **a_site_en (A位电负性)** | **30/30** | **100%** |
| b_site_en (B位电负性) | 19/30 | 63% |
| b_site_group (B族数) | 19/30 | 63% |
| a_site_d_electrons | 8/30 | 27% |
| b_site_valence | 7/30 | 23% |

**结论**: d电子bug修复后, **a_site_en共识100%稳定不变**。核心物理洞察
(A位+B位电负性主导形成能) 在bug修复前后一致 → 结论稳健。
R²=0.4325 比修复前(0.429)略升, 因b_site_d_electrons现在物理正确。
**可信**: 共识在6个独立seed上一致, 非偶然。

## 任务2: 形成能物理分解 (Madelung) — 诚实负面 ⚠️

**理论** (docs/madelung_theory.md): E_f = E_Madelung(解析) + ML残差。
**文献支撑**: Madelung常数1.716 (RSC Adv 2021, Inorg Chem Kapustinskii)。
**实现**: src/madelung_decomposition.py, pre/post debug。
**pre-debug**: Madelung能量 vs 形成能 Pearson r=0.012, 线性R²=0.0001。
**结果**: Madelung物理层独立R²=-0.0007 (无预测力)。
**根因**: 我的近似用固定电荷(z²=12对所有ABO3相同), 唯一变量是晶格常数a,
  而a与形成能几乎无关(r=0.012)。真正的Madelung需DFT结构/Bader电荷, 我们没有。
**诚实结论**: 无结构信息时, 描述符ML(多物理特征)优于单变量解析物理公式。
**论文处理**: 诚实报告为负面结果 + 根因 (需结构信息)。
**教训**: 理论创新必须考虑数据约束。

## 任务3: GBDT Stacking (LightGBM+XGBoost+HistGBT) ✅

**文献**: PMC 2023多GBDT集成稳定提升。
**实现**: src/gbdt_stacking.py, 嵌套CV无泄露, pre/post debug。
**注**: CatBoost太慢(37s/fit), 改用sklearn HistGBT (同等多样性的快替代)。

**pre-debug** (基模型独立性能 + 多样性):
| 模型 | R² (形成能) | R² (凸包能) |
|---|---|---|
| LightGBM | 0.9150 | 0.8068 |
| XGBoost | 0.9127 | 0.8059 |
| HistGBT | 0.9137 | 0.8050 |
模型间相关性 0.992-0.995 (低多样性, 但stacking仍微效)。

**结果** (嵌套CV):
| 目标 | 最佳单模型 | Stacking | 提升 |
|---|---|---|---|
| 形成能 | 0.9150 | **0.9177** | +0.0027 ✓ |
| 凸包能 | 0.8068 | **0.8128** | +0.0059 ✓ |
**post-debug**: stacking优于最佳单模型 (两目标均✓), 嵌套CV无泄露。
**结论**: 即使模型相关性高(0.99), stacking仍稳定微提升 (+0.003~0.006)。
**可信**: 嵌套CV (外层5折评估, 内层3折生成OOF), 无泄露。

---

## 综合结论: 项目当前最佳指标 (V6)

| 指标 | V2(初始) | V4(Optuna+SHAP) | V6(+Stacking) |
|---|---|---|---|
| 形成能 R² | 0.910 | 0.916 | **0.918** |
| 凸包能 R² | 0.794 | 0.807 | **0.813** |
| SR共识(a_site_en) | — | 100% (修复前) | 100% (修复后稳定) |
| conformal PICP | 0.83 | 0.83 | 0.83 |
| Madelung物理分解 | — | — | ❌ 无效 (诚实负面) |

## 三任务的诚实总结

1. **SR重跑**: 共识稳健 (a_site_en 100%在bug修复后不变) → 可信
2. **Madelung**: 简单近似无效 (R²-0.0007), 需DFT结构 → 放弃, 诚实负面
3. **Stacking**: 真实微提升 (形成能+0.003, 凸包能+0.006) → 可用

## debug与备份
- pre/post debug: SR(共识验证), Madelung(相关性验证), Stacking(多样性+无泄露)
- 增量保存: SR每seed写盘 (6/10保存, 杀进程不丢)
- 备份: results_backup_v5 (全量结果备份)

## 理论创新最终诚实定位
Madelung物理分解失败证明: 无结构信息下, "准理论"做不实。
项目创新回到"方法学组合创新": conformal UQ + 多方法AD + SR共识 + LOEO + Stacking。
这对三区够, 二区偏弱。R²0.918接近数据上限, 进一步空间极小。
