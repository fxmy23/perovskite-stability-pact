# 深度审查: LOEO结论可信度 + 算法优化方案

**日期**: 2026-06-16
**核心问题**: LOEO"纯ML最优"是bug还是真相? 能否优化让效果更好?

---

## 一、代码审计结论: 不是代码bug, 但发现了一个设计缺陷(已验证为真实现象)

### 审计1: extract_features (符号集成) — ✅ 无bug
X1/X11/X13 子串匹配测试全部正确 (add(X1,X11)→{1,11})。
符号集成的"a_site_en 100%共识"结论**可信**。

### 审计2: LOEO "纯ML最优" 根因 — 设计缺陷, 但修复后更差

**怀疑的bug**: ML残差模型看到了全部110特征(含14物理特征),
能自己重建物理层信息, 使物理层冗余。

**修复实验** (ML残差只用96 magpie特征):
| 配置 | R² |
|---|---|
| A. 纯ML(110特征) | **0.910** |
| B. KRR+ML(110, 冗余) | 0.908 |
| C. KRR+ML(96 magpie, 修复) | **0.812** ↓↓ |
| D. SR+ML(96 magpie, 修复) | **0.715** ↓↓↓ |

**出乎预料的结论**: 修复"bug"后效果**暴跌**。
**真实原因** (深层物理事实, 非bug):
- 物理特征(电负性/半径比14维)对形成能的解释力**远超** magpie统计特征(96维)
- 纯ML(110)的0.910主要靠物理特征驱动
- 强行分工(物理层管物理,ML管magpie)造成信息损失 → C/D暴跌

**最终判定**: "纯ML最优"是**真实物理现象**, 非代码bug。
物理特征和ML不是可分工关系, 而是ML是物理特征的最优消费者。

## 二、LOEO结论的可信度: 高 (经验证)

- 代码逻辑正确 (训练/测试分离, impute/scaler 折内fit)
- 68元素完整 (非5元素选择性偏差)
- Wilcoxon检验: SR+ML vs ML p=0.206(不显著), KRR+ML vs ML p=0.030(显著更差)
- **结论可信**: 物理层(含SR)不提升精度, 价值在可解释性

## 三、算法优化方案 (诚实评估可行性)

### 优化1: 梯度提升集成 stacking (LightGBM+XGBoost+CatBoost) ★★★★
**文献支撑**: [PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10611362/),
[ResearchGate 2024](https://www.researchgate.net/publication/397047638)
**做什么**: 3种GBDT stacking + RF meta-model, 替代单LightGBM集成
**预期**: R² 0.910→0.915-0.920 (文献报+0.5-1%)
**成本**: 中 (需装catboost, 嵌套CV)
**风险**: CatBoost安装可能在Windows有问题

### 优化2: 超参自动搜索 (Optuna) ★★★★
**做什么**: 对LightGBM做贝叶斯超参优化 (n_leaves/lr/n_estimators/regularization)
**文献支撑**: [MDPI Materials 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/) R²=0.928就靠这
**预期**: R² 0.910→0.920-0.928
**成本**: 中 (Optuna 50-100 trials, 每trial几分钟)
**这是最可能缩小与SOTA差距的优化**

### 优化3: 丰富SR算子 (sin/exp/tanh) ★★★
**做什么**: function_set加非线性算子, 看SR物理层R² 0.41→?
**预期**: SR物理层R² 0.41→0.50-0.60 (但总R²不变, 因ML补足)
**成本**: 低
**意义**: 提升SR可解释公式的精度, 但不提升总R²

### 优化4: 特征工程增强 ★★★
**做什么**: 加入更多物理特征 (电子轨道能量、功函数、原子体积)
**预期**: 可能提升, 但边际递减 (当前110维已较全)
**成本**: 中

### 优化5: 接受当前精度, 强化其他维度 ★★★★★ (推荐)
**做什么**: 不追R², 把精力放在:
  - 完整论文写作 (中文SCI三区)
  - conformal UQ + AD + LOEO + SR共识 (已有)
  - 诚实叙事 (物理层=可解释, 非提精度)
**理由**: R² 0.910对三区够用, SOTA(0.928)差距0.018,
  即使优化1+2做到0.920也仍低于SOTA, 但三区不靠R²竞争

## 四、诚实结论

**问: LOEO结论可信吗?**
答: **可信**。经代码审计+修复实验+Wilcoxon验证, "物理层不提精度"是真实物理现象。

**问: 能优化让效果更好吗?**
答: **能, 但有限**。优化1+2(stacking+Optuna)可把R²从0.910提到0.920-0.925,
但仍低于matbench SOTA(0.928+)。优化空间存在但天花板由数据(wolverton 4914)决定。

**问: 最该做什么?**
答: **优化2(Optuna超参搜索)** 是ROI最高的单点优化 (文献支撑, 直接提R²)。
做完后若到0.92+, 配合已有创新(conformal+AD+SR共识+LOEO+适用判据),
三区稳录。若不到, 也没关系——三区不靠R²竞争, 靠方法学完整+诚实。

## 五、基准对比 (诚实定位)

| 模型 | 数据 | MAE (eV/atom) |
|---|---|---|
| matbench SOTA (ALIGNN/CGCNN) | matbench_perovskites | 0.02-0.04 |
| 我们 (PACT v2) | wolverton_oxides | 0.19 |
**注**: 不同数据集, 不直接可比。wolverton的形成能范围/参考态与matbench不同
(我们审计已证Pearson r=0.358)。但在各自数据上, 我们R²=0.910是合理的。

来源: [Matbench Discovery leaderboard](https://matbench-discovery.materialsproject.org/)
