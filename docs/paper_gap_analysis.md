# 论文板块差距分析 (对照 2024-2025 模范论文)

**日期**: 2026-06-16
**对标论文**:
- [MDPI Materials 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/) (R²=0.928, 三区)
- [RSC Adv 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10813820/) (钙钛矿氧化物筛选, 最严谨模板)
- [npj Comput Mater 2021](https://www.nature.com/articles/s41524-021-00495-8) (综述, 引用骨架)
**目标期刊**: Computational Materials Science (Elsevier, IF≈3.3, 三区)

---

## 一、模范论文的标准章节结构 (3 篇对照)

```
1. Introduction (含工作流图)
2. Methods/Experimental
   2.1 Dataset (来源+规模+清洗+切分)
   2.2 Feature Engineering (Pearson去冗余+RFECV/SHAP筛选)
   2.3 Models (含MAE/RMSE/R²公式)
   2.4 Evaluation Protocol (k-fold CV + 指标)
3. Results & Discussion
   3.1 特征分析 (重要性+SHAP beeswarm+dependence)
   3.2 模型对比 (表: 含文献基准行)
   3.3 模型优化 (超参/消融)
   3.4 验证 (parity plot + 候选/DFT验证)
4. Conclusions
+ Acknowledgments / Author Contributions / Data Availability / Code Availability / Conflicts
```

## 二、逐板块差距分析 (我们 vs 模范)

### ✅ 我们已具备的板块 (强项, 超过模范)

| 板块 | 我们 | 模范 | 状态 |
|---|---|---|---|
| 不确定性量化 | conformal+CQR (ECE降44%) | ❌ 都没有 | **★领先** |
| 应用域 | σ/kNN/leverage三方法 | ❌ 都没有 | **★领先** |
| 外推评估 | 68元素LOEO | ❌ 都没有 | **★领先** |
| SR可解释方程 | 符号集成共识 | SHAP only | **★领先** |
| 统计严谨 | bootstrap CI+Wilcoxon+p值 | P2有t-test | 相当/略优 |
| 候选验证 | 三层(matbench/LOEO/Goldschmidt) | P1 FPC, P2 DFT | 相当 |
| 模型对比 | LGB/XGB/HistGBT/RF/SVR | LGB/XGB/RF等 | 相当 |

### ❌ 我们缺失/不足的板块 (需补全)

#### 缺失1: 工作流图 (Workflow Figure) ★★★★
**模范**: 两篇原创论文都在Introduction末尾放工作流图。
**我们**: 无。
**补全**: 画一张 PACT 框架工作流图 (输入→物理层→ML层→conformal→AD→输出)。
**工具**: matplotlib/PPT, 1-2小时。

#### 缺失2: Parity Plot (预测vs真实散点) ★★★★★
**模范**: P1有6面板parity (每模型一个), P2有散点+heatmap。
**我们**: 有数据(results/pact_v2_oof_*.csv), 但没画图。
**补全**: 用 oof_mu vs y_true 画 parity plot (含R²/MAE标注, y=x参考线)。
**这是审稿人第一眼看的图, 必须有。**

#### 缺失3: SHAP Dependence Plots (依赖图) ★★★★
**模范**: P2有top-8特征的SHAP dependence plot (最强可解释性元素)。
**我们**: 有SHAP重要性(shap_importance.csv), 但没dependence plot。
**补全**: 对top-6特征画 SHAP dependence (x=特征值, y=SHAP值, 显示非线性关系)。

#### 缺失4: 数据描述叙事 (Dataset narrative) ★★★
**模范**: P1 "4358→2703 after cleaning, 168→63→30→9 features"。
**我们**: 数据在代码里, 但没写成清晰叙事。
**补全**: 写"4914样本, 来源wolverton_oxides, 113维特征=96 Magpie+14物理+3电子,
  排除3衍生/死特征后110维, 5-fold CV"。

#### 缺失5: 模型对比表含文献基准行 ★★★★
**模范**: P2每张表都有"ref."行 (对比前人文献)。
**我们**: 表里只有自己的模型。
**补全**: 在模型对比表加"文献基准"行:
  - MDPI 2025 R²=0.928 (不同数据, 注明)
  - RSC 2024 (类似数据)
  - Emery&Wolverton 原始GBRT (同源数据)
**这强化"我们vs前人"的对比, 突出贡献。**

#### 缺失6: Confusion Matrix + ROC (分类) ★★★
**模范**: P2有混淆矩阵+ROC曲线。
**我们**: 有F1/AUC数值, 但没图。
**补全**: 对凸包能稳定分类画混淆矩阵+ROC。

#### 缺失7: 消融实验 (Ablation Study) ★★★★
**模范**: P2有(with/without stability label)。
**我们**: 有零散的对比(PCRL/物理层/stacking), 但没组织成"消融表"。
**补全**: 做一张消融表:
  - 去掉物理层 (纯ML)
  - 去掉conformal (用ensemble σ)
  - 去掉stacking (单LightGBM)
  - 去掉CQR (标准conformal)
  → 量化每个组件的贡献。

#### 缺失8: Data Availability Statement ★★★★★ (期刊强制)
**模范**: P1有, 期刊强制。
**我们**: 无。
**补全**: "Data: wolverton_oxides via matminer. Code: GitHub link. Features: data/processed/"。

#### 缺失9: Code Availability / GitHub ★★★★
**模范**: 都没有 (但我们有, 是加分)。
**我们**: 有完整src/, 但没整理README。
**补全**: 写README.md (安装/运行/复现步骤), 准备GitHub/Zenodo。

#### 缺失10: Limitations Section ★★★
**模范**: 都没有 (添加=差异化)。
**我们**: 散在docs/各文件, 没集中。
**补全**: 写一节"Limitations":
  - 数据单源 (wolverton only)
  - R²受数据噪声上限 (~0.93)
  - LOEO外推R²降到0.70
  - SR公式非唯一
  - 候选无DFT验证

#### 缺失11: 超参表 (主文或SI) ★★★
**模范**: P2在SI有超参表。
**我们**: Optuna结果在csv, 没整理成表。
**补全**: 把Optuna最佳超参整理成表 (主文或SI)。

## 三、补全优先级 (按ROI)

### P0 必做 (投稿前必须, 审稿人第一眼看)
1. **Parity plot** (预测vs真实) — ★★★★★
2. **工作流图** (PACT框架) — ★★★★
3. **Data Availability Statement** — ★★★★★ (期刊强制)
4. **模型对比表 + 文献基准行** — ★★★★

### P1 强烈推荐 (显著提升录用率)
5. **SHAP dependence plots** (top-6特征) — ★★★★
6. **消融实验表** (各组件贡献) — ★★★★
7. **Limitations section** — ★★★
8. **Code/README + GitHub** — ★★★★

### P2 锦上添花
9. **Confusion matrix + ROC** — ★★★
10. **超参表** (SI) — ★★★
11. **数据描述叙事** — ★★★
12. **元素分布图** (周期表heatmap) — ★★

## 四、我们独有的差异化卖点 (写论文要强调)

这些是模范论文**都没有**的, 是我们的真创新:
1. **CQR条件保形预测** (ECE降44%, ICML/ICLR 2025前沿)
2. **应用域三方法对照** (σ/kNN/leverage)
3. **68元素LOEO外推评估**
4. **SR符号集成共识** (a_site_en 100%)
5. **完整不确定性量化** (conformal+CQR+AD+σ-error)

**叙事策略**: 论文标题/摘要突出"uncertainty-aware + applicability-domain
  perovskite stability prediction with conditional conformal guarantees"——
  这是2篇模范论文都没做的维度, 是我们的护城河。

## 五、实施计划 (按顺序)

1. 先补 P0 (parity图+工作流图+数据声明+对比表) — 投稿门槛
2. 再补 P1 (SHAP dependence+消融+限制+代码) — 提录用率
3. P2 有时间再补
4. 全部基于已有数据 (results/metrics/), 只需可视化+组织, 无需新实验

## 六、文献来源
- [Computational Materials Science Guide for Authors](https://www.sciencedirect.com/journal/computational-materials-science/publish/guide-for-authors)
- [MDPI Materials 2025 R²=0.928](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/)
- [RSC Adv 2024 perovskite screening](https://pmc.ncbi.nlm.nih.gov/articles/PMC10813820/)
- [npj Comput Mater 2021 review](https://www.nature.com/articles/s41524-021-00495-8)
- [GitHub: chenebuah/perovskite-ML](https://github.com/chenebuah/perovskite-ML) (12模型对比基准)
