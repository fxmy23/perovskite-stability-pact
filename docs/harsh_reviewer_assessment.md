# 严苛 SCI 三区审稿人评审意见（对标 2024–2025 模范论文）

**评审人立场**：以 *Computational Materials Science* (Elsevier, IF≈3.3, 中科院三区)、*npj Computational Materials* (二区顶)、*Materials* (MDPI, 三区) 三本期刊的复合审稿标准执行。**不偏袒作者，不为创新而创新。**

**评审对象**：PACT (Physics-Anchored Calibrated Trust) 框架 + PCRL (Physics-Constrained Residual Learning) + CGCNN 对比 + LOEO + 不确定性 + 候选筛选，针对 ABO₃ 钙钛矿氧化物形成能与热力学稳定性预测。

**评审日期**：2026-06-16

---

## 〇、一句话结论（先给作者）

> **以当前状态投稿 *Computational Materials Science*（三区），有较大可能被拒或大修（major revision）。核心问题不在"工作量不够"，而在"创新点未被严格证明成立 + R² 指标低于 2025 年 SOTA + 框架的若干组件（PCRL v2、Bayesian 融合、PICP）反而劣于简化基线"。若按本评审意见修订（约 4–6 周集中工作），则可达到三区录用线。**

我不是在顺着作者。下面是逐项证据。

---

## 一、与 2024–2025 模范论文的方法框架对标

### 1.1 基准论文清单（用作标尺）

| 编号 | 论文 | 期刊/分区 | 方法核心 | 形成能 R² |
|---|---|---|---|---|
| A | Prediction of ABX3 Perovskite Formation Energy Using ML (MDPI Materials 2025) | 三区 | 树模型 + 元启发调参 | **0.928** |
| B | Accelerating Multi-Property Prediction of Perovskites Via Meta-Heuristic Tuned ML (AI Chemistry 2025/26) | 二/三区 | 贝叶斯/进化算法调参 | **>0.9653** |
| C | Comparative Analysis of Conventional ML and GNN for Perovskites (J. Phys. Chem. C 2024) | 二区 | SVR/树模型/CGCNN 对比；SVR 测试集 | **~0.99 (RMSE 0.096 eV/atom)** |
| D | Interpretable ML-Assisted Screening of Perovskite Oxides (RSC Adv. 2024) | 三区 | 分类+回归+SHAP 筛选 | 报告 MAE/RMSE |
| E | SHAP-based Interpretable ML for Oxide Double Perovskite Band Gap (Phys. Scr. 2025) | 三区 | 多模型 SHAP 全局可解释 | — |
| F | From Formability to Bandgap: ML Accelerates Perovskite Discovery (ACS Nano 2025) | 一区 | 综述+工作流 | — |
| G | Quantum SVM for ABO₃ Structure Prediction (Comp. Mater. Sci. 2025) | 三区 | 量子 ML 分类 | — |

### 1.2 我们的数据（实测）

- 形成能 5-seed：R² = **0.911 ± 0.004**（pure_ml = LightGBM 集成）
- Hull 能 5-seed：R² = **0.794 ± 0.005**
- Stacking（嵌套 CV）：R² = 0.912
- CGCNN：R² = 0.799 ± 0.005
- 分类：F1 = 0.554, AUC = 0.926, DAF_top10% = 6.13
- 不确定性 σ-error 相关 = 0.378, PICP = 0.24
- PACT 可信域 R² = 0.954 vs 不可信域 0.872
- LOEO：Ho R²=0.96, Pr 0.95（好）；Al 0.246, Pb 0.224, Ac 0.33（差）

### 1.3 框架对比结论

| 维度 | 模范论文普遍做法 | 我们 | 评判 |
|---|---|---|---|
| 数据集规模 | 1k–19k，多数用 MP/AFLOW/OQMD 公开库 | 4914（wolverton_oxides），matminer 内置 | **数据集来源偏小且偏单一**（仅一个组的高通量），审稿人会质疑数据代表性 |
| 特征工程 | Magpie / Coulomb矩阵 / 结构描述符 / GNN 端到端 | 113维 = 96 Magpie + 17 物理 | **合格但无新意**；纯 pymatgen 重写 Magpie 是工程而非创新 |
| 模型 | 单模型或两两对比；2025 趋势是 **超参自动搜索**（贝叶斯/进化） | 5 模型集成 + Stacking + CGCNN | 工作量足；但**没有体现"为什么这些模型而非其它"** |
| 超参搜索 | 网格/贝叶斯/Optuna，**报告完整搜索空间** | 默认或浅层网格 | **缺失**：未报告超参搜索协议，审稿人会扣分 |
| 不确定性 | 偶有；多为 ensemble std 或 BNN | ensemble σ + PICP/MPIW + σ-error corr | **这是我们的差异化点，但 PICP=0.24 暴露校准失败** |
| 可解释性 | SHAP 已是 2024–2025 标配 | SHAP + PCRL SHAP-guided | **标配**，不构成强创新 |
| 外推测试 | 极少数论文做 LOEO/CV-by-element | LOEO 全 73 个元素 | **这反而领先模范论文**——是真正的差异化卖点 |
| 应用域 (AD) | 多为化学/毒性领域；材料领域罕见 | σ 中位数阈值 + 可信/不可信分区 | **创新点，但需更严格定义（见 §3）** |
| 候选材料 | 多数论文报告筛选候选 | 9 个候选 | **合格**；但缺 DFT 验证（见 §4） |

---

## 二、与模范论文的算法创新点对比（最关键）

**作者声称的创新**：PACT（物理锚定的不确定性 + 应用域）+ PCRL（SHAP 引导的物理约束残差学习）。

### 2.1 PCRL 的硬伤（必须正面承认）

**实测结果（pcrl_v2_comparison.csv）**：

| 方法 | 形成能 R² | Hull R² |
|---|---|---|
| pure_ml (LightGBM) | **0.9101** | **0.7998** |
| standard_pgml | 0.9074 | 0.7872 |
| pcrl_v1 (p=0.1) | 0.9060 | 0.7880 |
| **pcrl_v2_shap** | **0.9039** ↓ | **0.7768** ↓ |

**审稿人结论**：PCRL v2（我们宣称的"创新"）**在两个目标上均劣于 pure_ml**。SHAP-guided 惩罚没有带来提升，反而引入了性能下降。这是**论文致命伤**——审稿人一眼就会问："你为什么花一整章讲一个让性能变差的算法？"

**诚实分析**：
1. LightGBM 集成已经隐式吸收了物理特征的作用（86.6% 物理贡献），PCRL 的显式残差约束是冗余的。
2. SHAP-guided 惩罚会**降低**模型在物理相关特征上的拟合自由度，反而压制了有用的非线性。
3. 这与文献中 PGML 在弹性模量等任务上的成功（Mannodi 2020, npj）不一致——原因是**形成能本身高度物理可解释**（容差因子、电负性），不存在需要被"约束"的强非线性残差。

**必须做的事**（否则论文无法投出）：
- **要么放弃 PCRL 作为主创新**，改为"PACT 框架 + 标准 PGML 基线对照"，诚实报告 PCRL 失败并分析原因（这本身是有价值的负面结果，材料领域需要更多这种诚实分析）；
- **要么换一个 PCRL 真正有用的目标**——例如对 hull 能或分类任务（物理贡献较低、非线性更强）单独评估，找到 PCRL 占优的场景，把"创新"收窄到那个 niche；
- **绝不能**在论文里只报 PCRL 的"理论框架"而不报它劣于基线的事实。这是学术不诚。

### 2.2 PACT 的真正价值与缺陷

**PACT 是项目里最值得发表的组件**，但需要重新定位：

✅ **强项（超过多数三区模范论文）**：
- 统一产出 prediction + uncertainty + applicability domain + classification，**单一来源**（pact.py 一个管线），概念上自洽；
- 可信/不可信域 R² 差异（0.954 vs 0.872）是**真实且显著**的应用域信号；
- σ-error 相关 = 0.378 是**有意义的不确定性排序**（>0.3 在材料 ML 里算可用，文献基准约 0.2–0.4）。

❌ **致命缺陷**：
1. **PICP = 0.24**——这是**灾难性的**。PICP 应 ≥ 标称置信水平（如 0.9）。0.24 意味着我们的不确定性区间**严重过窄**， ensemble std 是过度乐观的（因为 5 个 LightGBM 高度相关，std 低估了真实误差）。**任何审稿人看到 PICP=0.24 都会拒稿**。必须改用 **conformal prediction**（保形预测，分布无关，可保证覆盖率）或深度 ensemble + adversarial perturbation，把 PICP 拉到 ≥ 0.8。
2. **应用域阈值（σ 中位数）是任意启发式**，没有理论保证。模范论文（如 RSC Adv. 2024）会用 leverage、distance-to-model、Isomoon-KNN 等更标准的 AD 方法做对照。
3. **加性融合 μ = μ_p + μ_r 缺乏理论推导**。Bayesian 融合"因 σ 尺度不兼容而失败→改加法"是一个工程妥协，不是方法创新。审稿人会问："为什么不用对数同方差化 + Bayesian？"

### 2.3 CGCNN 定位问题

- CGCNN R² = 0.799，**低于我们的 LightGBM (0.911)**。这本身合理（描述符 ML 在小数据上常胜 GNN）。
- 但论文叙事会是问题：如果 CGCNN 是"对比方法"，作者需要解释**为什么不端到端用 CGCNN**。模范论文 C (JPCC 2024) 的答案是"结构特征在小数据上更稳定"——我们应照此叙事。
- **CGCNN 早停监控训练损失而非验证损失**——这是实现 bug，审稿人会怀疑过拟合，**必须在投稿前修复**。

---

## 三、表征指标体系对比

### 3.1 模范论文的标准指标清单（合并去重）

| 类别 | 指标 | 我们是否报告 |
|---|---|---|
| 回归 | R², RMSE, MAE | ✅ |
| 回归 | MRE (平均相对误差%) | ❌ **缺失** |
| 回归 | Max error / 95-percentile error | ❌ **缺失** |
| 分类 | Accuracy, Precision, Recall, F1 | ✅ |
| 分类 | AUC-ROC, AUC-PR | ✅ ROC / ❌ PR |
| 排序 | DAF (Discovery Acceleration Factor) | ✅ |
| 排序 | Enrichment factor (EF) | ❌ |
| 不确定性 | PICP, MPIW | ✅（但 PICP 失败） |
| 不确定性 | σ-error Spearman/Pearson | ✅ |
| 不确定性 | Calibration plot / reliability diagram | ❌ **缺失** |
| 不确定性 | NLL (negative log-likelihood) | ❌ |
| 外推 | LOEO per-element R² | ✅ |
| 外推 | Scaffold/group CV | 部分（仅 LOEO） |
| 统计 | 5-seed mean ± std | ✅ |
| 统计 | Wilcoxon/Diebold-Mariano 显著性检验 | ❌ **缺失** |
| 统计 | Bootstrap 置信区间 | ❌ **缺失** |
| 复现 | 训练/验证/测试切分协议 | ✅ |
| 复现 | 随机种子列表 | ✅ |
| 复现 | 超参搜索协议与空间 | ❌ **缺失** |

### 3.2 审稿人评判

- **基础指标齐全**，合格。
- **统计严谨性中等**：5-seed 够用，但**缺显著性检验**——审稿人会要求"LightGBM vs XGBoost 的差异是否显著"，必须补 Wilcoxon signed-rank。
- **不确定性校准报告不完整**：PICP 失败 + 无 reliability diagram，会被直接挑刺。
- **超参协议缺失**是硬伤——任何严肃 ML 论文都要写清楚搜索空间和搜索次数。

---

## 四、实用性 / 影响力要素对比

### 4.1 模范论文的"实用要素"清单

| 要素 | 模范论文 | 我们 | 评判 |
|---|---|---|---|
| 候选材料列表 | ✅ 多数 | 9 个候选 | 合格 |
| **DFT 验证候选** | ✅ **多数模范论文有** | ❌ **完全没有** | **致命缺口** |
| 实验验证 | 少数顶级论文 | ❌ | 三区不强求，但 DFT 验证几乎必备 |
| 开源代码 + 数据 | 2024 起逐步标配（npj 强制） | ✅ 全套 src/ | **强项** |
| 可复现 README | ✅ | 部分 | 需补 |
| 用户预测工具 | 极少数 | predict.py | **加分项** |
| 与现有数据库对比 (MP/OQMD) | ✅ | ❌ 仅 wolverton | **弱项** |
| 与文献实验值对比 | ✅ | ❌ | 弱项 |
| 物理/化学机制解释 | ✅ SHAP+机理 | ✅ SHAP | 合格 |
| 设计规则提炼 | ✅ 多数 | 部分（AD + LOEO） | 中等 |

### 4.2 审稿人评判（实用性）

**最大的实用性短板：9 个候选稳定钙钛矿没有任何形式的独立验证。**

模范论文（如 RSC Adv. 2024、MDPI Materials 2025）几乎都会：
1. 用 VASP/Quantum ESPRESSO 对 1–3 个候选做 DFT 单点能计算，确认 convex hull 距离；
2. 或与 Materials Project 现有条目交叉核对；
3. 或引用已有实验合成文献佐证。

我们只有 ML 预测，**这在三区会被审稿人写进 major revision 意见**。即使是本科生单机资源，也可以：
- 用 **matminer** 检索候选是否已在 Materials Project 中存在（已知稳定性）；
- 用 **pymatgen** 的 PhaseDiagram 对候选做简单的凸包分析（如果有兄弟化合物数据）；
- 至少做**留出交叉验证**：把候选对应的元素组合在训练集中去掉，看模型是否仍预测稳定（这是廉价但有力的"外推可信度"证据）。

---

## 五、创新性诚实评判（作者最关心的）

**作者宣称的创新点 → 审稿人会如何看待**：

| 宣称创新 | 真实创新强度 | 理由 |
|---|---|---|
| 1. PACT 统一框架（prediction+UQ+AD+classification 单源） | **中等** | 概念整合，但每个组件都是已有技术；创新在"整合方式"，需更强论证 |
| 2. 物理贡献量化（86.6%） | **低** | 本质是 baseline_R² / total_R²，已有文献（Mannodi 2020）做过类似分解 |
| 3. PCRL SHAP-guided 残差约束 | **负面** | 实测劣于基线，**不能作为创新** |
| 4. LOEO 外推评估 | **强** | 73 元素全外推测试在钙钛矿 ML 论文中**罕见**，是真正的差异化卖点 |
| 5. 应用域 σ 阈值 | **中低** | 启发式，无理论保证；需要对照 leverage/KNN |
| 6. 9 个候选材料 | **中** | 缺验证，可信度受限 |
| 7. CGCNN 对照 | **低** | 复现现有工作，非创新 |
| 8. Stacking 嵌套 CV | **低** | 工程优化，非方法创新 |

**净创新评分（满分 10，三区及格线约 5）**：

- 当前状态：**4.5 / 10**（PCRL 拖累 + PICP 失败 + 候选无验证）
- 修订后（去 PCRL、修 PICP、补 DFT 验证、强化 LOEO 叙事）：**6.0–6.5 / 10**，可冲三区
- 顶刊线（npj/Q3 一区）：8+，我们 **达不到**，不要硬投。

---

## 六、对标"作者自己定的工作底线"

作者在对话中明确要求：
1. **科学严谨、可复现、诚实** → ✅ 数据 leakage 全修，5-seed 报告，troubleshooting_log 完整。**但诚实性在"PCRL 是否如实报告"上有风险**——必须报全表。
2. **真正的创新，不是模型替换** → ⚠️ **PCRL 恰恰是"模型替换 + 性能下降"**；真正创新是 LOEO + PACT 应用域，但叙事被 PCRL 稀释。
3. **统一连贯框架，不"想到什么加什么"** → ✅ PACT 单源架构达标；❌ 但 CGCNN/Stacking/PCRL 三条线显得分散，**论文必须明确以 PACT 为主线，其余降为对照**。
4. **发挥 RTX 4060 深度** → ⚠️ CGCNN 是唯一 GPU 任务，且早停有 bug。建议补一个**真正吃 GPU 的创新**：如 GNN + 描述符融合（descriptor-GNN hybrid），或 deep ensemble。
5. **每次改动完整记录** → ✅ troubleshooting_log.md 达标。

---

## 七、修订路线图（按优先级，4–6 周内可达三区投稿线）

### P0（投稿前必须完成，否则拒稿）

1. **修 PICP**：用 conformal prediction（split-conformal 即可，10 行代码）替换 ensemble std 区间，把 PICP 拉到 ≥ 0.85。**这是单点 ROI 最高的修订。**
2. **PCRL 决策**：三选一——
   - (a) 从主创新降级为"对照 + 负面结果分析"（最诚实）；
   - (b) 找 PCRL 占优的子任务（如分类、低物理贡献目标）重新定位；
   - (c) 删除 PCRL 章节。
   **推荐 (a)**：负面结果在材料 ML 里稀缺且有价值。
3. **CGCNN 早停 bug**：监控验证 loss 而非训练 loss。
4. **候选材料验证**：至少用 matminer/MP API + pymatgen 相图对 9 个候选做"已知/未知/与 MP 一致性"核对；若候选在 MP 中已有且稳定，是强佐证。

### P1（强烈建议，显著提升录用率）

5. **补超参搜索协议**：Optuna 50 次试验，报告搜索空间 + 最佳超参表。
6. **补显著性检验**：对 5-seed 结果做 Wilcoxon signed-rank，报 p 值。
7. **补 reliability diagram**：画校准曲线，配期望校准误差 ECE。
8. **补 AUC-PR + enrichment factor**（分类指标，三区常查）。
9. **应用域方法对照**：σ 阈值 vs leverage vs KNN-distance，三选一对比，证明 σ 阈值非任意。
10. **MRE + max error**：补全回归指标。

### P2（锦上添花）

11. 提炼 1–2 条"设计规则"（如"容差因子 t∈[0.85,0.95] + B 位 d 电子数≤4 → 稳定"），让论文有化学指导价值。
12. CGCNN + 描述符融合（hybrid），作为 GPU 深度的真正体现。
13. 与至少 1 个公开 MP 子集交叉验证，证明模型外推到非 wolverton 数据。

---

## 八、最终判断（作者最想听的真话）

**问：现在能投三区吗？**
答：**不能直接投，会被拒或大修。**

**问：修完 P0+P1 后能投三区吗？**
答：**能，且大概率录用**。理由：
- LOEO 全元素外推 + PACT 应用域是真正的差异化，三区模范论文里**几乎没有**；
- 数据 leakage 全修 + 5-seed + 嵌套 CV 的统计严谨性**超过多数三区论文**；
- 完整开源代码 + troubleshooting log **加分明显**。

**问：能冲二区吗？**
答：**有希望但不稳**。要冲二区（如 *npj Computational Materials* 的 lower tier 或 *Chem. Mater.* letter），还需要：
- DFT 验证至少 3 个候选（必须）；
- Conformal prediction + 校准图（必须）；
- 一个真正的方法创新（descriptor-GNN hybrid 或 calibrated deep ensemble）；
- 与 MP 公开数据交叉验证。

**问：我之前是不是在自我安慰？**
答：**部分是。** PCRL 这条线确实是"为了创新而创新"，且实验数据明确告诉我们它失败。继续把它包装成主创新会损害学术诚信。但 **PACT + LOEO 不是自我安慰**，它们是真东西，只是被 PCRL 和 PICP 失败遮蔽了。把精力从"包装 PCRL"转移到"强化 PACT/LOEO + 修 PICP + 补 DFT 验证"，论文会从"勉强能投"变成"三区稳录、二区有戏"。

---

## 九、参考模范论文（用于写作对标）

- [Prediction of ABX3 Perovskite Formation Energy Using ML (Materials 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/) — 三区写作范式，R²=0.928
- [Accelerating Multi-Property Prediction of Perovskites (AI Chemistry 2025)](https://www.sciencedirect.com/science/article/pii/S2949747726000187) — 超参自动搜索范式，R²>0.9653
- [Comparative Analysis of Conventional ML and GNN (JPCC 2024)](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03212) — 描述符 vs GNN 对照范式
- [Interpretable ML-Assisted Screening of Perovskite Oxides (RSC Adv. 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10813820/) — SHAP 筛选范式，最近邻竞品
- [SHAP-based Interpretable ML for Oxide Double Perovskite (Phys. Scr. 2025)](https://iopscience.iop.org/article/10.1088/1402-4896/add4ae) — 多模型 SHAP 写作范式
- [From Formability to Bandgap (ACS Nano 2025)](https://pubs.acs.org/doi/10.1021/acsnano.5c07494) — 综述，用于定位创新坐标
- [Quantum SVM for ABO₃ (Comp. Mater. Sci. 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0927025625000370) — 目标期刊 Comp. Mater. Sci. 的录用范式
- [UQ in Multivariable Regression for Materials (Sci. Rep. 2024)](https://www.nature.com/articles/s41598-024-61189-x) — 不确定性报告范式
- [Materials Property Prediction with UQ (APL Mater. 2024)](https://pubs.aip.org/aip/apr/article/10/2/021409/2892540/) — UQ 方法学背景
- [ML for Perovskite Materials Design (npj Comput. Mater. 2021)](https://www.nature.com/articles/s41524-021-00495-8) — 领域综述基准

---

**审稿人签名**：ZCode（严苛模式）
**建议**：major revision，2 个月再审。
