# 文献综述与方法论指导 (Literature Synthesis)

> 基于 2024-2025 高影响因子文献的深度调研, 提炼可操作的方法论指导。
> 本文档是后续所有改进工作的智识基础。

---

## 一、顶刊钙钛矿 ML 的完整方法论标杆

### 1.1 数据层的标杆做法

**Nature Communications 2024 (热力学稳定性预测)** 的做法:
- 用**电子构型**作为核心特征(不是简单 Magpie), 捕捉化学键本质
- **Stack generalization( stacking 集成)** 提升泛化, 而非单一模型
- 明确的数据清洗 + 去重 + 跨数据集验证

**对我们的启示**:
- ✅ 我们已做: 组成特征 + LOEO 外推 + 跨数据集
- ❌ 我们缺: 电子构型特征(价电子构型是稳定性的物理本质)
- ❌ 我们缺: stacking 集成(目前只有单模型)

### 1.2 特征层的标杆做法

**RSC 2024 (可解释ML筛选钙钛矿氧化物)** + **Comp Mater Sci 2024 (八面体畸变特征)**:
- **氧八面体畸变模式**作为跨尺度结构特征(连接组成与性质)
- Fourier 变换特征 + 2D CNN
- 容忍因子 + 八面体因子 + **新定义的复合判据**

**对我们的启示**:
- ✅ 我们有容忍因子/八面体因子
- ❌ 缺: 氧八面体畸变特征(但我们用纯组成, 无法算——这恰好是 GNN 能补的)
- 💡 可加: 自定义复合判据(如 τ = (r_A/r_B)·(χ_A/χ_B))

### 1.3 模型层的标杆做法

**Nature Comm 2024**: Stack generalization (多模型 stacking)
**npj 2024 (混合Transformer+GNN+迁移学习)**: 组成GNN + 结构GNN + 迁移学习
**MDPI 2025 (掺杂钙钛矿迁移学习)**: 小数据集用预训练GNN迁移

**对我们的启示**:
- 当前我们只有单模型(LightGBM), 应考虑 stacking 集成
- GNN + 迁移学习是小数据集(5000条)的正确策略

---

## 二、物理引导 ML 的前沿(超越简单残差)

### 2.1 领域演进趋势 (2024-2025)

```
第一代 (2020前): 残差建模 y = f_phys + g_ML  ← 我们的PCRL在这层
第二代 (2022-24): 物理约束损失 L = L_data + λ·L_physics
第三代 (2024-25): 架构嵌入物理 (PE-PINN, equivariant GNN)
```

**关键发现**: PNAS 2024([等变网络原理](https://www.pnas.org/doi/10.1073/pnas.2415656122))指出,
最高水平的物理引导是把对称性**嵌入网络架构**, 而非加损失项。

### 2.2 对我们 PCRL 的启示

**我们的 PCRL 目前在"第一代"** (残差 + feature_penalty)。
要升级, 有两条路:

**路径A (务实, 不需GPU)**: 升级到"第二代"
- 把 feature_penalty 替换为**显式 SHAP 正则损失项**
- 数学形式: L = ||y-f-g||² + λ·SHAP_phys(g)² + γ·||w||²
- 这比启发式 penalty 有理论基础, 我们已部分做了(pcrl_shap.py)

**路径B (需GPU, 更出彩)**: 结合等变思想
- 用 CGCNN 学结构表示(天然嵌入空间对称性)
- 把 CGCNN 的 embedding 作为 PCRL 的额外输入
- **CGCNN-PCRL 融合**: 物理引导 + 深度学习的真正结合

**我的判断: 路径A 必做, 路径B 是冲二区的杀手锏。**

---

## 三、CGCNN/GNN 的最佳实践与陷阱

### 3.1 CGCNN 在钙钛矿上的表现 (MDPI 2024 对比研究)

**关键发现**:
- CGCNN 对单钙钛矿形成能 RMSE 提升 **>20%** (vs 传统ML)
- 双钙钛矿需要**混合训练集**才能学好
- **陷阱: 结构相似性导致的数据泄露** — 如果训练集和测试集有结构近重复,
  GNN 会"记住"结构而非学习规律

**对我们的启示**:
- wolverton 有晶体结构(matbench_perovskites 也有), 可以跑 CGCNN
- 但要注意: wolverton 是网格遍历, 化学空间平滑,
  CGCNN 在上面可能过拟合(训练好测试差)
- **正确做法**: 用 LOEO 评估 CGCNN 外推性, 与 PCRL 对比

### 3.2 小数据集 GNN 的陷阱 (多篇文献共识)

**ALIGNN 可复现性研究 (RSC 2024)** 发现:
- GNN 在小数据集(>5000)上**方差大**, 不同 seed 结果差异显著
- 过拟合风险高, 需要 early stopping + dropout
- 迁移学习(预训练+微调)是小数据集的关键策略

**对我们的启示**:
- 我们的 4914 条对 GNN 来说偏小
- **必须多 seed 报告**(≥5次, 报 mean±std)
- **应该用迁移学习**: 在 matbench_perovskites(18928条)预训练, wolverton 微调

### 3.3 推荐的 GNN 策略

基于文献, 最优策略:
1. **CGCNN**(轻量, 4060 友好)作为主 GNN
2. **迁移学习**: matbench 预训练 → wolverton 微调
3. **多 seed**(≥5次)报告置信区间
4. **LOEO 评估**外推性, 与 PCRL 公平对比

---

## 四、评估严谨性标杆

### 4.1 审稿人期待的评估标准 (多文献共识)

| 维度 | 标杆做法 | 我们当前 | 缺口 |
|------|----------|----------|------|
| **多次运行** | ≥5 seeds, 报 mean±std | 单 seed | ❌ 必须加 |
| **统计检验** | paired t-test 比较模型 | 无 | ❌ 应加 |
| **置信区间** | bootstrap CI | 无 | ❌ 应加 |
| **分类指标** | F1/Precision/Recall/AUC | ✅ 有 | ✓ |
| **发现指标** | DAF + PR曲线 | DAF有(但对比不公平) | ⚠️ 需修 |
| **外推测试** | LOEO/LOCO | ✅ LOEO有 | ✓ |
| **不确定性** | prediction interval | 有ensemble std | ✓ |

### 4.2 关键启示

**最大缺口: 单 seed 报告**。
所有顶刊都报 mean±std(至少5次运行)。我们的所有结果(R²、DAF等)都是单点估计,
审稿人会直接质疑"这个数字的方差多大? 稳定吗?"

**修复**: 所有核心实验重跑 ≥5 seeds, 报告置信区间。

---

## 五、对我们项目的具体指导 (可操作清单)

### 必须做的 (P0, 不做会被拒)

| # | 改进 | 文献依据 | 预计工作量 |
|---|------|----------|-----------|
| 1 | **多 seed (≥5) 报告 mean±std** | 所有顶刊标准 | 1天(重跑) |
| 2 | **修复架构矛盾**(基线统一+泄露闭环) | 自洽性要求 | 半天 |
| 3 | **DAF 诚实化**(撤回不公平对比) | Matbench Discovery 标杆 | 0.5天 |
| 4 | **PR 曲线 + ROC 曲线** | 分类评估标配 | 0.5天 |

### 应该做的 (P1, 提升深度)

| # | 改进 | 文献依据 |
|---|------|----------|
| 5 | **CGCNN + 迁移学习** | MDPI 2024 + npj 2024 |
| 6 | **PCRL 升级为显式 SHAP 正则** | PIML 第二代方法 |
| 7 | **Stacking 集成** | Nature Comm 2024 |
| 8 | **电子构型特征** | Nature Comm 2024 |
| 9 | **CGCNN-PCRL 融合实验** | 等变+物理引导前沿 |

### 可选的 (P2, 加分)

| # | 改进 | 文献依据 |
|---|------|----------|
| 10 | paired t-test 模型比较 | 统计严谨性 |
| 11 | bootstrap 置信区间 | 统计严谨性 |
| 12 | 自定义复合物理判据 | 特征工程创新 |

---

## 六、论文叙事重构建议 (基于文献)

### 当前叙事 (有断裂)
```
数据 → 特征 → LightGBM → PGML/PCRL → DAF(不公平) → 发现
```

### 文献指导下的理想叙事 (层层递进)
```
1. 问题定义: 钙钛矿稳定性预测的挑战
2. 数据: wolverton + matbench 双数据集
3. 方法 A: PCRL (组成+物理, 无GPU依赖, 通用)
   - 数学推导 + SHAP 约束 + 物理贡献量化
4. 方法 B: CGCNN (结构, 深度学习, 迁移学习)
   - 预训练+微调 + 多seed
5. 对比: PCRL vs CGCNN vs 融合 (公平, 多seed, 统计检验)
6. 评估: 回归(R²) + 分类(F1/DAF, 诚实) + 外推(LOEO) + 不确定性
7. 发现: 候选材料 + 反常材料 + 适用域
8. 讨论: 诚实局限 (Recall, 跨数据集, 组成 vs 结构)
```

这个叙事有**方法深度(PCRL数学+GNN)、评估严谨(多seed+分类+外推)、
物理洞见(SHAP+发现)、诚实局限**, 符合三区甚至二区标准。

---

## 参考来源 (核心文献)

### 方法论标杆
- [Nature Comm 2024: 电子构型+stacking 稳定性预测](https://www.nature.com/articles/s41467-024-55525-y)
- [npj 2021: 钙钛矿ML设计与发现综述](https://www.nature.com/articles/s41524-021-00495-8)
- [PMC: 钙钛矿ML特征选择综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC10146176/)

### 物理引导ML前沿
- [PNAS 2024: 等变网络原理](https://www.pnas.org/doi/10.1073/pnas.2415656122)
- [Springer 2025: 材料PINN综述](https://link.springer.com/article/10.1007/s11831-025-10448-9)
- [Adv Funct Mater 2025: OptiXNet等变GNN](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adfm.202523683)

### CGCNN/GNN实践
- [MDPI 2024: CGCNN vs 传统ML钙钛矿对比](https://www.mdpi.com/2304-6740/14/2/58)
- [RSC 2024: ALIGNN可复现性研究](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00064a)
- [npj 2024: 混合Transformer+GNN+迁移学习](https://www.nature.com/articles/s41524-024-01472-7)
- [MDPI 2025: 掺杂钙钛矿迁移学习](https://www.mdpi.com/2073-4352/15/12/1008)

### 评估严谨性
- [PMC: 材料ML不确定性预测](https://pmc.ncbi.nlm.nih.gov/articles/PMC8655759/)
- [Nature MI 2025: Matbench Discovery框架](https://www.nature.com/articles/s42256-025-01055-1)
- [SAGE 2025: 可靠ML预测的统计框架](https://journals.sagepub.com/doi/full/10.1177/18747655261420243)
