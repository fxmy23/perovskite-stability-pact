# 核心文献精读笔记

> 本文件汇总本项目的 5 篇核心文献,每篇给出**研究问题 / 方法 / 关键结果 / 与本课题关系**四要素。
> 这些文献是引言(Introduction)写作的主要素材,也是保研面试可能被问到的"你读过哪些文献"的标准答案。

---

## 文献 1:双钙钛矿带隙预测的开山之作(方法论奠基)

**Pilania, G. et al.** "Machine learning bandgaps of double perovskites."
*Scientific Reports* 6, 19375 (2016).
- 🔗 https://www.nature.com/articles/srep19375

### 研究问题
如何用机器学习从化学组成预测 A₂BB'O₆ 双钙钛矿的电子带隙?

### 方法
- 数据: 实验测定的 ~350 个双钙钛矿带隙
- 特征: 离子半径比、电负性、原子序数等 ~30 维组成描述符
- 模型: 高斯过程回归 (GPR) + 留一类交叉验证

### 关键结果
- 带隙预测 MAE ≈ 0.15 eV,与 DFT 误差量级相当
- 识别出 B-B' 位离子半径差是带隙主导因素

### 与本课题关系
- ★ 证明了**双钙钛矿 + 组成特征 + 经典 ML** 路线可行
- 我们的差异化: 用更大的 MP 数据集 + 现代 boosting 模型 + 预测形成能/稳定性(非带隙)
- 我们的 SHAP 分析可与其"特征重要性"结论对比, 体现方法进步

---

## 文献 2:单/双钙钛矿可形成性与稳定性预测(最直接的对照工作)

**Saad, Y. et al.** "A Machine Learning Approach for the Prediction of
Formability and Thermodynamic Stability of Perovskite Oxides."
*Chemistry of Materials* 33(16), 6451–6464 (2021).
- 🔗 https://pubs.acs.org/doi/10.1021/acs.chemmater.0c03402

### 研究问题
哪些原子特征决定钙钛矿(单+双)能否形成且热力学稳定?

### 方法
- 数据: ~4900 个 ABO₃ 与 A₂BB'O₆ 化合物 (DFT 计算的 E_hull)
- 特征: ~40 维手工组成描述符(电负性、半径、价电子数等统计量)
- 模型: SiR(Scalable Invariant Representation) + 随机森林分类/回归

### 关键结果
- 稳定性分类准确率 ~90%
- 识别出 A 位离子半径、B 位电负性是稳定性的关键描述符
- 高通量筛选出大量候选稳定双钙钛矿

### 与本课题关系
- ★★★ **本项目最直接的对照工作**。我们必须在引言中详述并差异化
- 我们的改进点:
  1. 用 Magpie 自动生成 132 维描述符(而非手工 40 维), 特征工程更系统
  2. 加入经典物理特征(容忍因子 t、八面体因子 μ), 增强可解释性
  3. 同时报告形成能 E_f 和 E_hull 双目标(Saad 仅做 E_hull)
  4. 用 SHAP 而非简单特征重要性, 提供样本级解释

---

## 文献 3:氧化物钙钛矿形成能与稳定性的可解释 ML(方法论标杆)

**Lu, S. et al.** "Predicting the formation and stability of oxide
perovskites by extracting underlying mechanisms using machine learning."
*Computational Materials Science* 211, 111531 (2022).
- 🔗 https://www.sciencedirect.com/science/article/abs/pii/S0927025622002622

### 研究问题
能否从 ML 模型中提取出决定钙钛矿稳定性的物理机制?

### 方法
- 数据: ~5000 个氧化物钙钛矿
- 特征: 大量组成描述符
- 模型: 随机森林 + **符号回归** (symbolic regression) 提取解析判据
- 可解释性: 从模型中"蒸馏"出类似容忍因子的解析公式

### 关键结果
- 成功用符号回归还原了一个与容忍因子高度相关的稳定性判据
- 证明 ML 模型并非黑盒, 内部确实学到了物理规律

### 与本课题关系
- ★★ **本项目"可解释性"叙事的方法论标杆**
- 我们采用 SHAP(基于博弈论的特征归因)而非符号回归, 是另一种可解释路径
- 在讨论部分可以对比两种方法的优劣: SHAP 更通用但无解析式, 符号回归有解析式但表达力受限

---

## 文献 4:ABX₃ 钙钛矿形成能预测(最新对照工作,R²=0.928)

**Authors.** "Prediction of ABX₃ Perovskite Formation Energy Using Machine
Learning." *Materials* 18(13), 2927 (2025).
- 🔗 https://www.mdpi.com/1996-1944/18/13/2927
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/

### 研究问题
用现代 ML 算法预测 ABX₃ 钙钛矿形成能, 能达到多高精度?

### 方法
- 数据: ABX₃ 体系(含卤化物和氧化物)
- 特征: 组成描述符
- 模型: 多种 ML 算法对比, 最优 R² = 0.928
- 可解释性: 特征重要性分析

### 关键结果
- 最优模型 R² = 0.928 (论文标题卖点)
- 给出了形成能的关键描述符排序

### 与本课题关系
- ★★ **必须与之对比的最新同类工作**
- 这篇只做 ABX₃ 单钙钛矿的形成能, 我们扩展到 A₂BB'O₆ 双钙钛矿 + 稳定性
- 这是 2025 年新发表的, 说明该方向**仍然活跃, 仍有发表空间**(对我们是利好)
- 我们的论文可以强调:"本研究将 [文献4] 的方法体系扩展到双钙钛矿并补充稳定性目标"

---

## 文献 5:matminer 工具包(方法学引用)

**Ward, L. et al.** "Matminer: An open source toolkit for materials data mining."
*Computational Materials Science* 152, 60–69 (2018).
- 🔗 https://www.sciencedirect.com/science/article/am/pii/S0927025618303252

### 研究问题
如何系统化材料数据的特征生成与 ML 流程?

### 方法
- 开源 Python 库, 集成 Magpie 描述符 + 多种结构/组成 featurizer
- 与 scikit-learn / pandas 无缝衔接

### 关键结果
- 提供了 50+ 种 featurizer, 覆盖组成、结构、电子性质
- 已成为材料信息学事实标准

### 与本课题关系
- 我们的特征工程方法学引用(论文方法部分必须引用)
- Magpie 预设生成的 132 维描述符即来自此工作

---

## 文献综述脉络(用于引言写作)

引言应按以下逻辑组织:

1. **钙钛矿的重要性**(催化、燃料电池、阻变存储的应用价值)
2. **传统筛选方法的局限**(DFT 高通量计算成本高; 实验筛选周期长)
3. **材料信息学的兴起**(文献 1, 5: ML 用于性质预测的可行性)
4. **钙钛矿 ML 预测的现有工作**(文献 2, 3, 4: 单/双钙钛矿形成能稳定性已有研究)
5. **现有工作的不足 → 本研究的切入点**:
   - 文献 4 (2025) 的 ABX₃ 形成能预测未覆盖较大元素空间
   - 文献 2 (2021) 仅用手工 40 维特征 + 做 E_hull 一个目标
   - 文献 3 (2022) 的可解释性方法(符号回归)较复杂, 不易推广
6. **本研究的三点创新**(大规模元素覆盖 + 双目标 + SHAP 可解释性)

---

## 保研面试可能被问到的问题(基于这些文献)

- Q: 你读过哪几篇相关文献? 最喜欢哪篇? 为什么?
  - A: 答 [文献2], 因为它最直接对照。然后讲清楚自己的差异化。

- Q: 你的工作和文献 4 (2025 那篇 R²=0.928) 有什么本质区别?
  - A: 体系(双钙钛矿 vs 单)、目标(双目标 vs 单)、可解释性方法(SHAP vs 特征重要性)。

- Q: 为什么选 SHAP 而不是 LIME 或符号回归?
  - A: SHAP 有博弈论严格基础(Shapley 值), 全局+局部一致; LIME 不稳定; 符号回归表达力受限。

(完整面试备答见 docs/interview_prep.md)
