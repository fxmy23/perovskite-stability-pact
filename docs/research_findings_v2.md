# 深度调研记录与评判 V2（2026-06-16）

本文件记录为"完善项目框架"所做的针对性调研，**结论导向**——每条调研都要落到"我们做什么改动"上。

---

## 调研 1：Conformal Prediction（保形预测）—— 修 PICP 的正确方法

### 1.1 核心结论
我们当前 PICP = 0.24（灾难性）的根因：**用 5 个高度相关的 LightGBM 的 std 当不确定性，严重低估真实误差**。这不是调参能修的，是**方法论错误**。

### 1.2 调研得到的正确方法：Split Conformal Prediction

**算法（5 步，分布无关，覆盖率有理论保证）**：

| 步骤 | 操作 |
|---|---|
| 1. 切分 | 训练集再切出 calibration set（如训练集的 25%）|
| 2. 训练 | 只在剩余训练集上拟合模型 |
| 3. 非一致性分数 | calibration set 上算残差 s_i = \|y_i − ŷ_i\| |
| 4. 分位数 | 取 q = ⌈(1−α)(n+1)⌉ 位分位数（α=0.2 → 80% 覆盖）|
| 5. 预测区间 | 新样本 ŷ ± q |

**理论保证**：在数据可交换（exchangeable）假设下，**真实 PICP ≥ 1−α**（有限样本严格成立，不是渐近）。这正是我们缺的"理论保证"。

### 1.3 适配 CV 的高级版：CV+ Conformal / Jackknife+
我们的场景是 5 折 CV 评估，单点切 calibration 会浪费数据。**正确做法**：
- 每折训练集内部再切 calibration 子集 → 得到该折的 conformal score；
- 或用 **Jackknife+**（每个样本作为 calibration 一次，更稳）。
- 工程上**最简单稳健**：**在每折内，把训练集再切 80%/20%，20% 做 calibration**，算该折分位数 q_fold；OOF 区间 = ŷ_oof ± q_fold。最后报告的 PICP 是 OOF 区间对真实值的覆盖率——**这个覆盖率会 ≈ 1−α**。

### 1.4 进一步：Conformalized Quantile Regression (CQR)
若想区间**异方差**（高不确定性样本区间更宽），用分位数回归 + conformal 校准。我们的 LightGBM 支持 `objective='quantile'`，可原生实现。**但 split conformal 已足够修 PICP**，CQR 作为升级项。

### 1.5 评判与决策
**决策：在 pact.py 中实现 split-conformal（CV 内部 calibration），把 PICP 从 0.24 拉到 ≥ 0.80。这是单点 ROI 最高的修订。** 同时保留 ensemble σ 作为"排序不确定性"（σ-error 相关 0.378 仍有意义），但区间宽度改用 conformal 分位数——两者分工：σ 排序，conformal 定宽。

参考文献：
- [UC Berkeley Conformal Prediction lecture](https://www.stat.berkeley.edu/~ryantibs/statlearn-s23/lectures/conformal.pdf)
- [MAPIE 文档（split/cross conformal）](https://mapie.readthedocs.io/en/latest/split_cross_conformal.html)
- [Wikipedia: Conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction)

---

## 调研 2：应用域（AD）标准方法对照

### 2.1 核心结论
我们当前用"σ < 中位数"做 AD 阈值，是**启发式**，无理论依据，审稿人会质疑。需要补**至少 2 种标准方法对照**。

### 2.2 调研得到的标准 AD 方法

| 方法 | 原理 | 复杂度 | 我们适用性 |
|---|---|---|---|
| **Leverage（Williams plot）** | h_i = x_iᵀ(XᵀX)⁻¹x_i，阈值 h* = 3p/n | O(p²)，需线性可逆 | ⚠️ 我们 p=113 > 折内 n 时不稳，需 PCA 降维或正则 |
| **k-NN distance** | 到训练集 k 近邻的平均距离，阈值 = 训练距离分布的 95 分位 | O(n·log n) | ✅ 通用，对树模型友好 |
| **Isolation Forest** | 异常检测，孤立深度反演 | O(n) | ✅ 适合高维 |
| **Std/ensemble（我们的）** | 集成方差 | O(1)（已有） | ✅ 已有 |

### 2.3 评判与决策
**决策：实现 3 种 AD 方法对照（k-NN distance + Leverage on PCA + ensemble σ），报告每种方法的"可信区 R² vs 不可信区 R²"差异，证明 σ 阈值不是任意选择。** 预期结论：σ 与 k-NN 高度相关（都反映外推度），互相佐证。这把"启发式"升级为"多方法一致性证据"。

参考文献：
- [Interpretable ML-Driven QSAR (PMC 2024, leverage + Williams plot)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12827418/)
- [Chemoinformatic Regression Methods and AD (Mol. Inform. 2024)](https://onlinelibrary.wiley.com/doi/10.1002/minf.202400018)
- [Applicability Domain — Wikipedia](https://en.wikipedia.org/wiki/Applicability_domain)

---

## 调研 3：候选材料验证工作流

### 3.1 核心结论
我们 9 个候选**零验证**，是实用性最大短板。本科生单机无法做 DFT，但可用**数据库交叉核对**作为强佐证。

### 3.2 可行的验证工作流（按强度递增）

**Tier 1（必做，零成本）：matminer 内置数据集核对**
- wolverton_oxides 本身就是 DFT 数据，候选若与训练集 A-B 组合重叠，可用训练集中同类化合物的 E_hull 作"同族类比"；
- 用 matminer 的 `load_dataset` 检查候选元素组合是否在其它数据集（matbench_perovskites）中已知稳定。

**Tier 2（强烈推荐，免费）：Materials Project API 交叉核对**
- 用 `mp_api.client.MPRester`（需免费 API key）查询候选 `formula_pretty` 是否在 MP 中存在；
- 若存在且 `energy_above_hull ≤ 0.05`，是**独立 DFT 验证**（强证据）；
- 若不存在，说明可能是未合成的新材料（也是发现价值）；
- 用户之前说 MP 网页打不开，但 **API 与网页是独立服务**，API 可能可用。需测试。

**Tier 3（廉价外推验证）：留出元素交叉验证**
- 对每个候选，从训练集中**移除该 A 或 B 元素的所有样本**，重训模型，看是否仍预测稳定；
- 这是"候选对训练集的依赖度"测试，无需外部数据，但能证明"非过拟合"。

**Tier 4（理想但贵）：单点 DFT**
- 用 VASP/Quantum ESPRESSO 算候选的形成能——本科生单机不现实，跳过。

### 3.3 评判与决策
**决策：实现 Tier 1 + Tier 2 + Tier 3 三层验证。** Tier 2 先测 API 是否可达；若 MP API 不可达，退化为 Tier 1 + Tier 3 + 文献检索（Google Scholar 查候选是否有合成报道）。验证脚本输出 `candidate_validation.csv`，每行候选附"已知/未知/外推稳定/外推不稳定"标签。

参考文献：
- [Materials Project API examples](https://docs.materialsproject.org/downloading-data/using-the-api/examples)
- [mp_api.client.MPRester 迁移说明](https://github.com/materialsproject/api/blob/main/mp_api/client/mprester.py)

---

## 调研 4：统计严谨性补充（Wilcoxon / ECE / AUC-PR / Enrichment Factor）

### 4.1 调研得到的必补项

| 缺失项 | 标准做法 | 我们怎么补 |
|---|---|---|
| **Wilcoxon signed-rank** | 对配对的 5-seed 结果做检验，报 p 值 | `scipy.stats.wilcoxon`，对比 LightGBM vs RF/XGBoost/SVR |
| **Reliability diagram + ECE** | 把预测分 10 桶（按 σ），每桶算实际覆盖率 vs 名义覆盖率，ECE = 加权平均偏差 | 画校准曲线 + 算 ECE（修 PICP 后必有）|
| **AUC-PR** | 不平衡数据比 ROC 更敏感 | `sklearn.metrics.average_precision_score` |
| **Enrichment Factor (EF)** | top-k 中真阳性比例 / 总阳性比例，DAF 的变体 | 与 DAF 并报 |
| **Bootstrap 95% CI** | 对 R²/MAE 做 1000 次 bootstrap，报 [lo, hi] | `np.random.choice` 重采样 |
| **MRE / Max error** | 平均相对误差、最差样本 | 补全回归指标 |

### 4.2 评判与决策
**决策：新建 `src/stats_eval.py`，实现全部上述指标。** 这是把"统计中等"升级为"统计严谨"的关键，且成本低（都是 sklearn/scipy 标准调用）。

---

## 调研 5：物理约束 ML 的真正 SOTA（决定 PCRL 命运）

### 5.1 核心结论
我们 PCRL v2 实测劣于基线，不是实现 bug，是**方法论不适合本任务**。文献中 PCRL 类方法成功的前提是"目标函数有强非线性且物理先验不完整"（如弹性模量、带隙）。**形成能高度物理可解释（容差因子、电负性），物理先验已近完整，残差约束是冗余的。**

### 5.2 2024–2025 真正的物理约束 ML SOTA 方向

| 方向 | 代表工作 | 与我们关系 |
|---|---|---|
| **Physics-informed neural network (PINN)** | 用 PDE/守恒律作硬约束损失 | 不适用（我们没有 PDE）|
| **Equivariant GNN（e3nn, MACE）** | 旋转/平移等变图网络 | 是 CGCNN 的升级，但需要多 GPU |
| **Multi-fidelity learning** | 高保真+低保真数据融合 | 我们只有单源 DFT |
| **Active learning + UQ** | 用不确定性驱动迭代采样 | 我们数据已固定，无法主动学习 |
| **Symbolic regression（PySR）** | 发现物理解析公式 | ✅ **可行且创新**——对物理层用符号回归发现 t、μ_oct 的新组合公式 |
| **Deep ensemble + adversarial** | 5 个独立初始化的网络 + 输入扰动 | ✅ 修不确定性校准的现代方法 |

### 5.3 评判与决策
**PCRL 决策（三选一，选 A）：**
- **(A) 诚实降级**：PCRL 从"主创新"改为"对照实验 + 负面结果分析"。论文明确写"我们发现对形成能这种高物理可解释目标，显式物理约束残差不带来提升，反而因压制有用非线性而劣化。这说明 PCRL 适用边界是物理先验不完整的任务"。**这是诚实且有科学价值的负面结果。**
- (B) 换目标：对 hull 能或分类单独评估 PCRL（但仍劣于基线，见实测）。
- (C) 删除：信息损失大。

**新创新方向（替代 PCRL 作为方法亮点）：**
- **方向 1（推荐，低成本高创新）：符号回归物理层**。用 PySR 或简化穷举，在 15 个物理特征上发现形成能的**解析公式**（如 E_f ≈ a·t² + b·μ_oct + c·χ_AB + ...），公式层替代 KernelRidge，ML 残差补足。这把"物理贡献 86.6%"从数字升级为**可写的物理方程**，创新性强且本科生可实现。
- **方向 2：Conformal 校准的 Deep Ensemble**——但需多训练，CPU 成本高。

**决策：PCRL 降级为对照（A）；新增"符号回归物理层"作为方法亮点（若时间允许）。** 若符号回归超时，退回到"PACT + LOEO + conformal"三件套作为创新主线，已足够三区。

---

## 综合评判：完善后的框架主线

**完善后的论文创新主线（单一连贯叙事）：**

> **PACT 框架**：物理锚定 + 保形校准的可信预测。
> 1. **物理层**：KernelRidge/符号回归 on 15 物理特征 → μ_p（物理贡献量化）
> 2. **ML 残差层**：LightGBM 集成 on 113 特征 → μ_r
> 3. **统一预测**：μ = μ_p + μ_r
> 4. **校准不确定性**：ensemble σ（排序）+ **split conformal 分位数**（定宽，PICP 有保证）
> 5. **应用域**：σ / k-NN / leverage 三方法一致性判定（不再是单启发式）
> 6. **外推评估**：LOEO 全 73 元素外推（差异化卖点）
> 7. **候选发现 + 三层验证**（matminer/MP/留出元素）

**PCRL** 降级为 §"对照实验：显式物理约束的适用边界"（负面结果）。

**CGCNN** 作为"描述符 vs 端到端结构学习"对照，修早停 bug。

**统计严谨性**：5-seed + Wilcoxon + bootstrap CI + ECE + AUC-PR。

这条线**单一来源（pact.py 主线）、每个组件有理论/实验支撑、创新点真实（conformal UQ + 多方法 AD + LOEO + 候选验证）、诚实（PCRL 负面结果）**，满足用户"效果好、可解释、有统计意义、创新强、实用强"的全部要求。
