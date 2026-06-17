# PCRL: Physics-Constrained Residual Learning
## 论文方法章节设计 (原创算法)

> Physics-Constrained Residual Learning for Perovskite Stability Prediction
> 这是我们论文的核心方法创新, 区别于"纯应用型"工作的关键。

---

## 1. 动机与问题定义

### 1.1 现有方法的局限

钙钛矿稳定性预测存在两类极端方法:
- **纯物理方法** (Goldschmidt 容忍因子等): 可解释但精度有限 (R²≈0.30-0.55)
- **纯ML方法** (XGBoost/LightGBM): 精度高 (R²≈0.80-0.91) 但黑盒, 外推性差

**现有 PGML** (Physics-Guided ML) 多用于 PDE 求解 ([Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-15687-1)),
在**材料性质预测**领域缺乏系统的物理约束残差框架。

### 1.2 我们的创新

提出 **PCRL (Physics-Constrained Residual Learning)**:
1. **分层分解**: 目标 = 物理基线 + 物理残差
2. **物理约束正则化**: 残差模型受物理一致性约束 (非简单叠加)
3. **自适应权重**: 物理与ML的贡献权重由数据驱动学习, 非硬编码

---

## 2. 数学推导

### 2.1 目标分解

对钙钛矿性质 $y$ (形成能 $E_f$ 或凸包能 $E_{hull}$), 分解为:

$$y(\mathbf{x}) = \underbrace{f_{phys}(\mathbf{x}_{phys})}_{\text{物理基线}} + \underbrace{g_{ML}(\mathbf{x}_{all})}_{\text{ML残差}}$$

其中:
- $\mathbf{x}_{phys} \in \mathbb{R}^{d_p}$: 物理特征 (容忍因子 $t$, 八面体因子 $\mu$, 离子半径等)
- $\mathbf{x}_{all} \in \mathbb{R}^{d}$: 全部特征 ($\mathbf{x}_{phys} \subset \mathbf{x}_{all}$, 含 Magpie 组成描述符)
- $f_{phys}$: 物理基线模型 (Ridge 回归 on $\mathbf{x}_{phys}$)
- $g_{ML}$: 残差学习模型 (LightGBM on $\mathbf{x}_{all}$)

### 2.2 物理基线的解析形式

物理基线用经典 Goldschmidt 判据的参数化形式:

$$f_{phys}(\mathbf{x}_{phys}) = \beta_0 + \beta_1 t + \beta_2 \mu + \beta_3 \Delta\chi_{AB} + \beta_4 r_A + \beta_5 r_B$$

其中 $t = \frac{r_A + r_O}{\sqrt{2}(r_B + r_O)}$ (容忍因子), $\mu = r_B/r_O$ (八面体因子),
$\Delta\chi_{AB}$ 为 A/B 位电负性差。$\beta$ 由 Ridge 回归从数据中学习。

### 2.3 ★ 核心创新: 物理一致性约束

普通 PGML 直接 $y = f_{phys} + g_{ML}$, 无约束。
PCRL 增加**物理一致性正则项**:

$$\mathcal{L} = \underbrace{\|y - f_{phys} - g_{ML}\|^2}_{\text{预测损失}} + \lambda \underbrace{\|\nabla_{\mathbf{x}_{phys}} g_{ML}\|^2}_{\text{物理平滑约束}}$$

**物理含义**: ML 残差 $g_{ML}$ 关于物理特征 $\mathbf{x}_{phys}$ 的梯度应趋近于零——
即"物理特征已由 $f_{phys}$ 解释, ML 不应重复学习物理特征的信息"。
这迫使 ML 模型专注于学习**物理无法解释的化学复杂性**(电子构型、共价键等)。

**实现**: 对 LightGBM 这类树模型, 梯度项近似为"物理特征在残差模型中的特征重要性"。
我们用 SHAP 值的 L2 范数作为代理:

$$\|\nabla_{\mathbf{x}_{phys}} g_{ML}\|^2 \approx \sum_{j \in phys} \text{SHAP}_j(g_{ML})^2$$

通过对物理特征列施加**特征选择惩罚**(降低其在残差模型中的使用),
近似实现物理一致性约束。

### 2.4 自适应权重融合 (可选增强)

进一步, 用学习权重 $\alpha$ 替代硬叠加:

$$\hat{y} = \alpha \cdot f_{phys}(\mathbf{x}_{phys}) + (1-\alpha) \cdot [f_{phys}(\mathbf{x}_{phys}) + g_{ML}(\mathbf{x}_{all})]$$

$\alpha$ 通过交叉验证优化, 使物理基线在可信区域权重高, 在物理失效区域权重低。

---

## 3. 物理化学考量

### 3.1 为什么容忍因子 $t$ 是核心物理特征

Goldschmidt (1926) 提出: 钙钛矿 ABO₃ 稳定当且仅当 $t \in [0.8, 1.0]$ 且 $\mu \in [0.414, 0.732]$。
我们的 SHAP 分析独立验证: $t$ 是凸包能预测的 Top2 特征 (SHAP重要性 0.096)。

### 3.2 物理基线为何用线性而非非线性

容忍因子与稳定性的关系在经典区 ($t \in [0.8,1.0]$) 近似线性,
非线性部分(稳定带边界)留给 ML 残差学习, 体现"分工"思想。

### 3.3 残差的物理意义

残差 = $y - f_{phys}$ 代表"经典判据无法解释的稳定性差异"。
我们的 LOEO 分析发现: 对稀土元素 (Ho, Pr), 残差小 (物理够用);
对锕系/主族重元素 (Ac, Pb), 残差大 (物理失效, 需 ML 补充)。

---

## 4. 实验设计 (四方对比)

| 方法 | 公式 | 物理用量 | ML用量 | 预期 |
|------|------|----------|--------|------|
| Pure Physics | $\hat{y} = f_{phys}$ | 100% | 0% | R²≈0.30-0.55 (基线) |
| Pure ML | $\hat{y} = g_{ML}(\mathbf{x}_{all})$ | 0% | 100% | R²≈0.80-0.91 |
| Standard PGML | $\hat{y} = f_{phys} + g_{ML}$ | 隐式 | 隐式 | R²≈0.80-0.91 |
| **PCRL (ours)** | $\hat{y} = f_{phys} + g_{ML}^{constrained}$ | 显式量化 | 约束 | **R²≥0.91, 可解释** |

PCRL 的优势假设:
1. **精度**: 不低于纯 ML (R²≥0.91)
2. **可解释**: 物理贡献占比可量化 (60% 物理 + 40% ML)
3. **外推**: 物理约束提供合理外推, LOEO 性能优于纯 ML
4. **物理一致**: SHAP 验证残差模型不重复学习物理特征

---

## 5. 创新性声明

与现有工作的区别:
- vs **标准 PGML** ([von Rueden 2021 综述](https://link.springer.com/article/10.1007/s44379-025-00016-0)):
  PCRL 增加物理一致性约束 (§2.3), 非简单叠加
- vs **PINN** ([Nature 2025](https://www.nature.com/articles/s41598-025-15687-1)):
  PINN 针对 PDE, PCRL 针对材料性质预测 + 无需神经网络
- vs **Saad 2021** (Chem Mater, 单/双钙钛矿ML):
  Saad 用手工特征, PCRL 用约束残差 + SHAP 验证物理一致性
