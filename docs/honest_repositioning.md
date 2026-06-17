# 诚实重定位: SR共识 / CQR理论 / 物理锚定 (P1-2)

**原则**: 不夸大每个组件的贡献, 每个定位都有依据。

---

## 1. SR符号集成共识 — 重定位

### 原定位 (过度)
"SR发现了a_site_en(电负性)主导形成能, 这是创新发现"

### 问题
- 电负性主导形成能是化学常识(化学键理论)
- gplearn有限算子让a_site_en"最易表达", 可能非真"最重要"
- 100%共识只共享一个特征(a_site_en), 公式结构仍各异

### 诚实重定位
"SR符号集成**验证**了已知物理直觉(电负性是形成能的主要描述符),
在50个独立公式中100%稳定出现。SR的真正贡献是:
(a) 提供了与黑盒ML互补的可解释视角;
(b) 定量给出了SR在不同材料性质上的适用边界(形成能SNR2.61成功vs凸包能SNR0.94失败)。
**SR共识本身不算'发现', 适用边界分析才是贡献。**"

### 论文措辞
"We employed symbolic regression not to discover new physics, but to provide
an interpretable cross-check of the ML model. The 100% consensus on A-site
electronegativity across 50 equations confirms known chemistry; the methodological
contribution is the empirical SR suitability criterion derived from comparing
formation energy (SNR=2.61, success) and hull energy (SNR=0.94, failure)."

## 2. CQR条件保形预测 — 重定位

### 原定位 (过度)
"CQR是我们的理论创新, ECE改善48%"

### 问题
- CQR是Romano 2019提出的方法, 我们是应用
- ECE改善是实证, 非理论推导

### 诚实重定位
"CQR (Romano et al. 2019) 是已有的条件保形方法。我们的贡献是:
(a) 首次将其应用于钙钛矿稳定性预测的不确定性量化;
(b) 实证显示条件覆盖偏差(ECE)在公平对照下改善~60% (0.085→0.034, 形成能);
(c) 提供了异方差区间(可信样本窄, 不可信样本宽), 优于均匀区间。
**这是'方法应用+实证改进', 非新理论。**"

### 论文措辞
"We adopt Conformalized Quantile Regression (CQR; Romano et al., 2019) for
uncertainty quantification. While CQR itself is an established method, its
application to perovskite stability prediction is novel, and we empirically
demonstrate a 60% reduction in expected calibration error (ECE) relative to
standard split conformal with a fair (same-family) point predictor, yielding
heteroscedastic intervals that adapt to sample-level uncertainty."

## 3. 物理锚定 (Physics-Anchored) — 重定位

### 原定位 (过度)
"PACT是Physics-Anchored框架, 物理层锚定整个预测"

### 问题
- 物理层KRR只锚定点预测(μ_p), CQR区间完全独立
- 物理层不提升精度(纯ML反超)
- "Anchored"暗示物理约束整个框架, 但实际只约束点预测

### 诚实重定位
"物理层(KernelRidge on 14物理特征)提供**点预测的可解释基线**(μ_p),
ML残差(stacking)在其上补足非线性。CQR区间是独立的数据驱动校准,
不依赖物理层。因此'physics-anchored'应理解为:
物理先验锚定**点预测的解释性**(而非约束不确定性区间)。
SR共识(a_site_en 100%)是该锚点合理性的证据。"

### 论文措辞
"The physics layer provides an interpretable baseline for the point prediction;
the CQR interval is calibrated independently. We term this 'physics-informed
point prediction with decoupled conformal intervals' rather than claiming
physics constrains the uncertainty estimates."

## 4. 框架整体命名建议
原: "PACT (Physics-Anchored Calibrated Trust)" — Anchored过强
建议: "PIC-PACT (Physics-Informed Point prediction with Calibrated Conformal intervals
  + Applicability-domain-guided Trust)" 或简化为
"uncertainty-aware perovskite stability prediction with conditional conformal intervals"
— 强调CQR(实证贡献)+AD(差异化), 弱化"anchored"

## 5. 投稿定位 (诚实修订)
- **三区 (Computational Materials Science)**: 稳, CQR应用+AD+LOEO+SR边界是差异化
- **二区**: 弱。无新理论(CQR是应用), 无DFT验证, 单源数据
- **论文卖点排序** (诚实):
  1. CQR条件覆盖实证改善60% (应用+实证, 非理论)
  2. 多方法应用域 (σ/kNN/leverage, 罕见组合)
  3. 68元素LOEO外推评估 (多数论文不做)
  4. SR适用边界判据 (经验, 基于两目标)
  5. SR共识验证已知物理 (非发现)
