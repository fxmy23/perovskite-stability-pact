# 四项必修任务: 科学论证 + 文献佐证 + 终极方案

**日期**: 2026-06-16
**原则**: 每项修改先论证科学正确性, 文献佐证, 再确定方案, 最后执行。

---

## 项1: 残差框架内ML层换stacking — 论证

### 待改什么
当前 PACT 的ML残差层 = 5×单LightGBM集成 (R²=0.910)。
改为 = GBDT stacking (LightGBM+XGBoost+HistGBT + Ridge元学习器, R²=0.918)。

### 科学问题: 这是否"绕过"物理层?
**质疑**: 把更强的ML放进残差层, 是否让物理层更被边缘化 (残差变小→物理层占比变高)?
**分析**:
- 残差 = y − μ_p(物理层). stacking学这个残差, 不接触y的直接值。
- 物理层的μ_p**不变** (仍KernelRidge on 14物理特征, R²≈0.79)。
- stacking只是把"残差预测得更好" → 总μ = μ_p + μ_r 更准。
- **这恰恰是残差框架的设计意图**: 物理层提供可解释基线, ML补足剩余。
- 换stacking不改变物理层角色, 只提升残差预测质量。

### 文献佐证 (stacking在残差/混合框架中是标准做法)
- **Residual-Aware Stacking (RAS)** [SSRN 2025](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5160281):
  "训练额外模型预测base model的残差" — 正是我们的框架, 是新近正式化的方法。
- **Hybrid Physics-ML with GP** [ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S2590123026010856):
  "残差学习在物理基线有系统偏差时最有益" — 支持物理层+残差分离。
- **Two-Stage Dynamic Stacking** [Emergent Mind](https://www.emergentmind.com/topics/two-stage-dynamic-stacking-ensemble-model):
  两阶段stacking是成熟方法。

### 结论: 改动科学正确 ✅
- stacking放残差层是RAS/混合框架的标准做法
- 不绕过物理层 (μ_p不变, 仍是可解释锚点)
- 物理层贡献占比 = base_R²/total_R², 换stacking后:
  base_R²(0.79)/total_R²(0.918) = 86% (比单LightGBM的0.79/0.910=87%略低, 几乎不变)
- 物理层叙事不受影响

### 方案
- 写 src/pact_final.py: 物理层(KRR) + ML残差(stacking LGB+XGB+HistGBT) + conformal + AD
- 嵌套CV: 外层5折评估, 内层3折生成基模型OOF + Ridge元学习器
- 物理层μ_p用KernelRidge(主线), SR作"可解释性附录"(报共识不报精度)

---

## 项2: 点预测与CQR区间解耦 — 论证

### 待改什么
当前CQR的点预测用中位数分位数回归(R²=0.893), 低于主线stacking(0.918)。
改为: 点预测 = stacking主线(0.918), 区间 = CQR分位数模型(条件覆盖好)。

### 科学问题: 点预测和区间预测用不同模型, 是否统计合理?
**质疑**: 这不是"一套模型出全套结果", 审稿人会不会质疑不一致?
**分析**:
- **conformal prediction本身就是解耦设计**: 区间预测基于"非一致性分数",
  与点预测模型可独立。这是conformal的核心灵活性。
- CQR (Romano 2019) 的原始设计: q10/q90分位数模型给区间, 点预测可以是任意模型。
- **"best of both worlds"是conformal社区公认模式**: 用最好的点模型+最好的区间校准。

### 文献佐证 (解耦是conformal的标准且推荐做法)
- **CQR原始论文** [Romano 2019 NeurIPS](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf) (1416引用):
  CQR的区间来自分位数模型, 点预测独立。
- **Tidymodels conformal** [tidymodels.org](https://www.tidymodels.org/learn/models/conformal-regression/):
  明确展示"interval模型独立于point预测模型"的实现。
- **predictset R包** [CRAN](https://cran.r-project.org/web/packages/predictset/refman/predictset.html):
  CQR实现"要求分位数/点预测用独立模型"。
- **Arel-Bundock conformal notebook** [arelbundock.com](https://arelbundock.com/posts/conformal/index):
  展示不同conformal score函数, 点/区间分离处理。

### 结论: 解耦科学正确, 且是推荐做法 ✅
- conformal的设计哲学就是区间独立于点预测
- 用stacking(最强点预测)+CQR(最佳区间校准)是"best of both worlds"
- 论文需明确写"点预测与区间预测采用解耦设计"并引用Romano 2019

### 方案
- 点预测: stacking主线 (R²0.918) — 主结果
- 区间: CQR分位数模型 (ECE 0.168, PICP 0.80) — 不确定性主结果
- 论文诚实写: "点预测采用stacking回归(R²0.918), 区间采用独立的CQR
  (条件覆盖ECE=0.168, 较标准conformal改善44%)。两者解耦设计遵循
  conformal prediction的灵活性原则"

---

## 项3: 统一R²数字 — 论证

### 待改什么
文档中R²有三个版本: 0.910(单LGB)/0.918(stacking)/0.893(CQR中位数)。
论文统一用一套。

### 科学问题: 用哪套?
**分析**: 论文的主结果是"PACT框架的预测能力", 应该报**最强且连贯的管线**。
- 点预测: stacking (0.918) — 这是PACT最终管线的点精度
- CQR的点R²(0.893)是区间模型的副产品, 不是主结果, 放方法/讨论里说明

### 方案 (无争议)
- 论文主表: 形成能R²=0.918, 凸包能R²=0.813 (stacking主线)
- CQR章节: 明确"CQR点预测R²=0.893(中位数回归), 但区间ECE改善44%, 我们采用解耦设计"
- handoff.md及所有文档同步更新

---

## 项4: SR公式简化/共识报告 — 论证

### 待改什么
SR公式深嵌套不可读 (`sub(-0.191,sqrt(add(sub...`), 论文怎么呈现?

### 科学问题: 论文怎么报告SR结果才规范?
**分析 + 文献佐证**:
- **parsimony是SR核心原则** [ResearchGate 2025 challenges](https://www.researchgate.net/publication/398225826):
  "SR的挑战包括模型选择、简化、benchmarking" — 论文应报**简化后**公式
- **共识频率报告** [MDPI bias-variance 2024](https://www.mdpi.com/2076-3417/14/23/11061):
  多次SR跑的分析应报"特征出现频率"(我们做的a_site_en 100%共识正是此规范)
- **不精确简化** [Cavalab 2024](https://cavalab.github.io/2024/04/14/inexact-simplification.html):
  允许"不精确但可读"的简化公式

### 结论: 报共识频率 + 简化短公式, 不展示深嵌套 ✅
### 方案
1. **主文**: 报SR共识频率表 (a_site_en 100%, b_site_group 63%, b_site_en 63%)
2. **附录/SI**: 选1-2个**最短公式** (从sr_equations_per_fold.csv按complexity排序选最短)
3. **可选**: 用sympy对最短公式做代数简化
4. **不展示**深嵌套原始公式 (如fold1的sqrt套sqrt)
5. 叙事: "符号集成(8 seed)发现a_site_en在100%公式中出现, 表明A位电负性是形成能的主导物理因素"

---

## 终极修改方案 (确定)

### 架构: PACT-Final (统一连贯管线)
```
输入: 110维特征 (96 Magpie + 14物理, 排除3衍生/死)
   │
   ├── 物理层: KernelRidge on 14物理特征 → μ_p (可解释锚点)
   │     └─ 附录: SR符号集成共识 (a_site_en 100%, 可解释性证据)
   │
   ├── ML残差层: GBDT stacking (LGB+XGB+HistGBT + Ridge元)
   │     └─ 残差 r = y - μ_p → μ_r → 点预测 μ = μ_p + μ_r (R²0.918) ★项1
   │
   ├── 不确定性 (解耦): ★项2
   │     ├─ 点预测: μ (stacking主线, R²0.918)
   │     ├─ 排序: ensemble σ (σ-err r=0.39)
   │     ├─ 区间: CQR分位数+conformal校准 (PICP0.80, ECE0.168)
   │     └─ 应用域: σ/kNN/leverage三方法
   │
   └── 输出: μ + [CQR区间] + AD标签 + (可选)稳定分类
```

### 实施步骤 (按顺序, 每步前后debug)
1. **写 src/pact_final.py**: 上述统一管线
   - 物理层KRR + stacking残差 → 点预测
   - CQR解耦 → 区间
   - conformal + AD → 不确定性+应用域
   - 嵌套CV无泄露
2. **跑 pact_final.py** → 得统一OOF + 所有指标
3. **统一R²**: 所有文档用 pact_final 的数字 (项3)
4. **SR表征**: 从sr_equations_per_fold选最短公式, sympy简化 (项4)
5. **前后debug**: 数值稳定/无泄露/PICP≥0.80/ECE改善
6. **记录**: 写 docs/pact_final_results.md + 更新handoff.md

### 终极指标 (预期, 基于已有结果组合)
| 指标 | 值 | 来源 |
|---|---|---|
| 形成能 R² (点预测) | 0.918 | stacking主线 |
| 凸包能 R² | 0.813 | stacking主线 |
| 形成能 PICP (CQR) | 0.80 | CQR区间 |
| 形成能 ECE (CQR) | 0.168 | 条件覆盖 |
| σ-err r | 0.39 | ensemble σ |
| SR共识 | a_site_en 100% | 符号集成 |

### 不做的事 (诚实)
- 不让CQR点预测当主线 (R²0.893低, 解耦更合理)
- 不展示深嵌套SR公式 (报共识频率)
- 不夸大SR适用判据为定理 (保持经验定位)

## 文献清单 (写论文引用)
1. Romano 2019 NeurIPS - CQR原始 (解耦设计依据)
2. SSRN 2025 - Residual-Aware Stacking (物理层+stacking残差依据)
3. ScienceDirect 2026 - Hybrid Physics-ML GP (残差学习依据)
4. ResearchGate 2025 - SR challenges (SR简化/共识报告依据)
5. Cavalab 2024 - inexact simplification (SR简化方法)
6. MDPI 2024 - SR bias-variance (SR共识频率报告)
