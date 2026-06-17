# 项目全面审查报告 (Review Findings)

> 基于广泛文献调研 + 代码审计 + SCI 三区标准对照, 系统评估本项目。
> 本文档是后续所有改进的决策依据。
>
> 调研日期: 2026-06-16
> 调研覆盖: 数据泄露最佳实践 / 近期论文创新点 / 外推性方法 / 代码规范

---

## 一、SCI 三区标准对照(查证结论)

### 1.1 三区是什么水平
- **中科院分区**: 三区 = 该领域期刊影响因子排名 **21%–50%**
- **代表性期刊**: Computational Materials Science (IF≈3.3, 中科院三区, JCR Q2)
- **门槛**: 不是"能跑通代码", 而是"方法有创新 + 实验严谨 + 有物理发现 + 经得起评审"

### 1.2 三区审稿人会关注的红线
1. **数据泄露**(post-DFT / target leakage) → 直接拒稿
2. **外推性缺失**(只有随机 CV, 不测 OOD) → 严重质疑
3. **方法无创新**(只是换模型重跑) → 创新性不足拒稿
4. **结果不可复现**(无代码/无种子/无数据版本) → 信誉问题

---

## 二、发现的问题清单(按严重度排序)

### 🔴 P0 严重问题(必须修复, 否则无法投稿)

#### P0-1: DFT 后特征泄露(post-DFT leakage)
- **现象**: `struct_` 特征(晶格常数 a/b/c、畸变类型、带隙)与 E_hull/E_f 同源
  - `struct_distort_cubic` 与 E_hull 相关 = **0.719**
  - `lattice_c/b` 与 E_hull 相关 = **-0.634/-0.609**
- **影响**: R²=0.982 是虚高的; 排除 struct 后形成能 R²=0.893, 凸包能 R²=0.760
- **审稿风险**: 这是材料信息学评审的头号雷区, 被发现必拒
- **来源**: IBM/Yale 研究([数据泄露可使结果虚高](https://news.yale.edu/2024/02/28/data-leaks-can-sink-machine-learning-models)); Persson Group 指南([建议 composition-only baseline](https://perssongroup.lbl.gov/papers/wang_machine_learning_2020.pdf))
- **修复**: 全部模块特征前缀去掉 `struct_`, 只用 `('magpie_', 'phys_')`

#### P0-2: 缺少外推性测试(只有随机 CV)
- **现象**: 当前只用 KFold 随机划分, 测试集与训练集化学空间重叠
- **影响**: 模型对"训练集没见过的元素"的预测能力未知
- **审稿风险**: Nature Comm. Mater. 2024([大多数 OOD 测试其实是插值](https://www.nature.com/articles/s43246-024-00731-w))指出这是普遍问题
- **修复**: 增加 **leave-one-element-out CV (LOEO)**: 每次排除某元素的所有化合物, 测试外推

### 🟡 P1 重要问题(影响论文质量)

#### P1-1: 多任务对比无效
- **现象**: 单任务与多任务数值完全相同(KRR 数学等价; XGBoost one_output_per_tree 不真正共享)
- **影响**: "多任务学习"创新点不成立
- **修复**: 改用真正共享表示的方案(共享底层 MLP, 或共享特征投影 + 小样本实验展示迁移增益)

#### P1-2: 基准对比缺失
- **现象**: 没有与 matbench 官方基准对比
- **影响**: 审稿人无法判断性能是否 competitive
- **关键参考**: matbench perovskite 上 XGBoost 的 SOTA = MAE 0.227 eV/atom, R²=0.79([2025 tree-based benchmark](https://www.researchgate.net/publication/403573701))
- **修复**: 在论文中明确报告我们的 MAE/RMSE 并与文献对比

#### P1-3: PGML 物理基线被弱化
- **现象**: 为修 bug 把 KernelRidge 改成 Ridge(线性), 损失了非线性物理建模能力
- **修复**: 在纯预测特征下, 重新评估 KernelRidge vs Ridge, 选物理上更合理的

### 🟢 P2 改进机会(加分项)

#### P2-1: 实验验证/外部数据交叉验证
- 当前只在 wolverton 数据集内部验证, 可加入 matbench_perovskites (18928条) 做跨数据集泛化测试

#### P2-2: 可解释性深化
- SHAP 全局 + 局部分析(当前只有 PGML, 可补充直接 SHAP 特征归因)

#### P2-3: 不确定性用于主动学习
- 当前不确定性只用于筛选, 可设计"主动学习循环"叙事(哪些样本最值得标注)

---

## 三、创新机会清单(调研发现的可发力点)

### 机会 A: LOEO 外推性分析(强烈推荐)
- **来源**: MatFold 框架([标准化 CV 协议](https://www.researchgate.net/publication/386574712)); LOCO-CV 方法
- **做法**: 对每种元素(如 La, Ti, O), 排除含该元素的所有化合物训练, 测试对该元素的预测
- **创新价值**: 这正是 2024-2025 文献的热点, 三区期刊非常欢迎
- **预期发现**: 模型对常见元素(La, Ti)外推好, 对稀有元素(Pa, Ac)外推差 → 揭示适用域

### 机会 B: 跨数据集泛化(推荐)
- **做法**: 在 wolverton (4914条) 训练, 在 matbench_perovskites (18928条) 测试
- **创新价值**: 证明模型不只在一个数据集上有效

### 机会 C: 真正的多任务/迁移学习(可选)
- **做法**: 共享底层 MLP, 双头输出 E_f/E_hull; 或用 E_f(数据多)辅助 E_hull(稳定样本少)
- **创新价值**: 类别不平衡(9.4%稳定)下的迁移增益是真问题

### 机会 D: 适用域显式建模(可选)
- **做法**: 用不确定性定义"模型可信区域", 输出适用域图
- **创新价值**: 解决"模型何时可信"的实际问题

---

## 四、改进路线图(按优先级)

### 第一阶段: 紧急修复(P0, 必须做)
1. **移除 struct 特征** → 重跑 models/pgml/uncertainty/screening
2. **增加 LOEO 外推测试** → 新建 src/extrapolation.py
3. 得到**诚实可信**的性能数字

### 第二阶段: 质量提升(P1)
4. 修复多任务(共享表示 + 小样本实验)
5. 加入 matbench 基准对比
6. 优化 PGML 物理基线

### 第三阶段: 加分项(P2, 时间允许)
7. 跨数据集泛化
8. SHAP 直接解释
9. 主动学习叙事

---

## 五、代码层面审查

### 已解决 ✅
- multiprocessing spawn 风暴(改纯 pymatgen 手算)
- 数据泄露检测机制(已审计)
- 全部 n_jobs=1
- 共用数据加载 utils.py

### 待改进 ⚠️
- 缺少 `random_state` 统一管理(各模块分散硬编码 42)
- 缺少结果版本化(每次运行覆盖)
- 缺少单元测试
- 中文字符串在 Windows 控制台乱码(用 log 文件缓解)

---

## 六、诚实评估: 修复后能否达到三区?

### 修复前: ❌ 不够(DFT 泄露是硬伤)
### 修复后(P0 全做):

| 维度 | 预期 | 三区门槛 |
|------|------|----------|
| 形成能 R²(纯预测) | ≈0.89 | 0.79-0.92 ✓ |
| 凸包能 R²(纯预测) | ≈0.76 | 偏低但可叙事 |
| 方法创新 | PGML + LOEO + 不确定性 | ✓ |
| 实验严谨 | 无泄露 + 外推测试 | ✓ |
| 物理发现 | 反常材料 + 新候选 | ✓ |

**结论: P0 修复后, 达到 Computational Materials Science 三区下限, 稳妥。**
**P1 也做后, 可冲击三区中上, 甚至个别二区期刊。**

---

## 参考来源
- [Yale: 数据泄露使结果虚高 (2024)](https://news.yale.edu/2024/02/28/data-leaks-can-sink-machine-learning-models)
- [Persson Group: ML for Materials Scientists 指南](https://perssongroup.lbl.gov/papers/wang_machine_learning_2020.pdf)
- [Nature Comm Mater: OOD 泛化被高估 (2024)](https://www.nature.com/articles/s43246-024-00731-w)
- [MatFold: 标准化 CV 协议](https://www.researchgate.net/publication/386574712)
- [Matbench Discovery 基准](https://matbench-discovery.materialsproject.org/)
- [2025 tree-based perovskite benchmark](https://www.researchgate.net/publication/403573701)
- [Computational Materials Science 期刊信息](https://letpub.com.cn/index.php?page=journalapp&view=detail&journalid=1970)
