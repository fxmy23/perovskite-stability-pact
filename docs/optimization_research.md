# 深度文献调研 → 模型优化方案

**日期**: 2026-06-16
**目标**: 广泛深度调研后, 找到真正能优化我们模型效果的技术。

---

## 一、调研发现: 我们的3个核心问题都有文献解法

### 问题A: R²=0.910 低于 SOTA (0.928)
**文献解法** (MDPI Materials 2025, R²=0.928):
- **SHAP 特征选择**: 选出9个关键特征, 降噪提精度
- **元启发超参优化**: 网格/贝叶斯搜索最优超参
- **适用**: ✅ 我们可直接做 (Optuna + SHAP特征选择)

### 问题B: 物理层冗余 (ML已吸收物理信息)
**文献解法** (Digital Discovery 2024, Adv. Extrapolative Prediction):
- **物理层不该是"独立预测器+残差", 而应是"物理约束特征"**
- 真正有效的是: 把物理量作为**层级描述符** (hierarchical), 而非平行特征
- **适用**: ⚠️ 需重新设计特征 (中等工作量)

### 问题C: LOEO 外推差 (R² 0.70, 多元素崩) ★最重要发现
**文献解法** (RSC Digital Discovery 2024, 直接对症):
- **RULSIF (相对无缝密度比估计)**: 实例权重调整, 给"像测试域"的训练样本更高权重
  → sparse-X 任务 MAE 降低 **28%**
- **KMM (核均值匹配)**: 类似, 匹配训练/测试分布
- **TrAdaBoost**: 迁移学习, 降低源域权重, 提升目标域
- **适用**: ✅✅ **这是直接解决我们LOEO问题的技术!** RULSIF实现简单(权重计算+加权训练)

## 二、优化方案优先级 (按 ROI 排序)

### 优化1 ★★★★★ RULSIF 域适应 (解决LOEO外推, 最大创新点)
**做什么**: 在LOEO中, 训练样本按"与被留出元素的分布相似度"加权。
相似度高的样本(化学性质接近被留出元素)权重高 → 模型更关注能迁移的知识。
**文献支撑**: Digital Discovery 2024, sparse-X MAE降28%
**预期**: LOEO R² 0.70→0.78-0.82 (显著)
**创新**: 这是真方法创新 (域适应+材料LOEO的组合罕见)
**成本**: 中 (RULSIF权重计算 ~50行代码)

### 优化2 ★★★★ Optuna 超参搜索 (提升CV R²)
**做什么**: 贝叶斯优化LightGBM超参 (n_leaves/lr/reg/colsample等)
**文献支撑**: MDPI 2025 R²=0.928就靠系统超参搜索
**预期**: CV R² 0.910→0.920-0.925
**成本**: 中 (50 trials, 每trial~1min, 可后台)

### 优化3 ★★★ SHAP 特征选择 (降噪)
**做什么**: 用SHAP选top-30/50特征, 去除噪声特征
**文献支撑**: MDPI 2025 选9特征反而R²更高
**预期**: R² +0.005-0.01, 模型更简洁
**成本**: 低

### 优化4 ★★★ GBDT Stacking (LGB+XGB+CatBoost)
**做什么**: 3种GBDT的stacking + Ridge meta
**预期**: R² +0.005-0.01
**成本**: 中 (需catboost)

### 优化5 ★★ 层级物理描述符
**做什么**: 物理特征按"A位性质/B位性质/AB组合/几何"分层
**预期**: 可能提外推, 不确定
**成本**: 中高

## 三、实施顺序 (ROI驱动)

1. **优化1 (RULSIF)** — 最高创新+解决核心痛点(外推), 先做
2. **优化2 (Optuna)** — 提CV精度, 文献直接支撑
3. **优化3 (SHAP特征选择)** — 简单, 配合优化2
4. (可选) 优化4 (stacking)

做完1+2+3, 预期: CV R² 0.910→0.92+, LOEO R² 0.70→0.80+,
且 RULSIF 域适应是真方法创新 (论文亮点)。

## 四、文献来源

- [Realistic material property prediction using domain adaptation (RSC Digital Discovery 2024)](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00162h) — RULSIF/KMM, 直接对症LOEO
- [Prediction of ABX3 Formation Energy R²=0.928 (MDPI Materials 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12250765/) — SHAP特征选择+超参
- [Advancing extrapolative predictions (Nature Comm. Mater. 2025)](https://www.nature.com/articles/s43246-025-00754-x) — 迁移学习外推
- [Extrapolative prediction with hierarchical descriptors (APL 2025)](https://pubs.aip.org/aip/apl/article/127/23/232704/3374713/) — 层级物理特征
- [Interpretable models for extrapolation (RSC Digital Discovery 2023)](https://pubs.rsc.org/en/content/articlehtml/2023/dd/d3dd00082f) — 物理特征线性映射外推
