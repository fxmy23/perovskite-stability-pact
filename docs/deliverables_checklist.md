# 论文待办图表/声明/数据 完整清单

**日期**: 2026-06-16
**用途**: 投稿前必须完成的所有交付物，按类别+优先级+数据来源+规范要求整理。
**所有数据来源已核实存在（见下方"数据来源"列）。**

---

## 一、图 (Figures)

| 编号 | 图名 | 优先级 | 数据来源(已核实) | 规范要求 | 状态 |
|---|---|---|---|---|---|
| F1 | **Parity plot** 预测vs真实散点(形成能+凸包能，2面板) | P0★★★★★ | pact_v2_oof_*.csv (y_true, oof_mu列) | y=x参考线，标注R²/MAE，等比例坐标，不跨0，色标=σ | ❌未做 |
| F2 | **PACT框架工作流图** | P0★★★★ | 手绘(无数据) | 输入→物理层→ML层→conformal/CQR→AD→输出，分块着色 | ❌未做 |
| F3 | **SHAP特征重要性条形图**(top-15，分类着色Magpie/物理) | P1★★★★ | shap_importance.csv | 横向条形，按重要性排序，颜色区分类别 | ❌未做 |
| F4 | **SHAP dependence plots**(top-6特征，非线性关系) | P1★★★★ | 需重跑SHAP(有model) | x=特征值，y=SHAP值，散点+趋势线，每特征一面板 | ❌未做 |
| F5 | **不确定性校准图**(reliability diagram) | P1★★★ | pact_v2_oof_*.csv (σ,conf_lower/upper) | x=σ分位桶，y=实际覆盖率，y=0.80参考线 | ❌未做 |
| F6 | **CQR vs 标准conformal区间对比** | P1★★★ | cqr_results.csv + 需重跑存OOF | 可信/不可信样本的MPIW对比条形图 | ❌未做 |
| F7 | **LOEO外推R²热力图**(68元素) | P2★★★ | loeo_sr_full_results.csv | 元素按周期表排列，颜色=R² | ❌未做 |
| F8 | **应用域可视化**(可信/不可信分区) | P2★★ | pact_v2_oof_*.csv (trust_sigma) | 散点x=预测y=误差，颜色=可信/不可信 | ❌未做 |
| F9 | **混淆矩阵**(凸包能稳定分类) | P2★★ | classification_oof.csv | 2×2混淆矩阵，标注TP/FP/FN/TN+百分比 | ❌未做 |
| F10 | **ROC曲线**(凸包能分类) | P2★★ | classification_oof.csv | ROC曲线，标注AUC，对角参考线 | ❌未做 |
| F11 | **SR公式共识条形图**(a_site_en 100%等) | P2★★ | sr_ensemble_consensus.csv | x=物理特征，y=出现频率%，100%高亮 | ❌未做 |
| F12 | **数据分布图**(形成能/凸包能直方图) | P2★ | perovskite_features.csv | 2面板直方图，标注均值/方差 | ❌未做 |
| F13 | **元素分布周期表热力图** | P2★ | features(a_site_element) | 周期表布局，颜色=样本数 | ❌未做 |

### 图的统一规范要求（投稿级）
- **分辨率**: ≥300 dpi（矢量图PDF/EPS优先，位图PNG≥300dpi）
- **字体**: Arial/Helvetica，8-12pt（坐标轴标签≥9pt，刻度≥8pt）
- **线宽**: 数据线≥1.5pt，参考线≥1pt虚线
- **配色**: 色盲友好（避免红绿对比，用viridis/colorblind palette）
- **坐标轴**: 必须有标签+单位（如"Formation energy (eV/atom)"）
- **图例**: 不覆盖数据，框线清晰
- **图注**: 图下方，Fig. 1. 开头，含必要说明
- **尺寸**: 单栏85mm，双栏180mm

---

## 二、表 (Tables)

| 编号 | 表名 | 优先级 | 数据来源(已核实) | 规范要求 | 状态 |
|---|---|---|---|---|---|
| T1 | **模型对比表**(R²/MAE/RMSE，多模型×2目标) | P0★★★★★ | pact_v2_results.csv + stacking_results.csv | 含**文献基准行**(MDPI2025 R²0.928等)，最佳值加粗 | ❌未做 |
| T2 | **消融实验表**(各组件贡献) | P1★★★★ | 综合多文件 | 行=配置(纯ML/物理层/stacking/CQR等)，列=R²/PICP | ❌未做 |
| T3 | **不确定性指标对比表**(PICP/MPIW/ECE) | P1★★★ | pact_v2_results.csv + cqr_results.csv | 标准conformal vs CQR对照 | ❌未做 |
| T4 | **SHAP top-15特征表** | P2★★ | shap_importance.csv | 特征名+重要性+类别+物理含义 | ❌未做 |
| T5 | **候选材料验证表** | P1★★★ | candidate_validation.csv | 公式+预测E_hull+σ+验证等级+matbench状态 | ❌未做 |
| T6 | **LOEO外推结果表**(代表性元素) | P2★★ | loeo_sr_full_results.csv | 选8-10元素展示R²，注明好/差外推 | ❌未做 |
| T7 | **超参表**(Optuna最佳，主文或SI) | P2★★ | optuna_results.csv | 模型×超参名×最佳值 | ❌未做 |
| T8 | **数据集描述表**(特征维度/来源/规模) | P1★★ | features.py | 特征类别+维度+含义 | ❌未做 |

### 表的统一规范要求
- **三线表**（顶线/栏目线/底线，无竖线）—— Elsevier标准
- **最佳值加粗**
- **数值精度一致**（R²保留3-4位小数，MAE保留3位）
- **表注**: 表下方，含缩写定义+统计学标注（如*p<0.05）
- **文献引用**: 表内直接标[ref]

---

## 三、声明/章节 (Statements/Sections)

| 编号 | 名称 | 优先级 | 内容要求 | 状态 |
|---|---|---|---|---|
| S1 | **Data Availability Statement** | P0★★★★★(强制) | "wolverton_oxides via matminer; features in data/processed/; ..." | ❌未写 |
| S2 | **Code Availability** | P1★★★★ | GitHub链接+license+运行说明 | ❌未写 |
| S3 | **Limitations Section** | P1★★★ | 单源数据/R²上限/LOEO外推差/SR非唯一/候选无DFT | ❌未写 |
| S4 | **CRediT Author Contributions** | P1★★★ | 按CRediT分类（概念/方法/代码/写作等） | ❌未写 |
| S5 | **Acknowledgments** | P2★★ | 致谢导师/基金（如有） | ❌未写 |
| S6 | **Conflicts of Interest** | P2★★ | "作者声明无利益冲突" | ❌未写 |

---

## 四、方法学描述需补全的内容

| 编号 | 内容 | 优先级 | 现状 | 状态 |
|---|---|---|---|---|
| M1 | **数据描述叙事**(4914样本来源+清洗+切分) | P0★★★★ | 数据在代码，无叙事 | ❌未写 |
| M2 | **特征工程叙事**(113维=96+14+3，排除3衍生) | P1★★★ | 在utils.py，无叙事 | ❌未写 |
| M3 | **CQR理论推导**(条件覆盖+ECE定义) | P1★★★★ | 在cqr_theory_results.md，需转论文体 | ❌未写 |
| M4 | **conformal理论**(split conformal算法+保证) | P1★★★ | 在conformal.py注释，需转论文体 | ❌未写 |
| M5 | **评估协议**(5-fold CV+bootstrap CI+Wilcoxon) | P1★★★ | 在stats_eval.py，需叙事 | ❌未写 |
| M6 | **SR方法叙事**(gplearn+符号集成共识) | P1★★★ | 在sr_ensemble.py，需叙事 | ❌未写 |

---

## 五、总览统计

| 类别 | P0必做 | P1推荐 | P2可选 | 总计 |
|---|---|---|---|---|
| 图 | 2 (F1,F2) | 4 (F3-F6) | 7 (F7-F13) | 13 |
| 表 | 1 (T1) | 4 (T2,T3,T5,T8) | 3 (T4,T6,T7) | 8 |
| 声明 | 1 (S1) | 3 (S2,S3,S4) | 2 (S5,S6) | 6 |
| 方法叙事 | 1 (M1) | 5 (M2-M6) | 0 | 6 |
| **合计** | **5** | **16** | **12** | **33** |

**P0必做（5项）是投稿门槛，必须在投稿前完成。**
**P1推荐（16项）显著提升录用率。**
**P2可选（12项）锦上添花。**

---

## 六、关键提醒（针对你担心的"图不规范"问题）

1. **所有图的数据已核实存在**（数据来源列已验证）
2. **图的规范在"统一规范要求"里明确**（分辨率/字体/配色/坐标轴）
3. **建议用matplotlib + 科学样式**（seaborn/SciencePlots），生成PDF矢量图
4. **每个图我会先列规范再画**，画完自检是否符合上述要求
5. **表用三线表格式**（Elsevier标准），不画竖线

## 七、实施顺序建议

**第一阶段（P0，投稿门槛）**: F1(parity) + F2(工作流) + T1(对比表) + S1(数据声明) + M1(数据叙事)
**第二阶段（P1，提录用）**: F3-F6(图) + T2,T3,T5,T8(表) + S2,S3,S4(声明) + M2-M6(叙事)
**第三阶段（P2，可选）**: 剩余图表
