# 项目交接摘要 (Handoff Document)

**用途**: 无论上下文如何压缩/新开会话, 先读本文件 + troubleshooting_log.md 即可恢复完整理解。
**最后更新**: 2026-06-16 (V9 — 23项问题全部修复)
**项目状态**: 全部问题处理完成, 诚信状态良好, 可进入论文写作

**★ 最新变化 (V9)**: 严苛审查发现23个问题, 全部处理。核心修复:
   - CQR ECE改善经公平对照验证**真实成立60%**(原报48%, 公平对照反升)
   - PHYS_FEATURES死特征移除(15→14维)
   - 固定超参乐观偏差+0.0053主动发现并诚实承认
   - SR共识/CQR理论/物理锚定全部诚实重定位(不夸大)
   - 6条Limitations起草(主动暴露单源/解耦张力/候选弱等)
   详见 docs/v9_all_fixes.md, docs/honest_repositioning.md, docs/limitations_draft.md

---

## 一、项目一句话概述

清华大学材料本科生, 用机器学习预测 ABO₃ 钙钛矿氧化物的**形成能**和**热力学稳定性(凸包能E_hull)**, 目标发中文 SCI 三区 (Computational Materials Science 定位)。

## 二、数据与目标

- **数据**: wolverton_oxides (matminer 内置), 4914 个 ABO₃ 样本, 无重复公式
- **特征**: 113 维 = 96 Magpie统计 + 14物理 + 3电子 (排除衍生启发+死特征后110维用于ML)
- **目标**: formation_energy_per_atom (形成能), energy_above_hull (凸包能)
- **数据上限**: R²≈0.92-0.93 (DFT数值噪声决定, 我们已达0.918)

## 三、核心框架: PACT (Physics-Anchored Conformal-Trust)

统一管线 (单一来源 src/pact_v2.py):
1. 物理层: KernelRidge on 14物理特征 → μ_p
2. ML残差层: 5×LightGBM集成 on 110特征 → μ_r
3. 统一预测: μ = μ_p + μ_r
4. 不确定性: ensemble σ (排序) + split conformal (定宽, PICP理论保证)
5. 应用域: σ/kNN/leverage 三方法一致性
6. 分类: F1/AUC/DAF 派生

**关键发现(诚实)**: 物理层(无论SR还是KRR)不提升精度 (纯ML反超), 价值在可解释性。
Magpie统计特征已冗余覆盖物理特征 (electroneg_diff被重建R²=0.998)。

## 四、最终指标 (V8 — PACT-Final 统一管线)

★ 论文所有指标来自同一管线 src/pact_final.py (复现性保证)

| 指标 | 形成能 | 凸包能 | 说明 |
|---|---|---|---|
| 点预测 R² | **0.9144** [0.907,0.922] | **0.7997** [0.781,0.816] | KRR物理层+stacking残差 |
| MAE | 0.186 | 0.171 | |
| CQR PICP | 0.802 | 0.807 | ≥0.80 理论保证 |
| **CQR ECE** | **0.034** | **0.049** | 条件覆盖偏差 |
| 标准conformal ECE | 0.066 | 0.095 | 对照 |
| **ECE改善** | **48%** | **48%** | CQR vs 标准 |
| σ-\|err\| r | 0.345 (p=1e-137) | 0.355 (p=1e-145) | 不确定性有意义 |
| 物理层独立R² | 0.773 | 0.581 | 可解释锚点 |
| AD可信区R² | 0.945 | 0.873 | 应用域有效 |
| SR共识 | a_site_en 100% (8 seed) | — | 电负性主导 |
| LOEO外推 R² | 0.70 (68元素) | — | 纯ML最优 |
| DAF | 6.13 | — | top10%富集 |

## 五、创新点 (诚实定位: 方法学组合创新, 非理论突破)

**真创新 (有文献空白)**:
1. SR物理层嵌入残差+conformal框架 (ACS JCTC 2022只做独立SR)
2. SR适用边界判据 (SNR≥2/线性R²≥0.5, 形成能成功vs凸包能失败6重证据)
3. 符号集成共识 (a_site_en 100%, 把非唯一解翻转为优势)

**组合创新**:
4. conformal UQ + 多方法AD + LOEO + 候选验证 (钙钛矿ML罕见组合)

**诚实负面 (有价值)**:
5. PCRL物理约束残差 → 劣于基线, 降级为适用边界分析
6. RULSIF域适应 → 树模型下无效 (p=0.98)
7. Madelung物理分解 → 无结构信息做不实 (R²-0.0007)

## 六、关键代码文件 (src/)

**主线**:
- `pact_v2.py` — PACT主框架 (conformal+AD+统计)
- `pact_sr.py` — PACT-SR (SR物理层版, 仅形成能)
- `conformal.py` — split conformal prediction (修PICP)
- `ad_methods.py` — 应用域三方法对照
- `stats_eval.py` — 统计严谨 (CI/ECE/AUC-PR/Wilcoxon)

**特征+数据**:
- `features.py` — 特征工程 (★V5修复d/f电子bug)
- `utils.py` — PHYS_FEATURES(14), EXCLUDE_FROM_ML, load_features

**SR (符号回归)**:
- `symbolic_regression.py` — SR探索 (帕累托扫描)
- `sr_physics_layer.py` — SR vs KRR CV评估
- `sr_ensemble.py` — 符号集成 (10seed共识)
- `sr_debug.py` — SR深度debug

**优化**:
- `optuna_search_v2.py` — Optuna超参 (R²+0.005)
- `shap_selection.py` — SHAP特征选择
- `gbdt_stacking.py` — GBDT Stacking (R²+0.003)
- `domain_adapt.py` — RULSIF域适应 (负面)
- `madelung_decomposition.py` — Madelung物理分解 (负面)

**其他**:
- `cgcnn.py` — CGCNN (R²=0.799, 早停已修)
- `candidate_validation.py` — 候选三层验证
- `loeo_sr_full.py` — LOEO全元素外推
- `stacking.py` — 旧stacking (嵌套CV)
- `screening.py` — 材料筛选
- `pcrl.py` — PCRL (诚实降级为对照)

## 七、已修复的Bug (按重要性)

1. ★ **d/f电子计算bug** (V5): 内层轨道被全数, Ac得d=31应为1. 已修复(只数价层)
2. PICP=0.24灾难 → conformal修复 (0.83)
3. DFT特征泄露 (struct_) → 排除
4. NaN imputation泄露 → 折内Pipeline
5. PGML/PCRL基线不一致 → 统一KernelRidge
6. Stacking单CV泄露 → 嵌套CV
7. CGCNN早停监控训练损失 → 改验证损失

## 八、文档地图 (docs/)

- `troubleshooting_log.md` — ★完整V1-V6记录 (最权威, 先读这个)
- `v6_three_tasks_summary.md` — 最新三任务结果
- `v5_theory_algorithm_debug.md` — 理论评判+全量debug
- `optimization_final_v4.md` — 三项优化总结
- `deep_debug_synthesis.md` — 6角度深度debug解释
- `harsh_reviewer_assessment.md` — 严苛三区审稿评判
- `harsh_review_sr.md` — SR严苛审稿
- `sr_applicability_boundary.md` — SR适用边界分析
- `madelung_theory.md` + `madelung_honest_negative.md` — Madelung理论+负面
- `optim1_rulsif_honest.md` — RULSIF负面
- `audit_report_v2_1.md` — 审计报告
- `research_findings_v2.md` — 深度调研
- `v2_summary.md`, `optimization_research.md`, `optimization_final_v4.md` — 各阶段总结

## 九、诚实结论 (项目水平)

- **能否达三区**: 能, 且比初始稳。R²0.918+conformal+AD+SR共识+LOEO+诚实负面
- **能否冲二区**: 偏弱。R²低于SOTA(0.928, 不同数据集), 创新是组合非理论
- **理论创新**: 不现实 (Madelung失败证明无结构信息做不实)
- **下一步**: 论文写作 (框架已完整), 或补跑SR剩余4seed (可选)

## 十、运行环境

- Windows 10, Python 3.12.3 (venv: .venv)
- 包: lightgbm/xgboost/scikit-learn1.9/scipy/pymatgen/matminer/shap/gplearn/optuna
- 无Julia (故用gplearn非PySR)
- MP API不可达 (heartbeat无响应)
- GPU: RTX 4060 (仅CGCNN用)
- 备份: results_backup_v5

## 十一、关键教训 (写论文/答辩用)

1. 物理层价值是可解释性, 非精度 (纯ML反超)
2. Magpie已冗余物理特征 (electroneg_diff重建R²=0.998)
3. R²天花板由数据决定 (线性残差std=0.65eV)
4. SR适用需SNR≥2 (形成能2.61成功, 凸包能0.94失败)
5. 域适应对树模型无效 (文献28%改善在线性模型上)
6. 符号回归非唯一解, 但集成共识稳定 (a_site_en 100%)
