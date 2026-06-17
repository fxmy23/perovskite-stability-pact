# PACT-Final 统一管线结果 (四项必修完成)

**日期**: 2026-06-16
**论证**: 见 docs/four_fixes_justification.md (含文献佐证)
**管线**: src/pact_final.py (统一连贯, 审稿人跑一个脚本得所有指标)

---

## 终极统一管线架构

```
输入: 110维特征 (96 Magpie + 14物理)
   │
   ├── 物理层: KernelRidge on 14物理 → μ_p (可解释锚点, R²0.77/0.58)
   │     └─ 附录: SR符号集成共识 (a_site_en 100%, 可解释性证据)
   │
   ├── ML残差层: GBDT stacking (LGB+XGB+HistGBT + Ridge元)
   │     └─ 残差 r = y - μ_p → 点预测 μ = μ_p + μ_r (R²0.914/0.800)
   │
   ├── 不确定性 (解耦设计):
   │     ├─ 点预测: μ (stacking主线)
   │     ├─ 排序: ensemble σ (σ-err r=0.345/0.355)
   │     ├─ 区间: CQR分位数+conformal (PICP≥0.80, ECE改善48%)
   │     └─ 应用域: σ/kNN/leverage三方法
   │
   └── 输出: μ + [CQR区间] + AD标签
```

## 最终统一指标 (PACT-Final, 所有数字来自同一管线)

| 指标 | 形成能 | 凸包能 |
|---|---|---|
| 点预测 R² | **0.9144** [0.9074, 0.9215] | **0.7997** [0.7807, 0.8164] |
| MAE | 0.186 | 0.171 |
| CQR PICP | 0.802 (≥0.80✓) | 0.807 (≥0.80✓) |
| CQR ECE | **0.034** | **0.049** |
| 标准conformal ECE | 0.066 | 0.095 |
| **ECE改善** | **48%** | **48%** |
| σ-\|err\| r | 0.345 (p=1e-137) | 0.355 (p=1e-145) |
| 物理层独立R² | 0.773 (锚点) | 0.581 |
| AD可信区R² | 0.945 | 0.873 |

## 四项必修完成状态

### 项1: stacking作ML残差层 ✅
- 已集成进pact_final.py
- 点预测R²=0.9144 (物理层KRR+stacking残差)
- 文献佐证: Residual-Aware Stacking (SSRN 2025), 不绕过物理层
- post-debug: 残差stacking(0.9144) vs 纯stacking(0.918) 差0.004,
  在bootstrap CI内, 非显著回退。物理可解释性锚点值得此代价。

### 项2: 点/区间解耦 ✅
- 点预测=stacking主线, 区间=CQR分位数模型, 独立
- 文献佐证: Romano 2019 NeurIPS CQR (解耦是conformal标准)
- ECE改善48% (在统一管线内证实)

### 项3: 统一R²数字 ✅
- 所有论文指标用PACT-Final: 形成能0.9144, 凸包能0.7997
- CQR的点R²不再作主结果 (放方法说明)
- handoff.md同步更新

### 项4: SR公式表征 ✅ (方案确定)
- 主文: 报共识频率 (a_site_en 100%, b_site_group 63%)
- 附录: 选最短公式 (从sr_equations_per_fold按complexity排序)
- 不展示深嵌套原始公式
- 文献佐证: SR parsimony原则 (ResearchGate 2025)

## post-debug 验证

1. **数值稳定**: OOF预测无inf/nan (pact_final_oof_*.csv已保存)
2. **无泄露**: 嵌套CV (外5折评估, 内3折stacking OOF), 物理层KRR折内fit
3. **PICP保证**: CQR PICP=0.802/0.807 均≥0.80 ✓
4. **ECE改善**: 48% (统一管线内证实, 非独立实验拼凑)
5. **物理层**: R²=0.77/0.58 (合理锚点, 不为零不泄露)

## 关键文献 (论文引用)
1. Romano 2019 NeurIPS - CQR (解耦设计依据)
2. SSRN 2025 - Residual-Aware Stacking (残差stacking依据)
3. ScienceDirect 2026 - Hybrid Physics-ML (残差学习依据)
4. ResearchGate 2025 - SR challenges (SR简化报告依据)

## 统一管线的论文价值
- **审稿人跑一个pact_final.py** 得到所有论文指标 (复现性)
- 不再是3个独立实验(stack/conformal/AD)拼凑, 是1个连贯管线
- 每个组件有文献支撑: stacking(RAS), CQR(Romano), AD(标准), 物理层(混合ML)
