# 形成能物理分解理论 (准理论创新)

**日期**: 2026-06-16
**文献支撑**:
- [Revisiting perovskite Madelung potential (RSC Adv 2021)](https://pubs.rsc.org/en/content/articlehtml/2021/ra/d1ra01979a)
- [Generalized Kapustinskii for perovskites (Inorg Chem)](https://pubs.acs.org/doi/10.1021/ic00124a003)
- [Madelung constants for cubic crystals (PRB)](https://link.aps.org/doi/10.1103/PhysRevB.79.012102)
- [Solid-State Energetics: Madelung & Lattice (Inorg Chem)](https://pubs.acs.org/doi/10.1021/ic2023852)

---

## 一、理论基础

钙钛矿氧化物 ABO₃ 的形成能 E_f (相对单质) 的物理来源可分解:

```
E_f = E_Madelung (离子静电) + E_共价修正 + E_畸变(结构) + E_电子结构
       ↑可解析算           ↑ML学         ↑ML学        ↑ML学(部分Magpie)
```

### 1.1 Madelung 能量 (离子静电主导项)
理想立方 ABO₃ 的 Madelung 常数 M ≈ **1.716** (文献已知)。
静电(库仑)能量:
  U_Madelung = -M · (z_A · z_B · z_O) · e² / (4πε₀ · a)
  其中 a = 晶格常数 (≈ 2(r_A+r_O)/√2 或用平均离子半径估计)

**关键**: 这一项**完全可解析计算**, 只需离子电荷 + 离子半径。
对离子性强的钙钛矿 (如 BaTiO₃), Madelung能占形成能的 60-80%。

### 1.2 共价修正 (ML 学)
纯离子模型高估了离子性。实际 A-O, B-O 键有共价成分 (尤其过渡金属-氧)。
电负性差 χ_AB 是共价性的度量 → ML 用 χ_AB 等学修正。

### 1.3 畸变项 (ML 学)
实际钙钛矿非理想立方 (有八面体倾斜/旋转)。容差因子 t 偏离 1 越多, 畸变越大。
ML 用 t, μ_oct 学畸变对能量的影响。

### 1.4 电子结构 (ML 学, Magpie)
d/f 电子构型影响磁性和 Jahn-Teller 畸变。Magpie 统计 + 物理电子特征学此项。

## 二、我们的物理分解模型 (PACT-Madelung)

```
E_f = α · E_Madelung(t, μ_oct, r_A, r_B, z)  +  ML_residual(X_all)
       ↑物理锚点(可解析)                        ↑学共价+畸变+电子
       
物理层: E_Madelung 由 Born-Landé/Kapustinskii 公式算 (解析)
ML层: LightGBM 学残差 = E_f - α·E_Madelung
```

**与之前 PACT-SR 的区别**:
- PACT-SR: 物理层 = SR 发现的方程 (数据驱动, 无物理先验)
- PACT-Madelung: 物理层 = Madelung 物理公式 (物理先验, 可推导)

**理论创新性**: 这给项目一个**物理理论锚点**——不是纯黑盒ML,
也不是数据驱动的SR, 而是基于经典离子模型的可分解预测。
论文可写"物理可分解的形成能预测模型"。

## 三、实现方案

### Madelung 能量近似 (无需 DFT 结构)
对每个 ABO₃:
1. 离子电荷: z_A=+2, z_B=+4, z_O=-2 (典型钙钛矿价态)
2. Madelung 常数: M=1.716 (理想立方, 用近似值)
3. 特征长度: a ≈ (r_A + r_O) / √2 · 2 (用离子半径估晶格常数)
4. U_Madelung = -M · z_eff² · e² / (4πε₀ · a)
   简化 (单位换算后): U ∝ -M · z² / a

**注**: 这是近似 (实际结构非理想立方, M会变), 但作为"物理锚点"足够。
ML 残差会学习"偏离理想离子模型"的部分。

### 评估
- 物理层 (Madelung) 独立 R²: 预期 0.3-0.5 (类似 SR)
- 总 R² (Madelung + ML残差): 预期 ≈ 0.91 (ML补足)
- **创新性**: 物理层有明确物理推导, 非 SR 的数据拟合

## 四、与文献的关系 (诚实)
- [RSC Adv 2021](https://pubs.rsc.org/en/content/articlehtml/2021/ra/d1ra01979a) 算了 perovskite Madelung 势, 但未做 ML 残差分解
- [Inorg Chem generalized Kapustinskii](https://pubs.acs.org/doi/10.1021/ic00124a003) 做了格子能, 但未结合 conformal UQ
- **我们的组合 (Madelung物理分解 + ML残差 + conformal UQ + LOEO) 据调研无先例**

## 五、风险
- Madelung 近似 (理想立方 + 固定电荷) 可能太粗, 物理层 R² 可能 <0.3
- 若如此, 退化为"物理特征之一", 但仍有方法学价值 (物理可分解叙事)
