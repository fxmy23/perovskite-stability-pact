# Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance

**Author:** Xumingyong Feng
**Affiliation:** Weiyang College, Tsinghua University, Beijing 100084, China
**Corresponding author:** Xumingyong Feng (fxmy23@mails.tsinghua.edu.cn)
**Target journal:** Computational Materials Science (Elsevier)

---

## Abstract

Machine learning (ML) accelerates the discovery of ABO₃ perovskite oxides, yet most existing models provide only point predictions without quantifying *when* they can be trusted—a critical gap for high-throughput screening where false positives waste costly DFT verification. We present an uncertainty-aware prediction framework for perovskite formation energy and thermodynamic stability (energy above hull) that combines (i) a physics-informed point predictor (KernelRidge physics baseline + gradient-boosted-decision-tree stacking residual), (ii) Conformalized Quantile Regression (CQR) providing sample-adaptive prediction intervals with finite-sample coverage guarantees, (iii) a multi-method applicability domain (ensemble σ, k-NN distance, leverage) delimiting the trusted region, and (iv) symbolic regression consensus validating known chemistry (A-site electronegativity appears in 100% of discovered equations). On 4,914 perovskites, the framework achieves R² = 0.914 (formation energy) and 0.800 (hull energy), with CQR reducing the expected calibration error by 43–60% relative to standard split conformal under fair comparison. Leave-one-element-out extrapolation over 68 elements and a three-tier candidate validation (database cross-reference, Goldschmidt synthesizability, robustness) quantify the model's reliability boundary. We discuss the decoupling tension between point and interval predictions, the empirical applicability boundary of symbolic regression (formation energy succeeds at SNR = 2.61; hull energy fails at SNR = 0.94), and the role of the physics baseline as an interpretability anchor rather than an accuracy contributor.

**Keywords:** perovskite oxides; formation energy; conformal prediction; applicability domain; symbolic regression; materials informatics

---

## 1. Introduction

ABO₃ perovskite oxides constitute one of the most versatile families of functional materials, exhibiting catalytic activity [ref], ferroelectricity [ref], superconductivity [ref], and ionic conductivity [ref] that span energy conversion, information storage, and chemical synthesis applications. The chemical space of possible A-site and B-site elemental combinations is vast (dozens of candidate elements at each site), yet experimentally realizing and verifying each candidate via density functional theory (DFT) or synthesis is prohibitively expensive—each DFT relaxation consumes hours to days of computation. Machine learning (ML) has emerged as a powerful accelerator: descriptor-based regressors trained on existing DFT databases can predict formation energy and thermodynamic stability (energy above the convex hull, E_hull) in milliseconds, enabling high-throughput pre-screening before DFT verification [Emery 2018; Saad 2021 npj review].

Recent advances have pushed prediction accuracy substantially. [MDPI Materials 2025] achieved R² = 0.928 for ABX₃ formation energy using SHAP-guided feature selection and meta-heuristic hyperparameter tuning; [RSC Adv 2024] demonstrated interpretable ML screening of perovskite oxides with stability classification and regression; [J. Phys. Chem. C 2024] compared descriptor ML against crystal-graph neural networks. These works, however, share a common limitation: **they report point predictions and aggregate metrics (R², MAE) but do not quantify the uncertainty of individual predictions or delineate the region where the model can be trusted.** For materials discovery, this is not a minor omission. A model that predicts E_hull = 0.02 eV/atom for an unseen elemental combination may be highly reliable (if the combination resembles the training distribution) or grossly erroneous (if it is an extrapolation to a chemistry the model has never seen). Consider a screening campaign over 1,000 virtual A–B combinations: if the model has no uncertainty estimate, all candidates predicted stable (E_hull < 0.05 eV/atom) must be forwarded to DFT verification at uniform cost. With a calibrated per-sample interval, however, the ~30% of candidates whose 80% interval straddles the stability threshold can be flagged as ambiguous and deprioritized, reserving DFT resources for high-confidence predictions. Without an applicability domain, the model offers no warning that a candidate involving, e.g., an actinide A-site—poorly represented in the training data—is far less reliable than one involving a well-sampled alkaline-earth cation, even if both receive the same point estimate. Applicability-domain (AD) methods, standard in cheminformatics QSAR [Netzeva 2005; Sahigara 2012], address this by delimiting the chemical space where predictions are trustworthy, yet they remain rarely applied to inorganic perovskite stability models.

Conformal prediction [Vovk 2005; Shafer & Vovk 2008] offers a principled remedy: it wraps any point predictor with prediction intervals that satisfy a finite-sample, distribution-free marginal coverage guarantee (P(y ∈ C(X)) ≥ 1 − α). Conformalized Quantile Regression (CQR) [Romano et al. 2019 NeurIPS] further yields *heteroscedastic* intervals whose width adapts to sample-level uncertainty—wide for extrapolation samples, narrow for interpolation—while preserving the coverage guarantee. Despite their maturity in statistics and growing adoption in clinical/ML venues [ICML 2025; ICLR 2025], conditional conformal methods have seen limited application in materials informatics, where ensemble variance and heuristic confidence scores remain the norm. Similarly, applicability-domain methods (leverage/Williams plots, k-NN distance) are standard in cheminformatics QSAR [Netzeva 2005] but rarely applied to inorganic perovskite stability models.

Symbolic regression (SR) offers a complementary, interpretability-focused tool: it searches the space of mathematical expressions to discover closed-form equations relating descriptors to targets [Cranmer 2023 PySR; Ouyang 2018 SISSO]. While SR equations are non-unique and require consensus analysis across runs, they provide a human-readable counterpoint to black-box ML feature attributions and can validate whether the model has captured physically meaningful structure–property relationships.

In this work, we integrate these four components into a unified framework—**uncertainty-aware perovskite stability prediction with conditional conformal intervals and applicability-domain guidance**—and evaluate it on 4,914 ABO₃ perovskites from the wolverton_oxides dataset. Our contributions are:

1. **Conditional coverage via CQR.** We apply CQR (Romano et al., 2019) to perovskite stability prediction and demonstrate, under a fair same-family baseline, a 43–60% reduction in expected calibration error (ECE) relative to standard split conformal—yielding sample-adaptive intervals whose conditional coverage more closely matches the nominal level.
2. **Multi-method applicability domain.** We compare three AD criteria (ensemble σ, k-NN distance, PCA leverage) and show the trusted region achieves R² = 0.945 vs. 0.886 in the untrusted region, providing a concrete reliability boundary for downstream screening.
3. **Extrapolation assessment via LOEO.** We perform leave-one-element-out (68 elements) and quantify per-element extrapolation R², exposing where the model generalizes and where it fails—information absent from aggregate CV metrics.
4. **Symbolic regression consensus as a chemistry cross-check.** Across 50 independent SR equations, A-site electronegativity appears with 100% consensus, validating the known dominance of electronegativity in formation energy. We further derive an *empirical* SR applicability criterion (SNR ≥ 2) by comparing formation energy (success) and hull energy (failure), and discuss its boundary frankly.

We are explicit about the framework's limitations: single data source, the decoupling tension between the stacking point predictor and the CQR interval, the absence of DFT-validated candidates, and the empirical (non-universal) nature of the SR criterion. By foregrounding these, we aim to provide a reproducible, honestly-bounded tool for uncertainty-aware perovskite screening rather than an over-claimed accuracy benchmark.

[**Figure 1: Overview of the uncertainty-aware prediction framework.** Input features → physics-informed point predictor (KernelRidge baseline + GBDT stacking residual) → conditional conformal interval (CQR) + applicability domain (σ/k-NN/leverage) → point prediction, adaptive interval, and trust label.]

---

## 2. Methods

### 2.1 Dataset

We use the **wolverton_oxides** dataset (4,914 cubic/tilted ABO₃ perovskites), distributed via the matminer library [Ward 2018]. Each sample contains the composition, the DFT-computed formation energy per atom (E_f, range −3.8 to +1.9 eV/atom) and the energy above the convex hull (E_hull, range 0 to +3.9 eV/atom, non-negative by definition). All 4,914 formulas are unique. The dataset originates from a single high-throughput DFT study; this single-source limitation and its implications for cross-dataset generalization are discussed in §5.

### 2.2 Feature engineering

Each composition is described by 113 features combining:
- **96 Magpie statistical descriptors** [Ward 2016] computed via a pure-pymatgen reimplementation (elemental properties—atomic number, electronegativity, atomic radius, row, group, valence, etc.—aggregated by composition-weighted mean, avg-dev, mode, minimum, maximum).
- **14 physics-informed descriptors** with explicit perovskite meaning: Goldschmidt tolerance factor *t* = (r_A + r_O)/(√2(r_B + r_O)), octahedral factor μ = r_B/r_O, A/B electronegativity difference, A/B ionic radius ratio, B-site valence, A/B ionic radii, A/B electronegativities, B-site row/group, and A/B-site d-/f-electron counts. Ionic radii use Shannon 12-/6-coordination tables with pymatgen fallback.

Three features were excluded from the ML input after audit: two derived heuristics (phys_in_stable_zone, phys_stability_score, composites of the same physical inputs that would create feature redundancy) and one dead feature (phys_b_site_unpaired, 100% NaN due to missing elemental data). The final ML input dimensionality is **110**. The d/f-electron counts were corrected to count only the valence-shell (maximum principal quantum number) orbitals, as full-electronic-structure summation erroneously included inner-shell d electrons (e.g., Ac reported d = 31 instead of 1).

### 2.3 Physics-informed point predictor

The point prediction adopts a two-stage residual architecture [Residual-Aware Stacking; SSRN 2025]:
1. **Physics baseline** μ_p: KernelRidge (RBF kernel, α = 1.0, γ = 0.1) on the 14 physics features, providing an interpretable baseline (standalone R² ≈ 0.77 for formation energy, 0.58 for hull energy).
2. **ML residual** μ_r: a stacking ensemble of three gradient-boosted-decision-tree (GBDT) base learners—LightGBM (leaf-wise), XGBoost (level-wise), and scikit-learn HistGradientBoosting (histogram-based)—combined by a Ridge meta-learner, trained on the residual (y − μ_p) using the full 110 features.

The point prediction is μ = μ_p + μ_r. The physics baseline serves as an *interpretability anchor* (its standalone R² quantifies the variance captured by physically meaningful descriptors); it does not, by itself, improve total accuracy over pure ML, because the Magpie descriptors already redundantly encode much of the physical information (electronegativity difference is reconstructible from Magpie features with R² = 0.998). We report this honestly rather than claiming the physics layer boosts performance.

Hyperparameters for the GBDT base learners were selected by Optuna (TPE sampler, 25 trials, 3-fold inner CV). We note that hyperparameter selection on the full dataset incurs a small optimistic bias: nested CV estimates ~0.005 R² inflation, within the bootstrap confidence interval reported in §3.

### 2.4 Conditional conformal prediction (CQR)

Uncertainty quantification uses **Conformalized Quantile Regression** (CQR; Romano et al., 2019), which combines the adaptivity of quantile regression with the finite-sample coverage guarantee of split conformal. The procedure, applied within each outer CV fold:

1. Split the training fold into a proper-training set (80%) and a calibration set (20%).
2. Fit two LightGBM quantile regressors on the proper-training set: q̂_0.10(X) and q̂_0.90(X).
3. Compute non-conformity scores on the calibration set: s_i = max(q̂_0.10(x_i) − y_i, y_i − q̂_0.90(x_i)).
4. Take the ⌈(1−α)(n_cal+1)⌉-th quantile d of {s_i} (the rank-based finite-sample conformal quantile).
5. Prediction interval for test samples: C(X) = [q̂_0.10(X) − d, q̂_0.90(X) + d].

**Theoretical guarantee.** Under exchangeability of (calibration, test), the *marginal* coverage satisfies P(y ∈ C(X)) ≥ 1 − α for any α, with no distributional assumptions (Vovk et al., 2005). CQR additionally yields *heteroscedastic* intervals: the quantile regressors produce sample-specific widths, so high-uncertainty (extrapolation) samples receive wider intervals than low-uncertainty (interpolation) samples.

**Decoupling design.** The CQR interval is calibrated independently of the stacking point predictor (§2.3), following the conformal prediction philosophy that the interval need not derive from the same model as the point estimate (Romano et al., 2019; Romano, Patterson & Candès, 2019). Consequently, the point prediction may occasionally fall outside its 80% CQR interval (~9% of samples); this is consistent with the marginal guarantee (which concerns y, not the point estimate) but introduces a practical tension discussed in §5.

**Fair baseline.** Standard split conformal (uniform ±d intervals around a single point predictor) is reported as a baseline. To ensure a fair comparison of conditional coverage, the baseline point predictor uses the same LightGBM family (not a weaker Ridge), so the ECE improvement attributable to CQR is not inflated by baseline weakness.

**Expected Calibration Error (ECE).** We quantify conditional coverage via ECE: sort samples by ensemble σ, partition into 10 equal bins, and compute the weighted mean absolute deviation of each bin's empirical coverage from the nominal 1 − α. Lower ECE indicates more uniform (better-calibrated) conditional coverage.

### 2.5 Applicability domain

Three AD criteria delimit the trusted region, computed within each CV fold (no leakage):
- **Ensemble σ** (model uncertainty): the standard deviation of the three GBDT base learners' residual predictions. Trusted if σ < median.
- **k-NN distance** (input-space extrapolation): the mean Euclidean distance to the 5 nearest training neighbors in PCA(20)-reduced space. Trusted if below the 95th percentile of training-set distances.
- **PCA leverage** (Williams plot): h_i = x_iᵀ(XᵀX)⁻¹x_i in PCA(20) space. Trusted if h_i ≤ 3p/n.

We report the trusted/untrusted R² partition for each method and their agreement.

### 2.6 Evaluation protocol

- **5-fold cross-validation** (shuffle, seed 42) for all reported OOF metrics.
- **Stacking meta-learner** trained on inner 3-fold OOF to avoid leakage.
- **Bootstrap 95% CI** (1000 resamples) for R² and MAE.
- **Imputation and standardization** fit within each training fold (Pipeline), never on the full dataset.
- **σ–error correlation** (Pearson r, with p-value) to validate that ensemble uncertainty ranks errors correctly.

### 2.7 Symbolic regression (interpretability cross-check)

Symbolic regression is employed not to discover new physics but to provide a human-readable cross-check of the ML model. We run gplearn (genetic programming, basic operators + sqrt, parsimony 0.001, 40 generations, 2000 population) with 10 random seeds × 5 CV folds = 50 independent equations on the 14 physics features. Feature **consensus frequency** (fraction of equations in which each physics feature appears) is reported, rather than any single non-unique equation. We additionally derive an *empirical* SR applicability criterion by comparing SR success on formation energy (signal-to-noise ratio SNR = 2.61) vs. failure on hull energy (SNR = 0.94), where SNR = R²(KRR)/(1 − R²(KRR)).

---

## 3. Results and Discussion

### 3.1 Point prediction performance

[**Table 1**: Model comparison with literature benchmark row — R²/MAE/RMSE for formation energy and hull energy; rows: pure LightGBM, GBDT stacking, PACT-Final (KRR + stacking), and literature reference (MDPI 2025 R² = 0.928 on a different dataset, noted).]

[**Figure 2**: Parity plots — predicted vs. DFT for (a) formation energy, (b) hull energy, colored by ensemble σ, with y = x reference line, R² and MAE annotated.]

The framework achieves **R² = 0.914 [0.907, 0.922]** for formation energy and **R² = 0.800 [0.781, 0.816]** for hull energy (5-fold CV, bootstrap 95% CI), with MAE of 0.186 and 0.171 eV/atom respectively (Table 1). The formation-energy R² is comparable to recent descriptor-ML benchmarks (0.91–0.93) though slightly below the 0.928 reported by [MDPI 2025] on a distinct Materials-Project-derived dataset; direct comparison is complicated by differing reference frames (we verified Pearson r = 0.358 between our dataset and matbench_perovskites, precluding numerical cross-validation). Hull-energy prediction is inherently harder (R² = 0.800): E_hull depends on all competing phases, a global property not fully captured by single-compound descriptors. The stacking ensemble (LightGBM + XGBoost + HistGradientBoosting) marginally improves over single LightGBM (+0.003 R²), consistent with the high correlation (0.99) among GBDT family members limiting ensemble diversity gains.

**Material interpretation.** The physics baseline alone (KernelRidge on 14 physics features) explains R² = 0.773 of formation-energy variance—confirming that physically meaningful descriptors (tolerance factor, electronegativity, ionic radii) capture the majority of the signal, consistent with the ionic-model understanding of perovskite stability. The stacking residual adds the non-linear/secondary-descriptor structure.

**Error analysis.** The 8.0% of samples with absolute error exceeding 0.5 eV/atom (394/4914) cluster at the extreme of the formation-energy distribution: their mean E_f = −1.384 eV/atom is less negative than the global mean (−1.663), indicating the model struggles most with compounds of intermediate stability—neither strongly stable nor clearly unstable—where small descriptor changes produce large energy shifts. Conversely, the most accurately predicted compounds are strongly stable perovskites (E_f < −2.5 eV/atom, e.g., alkaline-earth-based oxides) whose formation energy is dominated by well-captured ionic contributions. Notably, 4 of the 5 worst predictions occur at σ < 0.10 eV/atom, suggesting the ensemble underestimates uncertainty for certain extrapolative samples; this residual miscalibration motivates the conditional (CQR) intervals in §3.2, which do not rely on σ alone.

### 3.2 Uncertainty quantification: CQR conditional coverage

[**Table 2**: CQR vs. standard split conformal — PICP, MPIW, ECE, for both targets, fair (same-family) baseline.]

[**Figure 3**: Reliability diagrams — empirical coverage per σ-decile for standard conformal vs. CQR, with nominal 0.80 reference.]

CQR delivers intervals satisfying the marginal coverage guarantee (PICP = 0.802 / 0.807, both ≥ 0.80 nominal) while **reducing ECE by 43–60%** relative to standard split conformal under a fair (same LightGBM-family) baseline (Table 2): formation-energy ECE drops from 0.085 to 0.034 (−60%), hull-energy from 0.086 to 0.049 (−43%). This means conditional coverage is markedly more uniform—high-uncertainty (extrapolation) samples no longer suffer severe under-coverage, and low-uncertainty (interpolation) samples receive appropriately tight intervals.

The heteroscedasticity is empirically visible: the mean interval width for the highest-σ tercile (1.03 eV/atom for formation energy) is ~1.5× that of the lowest-σ tercile (0.68), whereas standard conformal assigns uniform width (0.66 to both). For high-throughput screening, this sample-adaptive behavior directly translates to fewer false-positive stable predictions among uncertain candidates.

**Material interpretation.** The σ–error Pearson correlation (r = 0.345 for formation energy, p = 1.3 × 10⁻¹³⁷) confirms that ensemble disagreement tracks actual error: when the GBDT models disagree, the prediction is genuinely less reliable. This physical meaningfulness of the uncertainty estimate is a prerequisite for AD-based screening (§3.3).

### 3.3 Applicability domain

The ensemble-σ AD criterion partitions samples into a trusted region (σ < median) achieving **R² = 0.945** (formation energy) versus **R² = 0.886** in the untrusted region—a 0.059 R² gap that validates the criterion. The k-NN distance and PCA leverage criteria yield trusted-region R² of 0.915 and 0.916 respectively, with 50–53% agreement with the σ criterion. The three methods provide complementary, convergent evidence for the model's reliability boundary rather than a single arbitrary threshold.

[**Figure 4**: Applicability-domain visualization — predicted vs. error scatter, colored by trusted (σ < median) / untrusted, illustrating the tighter error distribution in the trusted region.]

**Material interpretation.** For experimentalists using the model to prioritize DFT verification, restricting to the trusted region roughly halves the error variance, directly reducing wasted computation on unreliable predictions.

### 3.4 Extrapolation assessment (LOEO)

Leave-one-element-out evaluation over 68 A-site elements (each held out in turn; minimum 50 samples per element) yields a mean per-element R² of **0.70** for formation energy, with substantial heterogeneity: Ho, Pr, Nd extrapolate well (R² > 0.85) whereas B, Ac, and several p-block elements extrapolate poorly (R² < 0 or near zero). Pure ML slightly outperforms the physics-informed variant on LOEO, indicating the physics baseline does not improve extrapolation in this regime.

[**Figure 5** / Table: LOEO per-element R² for representative elements, annotated with chemical interpretation.]

**Material interpretation.** The extrapolation pattern is chemically coherent. The best-extrapolated A-site elements are large, highly electropositive cations—alkali metals (Rb R²=0.96, K 0.93, Na 0.94, Cs 0.92), alkaline earths (Ba 0.94), and lanthanides (Pr 0.95, La 0.91, Gd 0.92, Ho 0.93, Dy 0.94). These elements share consistent +2 or +3 oxidation states, large ionic radii, and well-defined coordination chemistry, so the descriptor space around them is densely sampled and their A-site contributions to formation energy are nearly linear in ionic radius and electronegativity. In contrast, the worst-extrapolated elements occupy sparsely populated or chemically anomalous regions of the descriptor space: B (R²=−0.88) and Os (−0.89) have very few isochemical neighbors in the training set; Ac (0.33) and Pa (0.32) are actinides with complex f-electron chemistry underrepresented in the dataset; Al (0.33), Be (0.48), and Te (0.30) form atypical A-site cations whose bonding character (covalent/metallic rather than ionic) deviates from the perovskite norm. This element-specific failure map directly informs screening: predictions involving underrepresented actinides or atypical main-group A-site cations should be treated as unreliable regardless of the point estimate, whereas screening among alkali/alkaline-earth/lanthanide chemistries is expected to transfer well.

### 3.5 Symbolic regression consensus and applicability boundary

Across 50 independent SR equations (10 seeds × 5 folds), **A-site electronegativity appears in 100% of equations**, with B-site electronegativity (63%) and B-site group number (63%) the next most conserved. This consensus validates the known dominance of electronegativity in governing perovskite formation energy (consistent with ionic/covalent bonding theory) and provides an interpretable cross-check that the ML model has captured physically meaningful structure.

The SR applicability is target-dependent: SR succeeds on formation energy (standalone R² = 0.43, SNR = 2.61) but **fails on hull energy** (R² ≈ 0.19, SNR = 0.94, with the SR solution collapsing to a constant in several folds). We trace this to the differing physical nature of the two targets—formation energy is dominated by single-compound bond chemistry (near-linear, well-expressed by basic operators), whereas hull energy depends on competition with all other phases (strongly non-linear, non-monotonic in geometric descriptors). This yields an *empirical* SR applicability criterion (SNR ≥ 2, linear-R² ≥ 0.5), which we caveat as based on only two targets and not yet universal.

[**Figure 6**: SHAP feature importance vs. SR consensus frequency — illustrating the complementary perspectives of the black-box ML (Magpie-statistical-dominated) and symbolic regression (physics-descriptor-dominated).]

### 3.6 Candidate materials screening

Applying the framework to virtual A–B combinations yields 9 candidate stable perovskites (predicted E_hull < 0.05 eV/atom within the trusted AD region). Three-tier validation provides indirect evidence: 3 candidates (LaTiO₃, YCoO₃, LaAlO₃) appear in the independent matbench_perovskites DFT database (confirming they have been studied); 8/9 fall within the Goldschmidt synthesizability zone (t ∈ [0.8, 1.05], μ ∈ [0.4, 0.9]); and leave-element-out robustness testing confirms the predictions are not artifacts of single-element memorization for the moderate-grade candidates. We emphasize that database presence confirms study, not numerical accuracy of our predictions (due to the reference-frame mismatch noted in §2.1), and that DFT verification remains necessary.

[**Table 3**: Candidate perovskites with predicted E_hull, ensemble σ, validation tier, and Goldschmidt factors.]

---

## 4. Limitations

This work has several limitations that should be acknowledged:

1. **Single data source.** All 4,914 samples originate from one DFT study (wolverton_oxides). Cross-dataset generalization could not be numerically verified: the Materials Project API was inaccessible, and matbench_perovskites uses a different formation-energy reference frame (Pearson r = 0.358 between sources). LOEO extrapolation partially addresses generalization but does not substitute for independent DFT validation.

2. **Point–interval decoupling tension.** The point prediction (stacking) and CQR interval are based on independent models, per conformal prediction philosophy. Consequently, the point prediction falls outside the 80% CQR interval for ~9% of samples—consistent with the marginal guarantee (which concerns the true value, not the point estimate) but introducing a practical tension that users should resolve by treating the interval as the primary uncertainty statement.

3. **Candidate validation depth.** The 9 predicted candidates lack DFT or experimental verification; the three-tier validation provides indirect evidence only.

4. **Empirical SR applicability criterion.** The SNR-based criterion derives from two targets only and may not generalize to other material properties.

5. **Limited extrapolation.** LOEO R² averages 0.70 with several negative-R² elements; users should restrict predictions to the applicability domain.

6. **Hyperparameter optimistic bias.** Fixed Optuna hyperparameters incur ~0.005 R² optimistic bias (nested CV estimate), within the reported bootstrap CI.

---

## 5. Conclusions

We presented an uncertainty-aware framework for predicting ABO₃ perovskite formation energy and thermodynamic stability that integrates a physics-informed point predictor, conditional conformal (CQR) intervals with finite-sample coverage guarantees, a multi-method applicability domain, and symbolic-regression-based interpretability cross-checks. The framework achieves R² = 0.914 / 0.800 for the two targets while delivering sample-adaptive intervals whose conditional coverage improves 43–60% over standard conformal, an applicability domain that separates a trusted region (R² = 0.945) from an untrusted region (R² = 0.886), and an element-level extrapolation map via 68-element LOEO. Symbolic regression consensus validates the known electronegativity dominance and exposes an empirical applicability boundary (SNR ≥ 2) that we frame honestly as target-specific.

The central message is methodological: for materials-discovery pipelines, point accuracy alone is insufficient—per-sample uncertainty, applicability-domain guidance, and extrapolation assessment are necessary components of a trustworthy predictor. The framework is fully reproducible (single pipeline, open code) and its limitations are documented to support informed downstream use. Future work should validate candidates via DFT, test cross-dataset generalization with accessible MP data, and refine the SR applicability criterion across additional material properties.

---

## Data Availability

The wolverton_oxides dataset is publicly available via the matminer Python library (`matminer.datasets.load_dataset("wolverton_oxides")`). Processed feature matrices, per-sample out-of-fold predictions, and all experimental result tables are provided in the supplementary data and in the project repository. A permanent archived snapshot of the code and data has been deposited on Zenodo: [DOI to be inserted upon acceptance].

## Code Availability

All source code (feature engineering, PACT-Final unified pipeline, Conformalized Quantile Regression, applicability-domain methods, symbolic regression ensemble, evaluation, and figure generation) is available at [GitHub repository URL to be inserted upon acceptance] under the MIT license. The main results can be reproduced with a single command: `python src/pact_final.py`.

## CRediT Author Contributions

**Xumingyong Feng**: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data Curation, Writing—Original Draft, Writing—Review & Editing, Visualization. (Single-author work; all roles contributed by the author.)

## Declaration of Competing Interests

The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors. Computational work was carried out on a personal workstation equipped with an NVIDIA RTX 4060 GPU.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used an AI-assisted coding and writing tool (ZCode, powered by a large language model) in order to assist with software development, debugging, data analysis scripting, and manuscript drafting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

## Acknowledgments

The author thanks the developers of the open-source tools that made this work possible, including pymatgen, matminer, scikit-learn, LightGBM, XGBoost, Optuna, gplearn, and SHAP. The wolverton_oxides dataset and the matbench_perovskites benchmark are gratefully acknowledged. The author is a junior undergraduate student at Weiyang College, Tsinghua University, and this work was conducted as an independent research project.

---

## References (to be finalized, ~40 entries)

1. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
2. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized quantile regression. *NeurIPS*, 32.
3. Ward, L. et al. (2016). A general-purpose machine learning framework for predicting properties of inorganic materials. *npj Comput. Mater.*, 2, 16028.
4. Ward, L. et al. (2018). Matminer: An open source toolkit for materials data mining. *Comput. Mater. Sci.*, 152, 60–69.
5. Emery, A. A. & Wolverton, C. (2018). Machine learning for perovskite stability prediction. *Sci. Data*, 5, 180154.
6. [MDPI Materials 2025] Prediction of ABX₃ perovskite formation energy using machine learning.
7. [RSC Adv 2024] Interpretable ML-assisted screening of perovskite oxides.
8. Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. *arXiv:2305.01582*.
9. [Residual-Aware Stacking, SSRN 2025]
10. [ICML 2025 rectifying conformity scores]
... (full list to be compiled)
