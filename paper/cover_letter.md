# Cover Letter (Computational Materials Science submission)

**To:** The Editor, *Computational Materials Science*
**Date:** [submission date]
**Subject:** Submission of manuscript "Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance"

---

Dear Editor,

I am pleased to submit my manuscript entitled "Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance" for consideration in *Computational Materials Science*. I am a junior undergraduate student at Weiyang College, Tsinghua University, and this work was conducted as an independent research project using open data and a personal workstation.

Machine learning has become an indispensable tool for accelerating the discovery of ABO鈧?perovskite oxides, with recent models achieving R虏 > 0.92 for formation-energy prediction. However, the vast majority of published models report only point predictions and aggregate accuracy metrics, without quantifying *when* their predictions can be trusted鈥攁 critical gap for high-throughput screening pipelines where false-positive stability predictions waste expensive DFT verification budget. This manuscript addresses that gap directly.

**What this work contributes to computational materials science:**

1. **A reliability-aware prediction framework, not just an accuracy benchmark.** I integrate a physics-informed point predictor with Conformalized Quantile Regression (CQR), which provides sample-adaptive prediction intervals satisfying a finite-sample, distribution-free coverage guarantee. CQR reduces the expected calibration error by 43鈥?0% relative to standard split conformal under a fair baseline, yielding intervals that widen appropriately for extrapolation samples and tighten for interpolation samples.

2. **An applicability domain for perovskite stability models.** Three applicability-domain criteria (ensemble 蟽, k-NN distance, PCA leverage) are compared, and the trusted region (R虏 = 0.945) is clearly separated from the untrusted region (R虏 = 0.886), giving experimentalists a concrete boundary for reliable use.

3. **Element-level extrapolation mapping.** Leave-one-element-out evaluation over 68 A-site elements quantifies where the model generalizes and where it fails鈥攊nformation absent from conventional cross-validation metrics and directly relevant to screening unseen chemistries.

4. **Interpretability grounded in materials chemistry.** Symbolic regression consensus (50 independent equations) recovers the known dominance of A-site electronegativity in formation energy, and an empirical applicability boundary for symbolic regression is derived by contrasting formation energy (success) with hull energy (failure).

I emphasize that this work is positioned as a *materials-informatics methodology* contribution: the value lies in the uncertainty quantification, applicability domain, and extrapolation assessment components, which make high-throughput screening more trustworthy鈥攏ot in chasing the highest R虏. The framework's limitations (single data source, decoupling tension between point and interval predictions, absence of DFT-validated candidates) are stated explicitly, which I believe strengthens rather than weakens the contribution. To support reproducibility and FAIR data principles, all source code and processed data are provided (GitHub + Zenodo, links to be inserted upon acceptance).

This manuscript has not been published elsewhere and is not under consideration by another journal. I confirm the use of AI-assisted tools is declared in the manuscript.

I believe this work fits the scope of *Computational Materials Science*鈥攁dvancing computational methods applied to materials prediction鈥攁nd would be of interest to readers working on ML-accelerated materials discovery, uncertainty quantification, and perovskite design.

Thank you for your consideration.

Sincerely,

Xumingyong Feng
Weiyang College, Tsinghua University, Beijing 100084, China
fxmy23@mails.tsinghua.edu.cn

---

## Suggested reviewers (optional)

1. An expert in machine learning for perovskite stability prediction.
2. An expert in conformal prediction or uncertainty quantification in materials science.
3. An expert in applicability-domain methods or cheminformatics.

## Declared conflicts

None.

