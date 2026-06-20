# Cover Letter — Computational Materials Science submission

**To:** The Editor-in-Chief, *Computational Materials Science*
**Date:** [submission date]
**Subject:** Submission of manuscript "Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance"
**Article type:** Full research article (single author)

---

Dear Editor,

I am pleased to submit my manuscript entitled "Uncertainty-Aware Prediction of Perovskite Oxide Stability with Conditional Conformal Intervals and Applicability-Domain Guidance" for consideration as a full research article in *Computational Materials Science*. I am a junior undergraduate student at Weiyang College, Tsinghua University, and this work was conducted as an independent research project using open data and a personal workstation.

**The gap this work addresses.** Machine learning has become an indispensable tool for accelerating the discovery of ABO3 perovskite oxides, with recent descriptor-based models reaching R2 > 0.91 for formation-energy prediction. However, the vast majority of published models report only point predictions and aggregate accuracy metrics, without quantifying *when* their predictions can be trusted — a critical gap for high-throughput screening pipelines where false-positive stability predictions waste expensive DFT verification budget. This manuscript addresses that gap directly by integrating point prediction, calibrated uncertainty, applicability-domain guidance, and extrapolation assessment into a single reproducible framework.

**What this work contributes to computational materials science:**

1. **A reliability-aware prediction framework, not just an accuracy benchmark.** I integrate a physics-informed point predictor (KernelRidge baseline + GBDT stacking residual) with Conformalized Quantile Regression (CQR), which provides sample-adaptive prediction intervals satisfying a finite-sample, distribution-free coverage guarantee. Under a fair same-family baseline, CQR reduces the Expected Calibration Error by 43–60%, yielding intervals that widen appropriately for extrapolation samples and tighten for interpolation samples.

2. **An applicability domain for perovskite stability models.** Three complementary applicability-domain criteria — ensemble σ, k-NN distance, and PCA leverage — are compared, and the trusted region (R2 = 0.945 for formation energy) is clearly separated from the untrusted region (R2 = 0.886), giving experimentalists a concrete boundary for reliable use.

3. **Element-level extrapolation mapping.** A leave-one-element-out evaluation over all 73 A/B-site elements quantifies where the model generalizes and where it fails (mean per-element R2 = 0.739, no negative-R2 elements), directly informing screening of unseen chemistries.

4. **Interpretability grounded in materials chemistry.** A symbolic-regression ensemble (50 independent equations) recovers the known dominance of A-site electronegativity in formation energy, and an empirical applicability boundary for symbolic regression is derived by contrasting formation energy (SNR = 3.40, SR succeeds) with hull energy (SNR = 1.39, SR fails).

**Fit with the journal's scope.** I believe this work fits the scope of *Computational Materials Science* — advancing computational methods applied to materials prediction — and would be of interest to readers working on ML-accelerated materials discovery, uncertainty quantification, and perovskite design. I emphasize that the value of this work lies in the uncertainty, applicability-domain, and extrapolation components, which make high-throughput screening more trustworthy, rather than in chasing the highest R2.

**Reproducibility and transparency.** All source code and processed data are provided under the MIT license at https://github.com/fxmy23/perovskite-stability-pact, with the main results reproducible via a single command (python src/pact_final.py). The limitations of the framework — single data source, decoupling tension between point and interval predictions, absence of DFT-validated candidates — are stated explicitly in the manuscript, which I believe strengthens rather than weakens the contribution.

**Declarations.** This manuscript has not been published previously, is not under consideration for publication elsewhere, and all its data and results are original to this work. I declare no conflicts of interest. The use of AI-assisted tools during manuscript preparation is declared in the manuscript, and I take full responsibility for its content.

Thank you for your consideration. I look forward to your response.

Sincerely,

**Xumingyong Feng**
Weiyang College, Tsinghua University, Beijing 100084, China
Email: fxmy23@mails.tsinghua.edu.cn

---

## Notes (not part of the letter — for submission system fields)

- **Suggested reviewers** (entered separately in Editorial Manager):
  1. An expert in machine learning for perovskite stability prediction.
  2. An expert in conformal prediction or uncertainty quantification.
  3. An expert in applicability-domain methods or cheminformatics.
- **Funding:** None declared (entered in the submission system, not in the letter).
