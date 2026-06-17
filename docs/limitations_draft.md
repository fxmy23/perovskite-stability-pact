# 论文 Limitations 草稿 (P1-4, 主动暴露)

**原则**: 诚实承认局限比被审稿人发现强。这些是项目的真实边界。

---

## Limitations (论文小节草稿)

This work has several limitations that should be acknowledged:

**1. Single data source.** All 4914 samples originate from a single high-throughput DFT
study (wolverton_oxides). Cross-dataset generalization could not be verified because
(a) the Materials Project API was inaccessible during this work, and (b) the
matbench_perovskites dataset reports formation energies in a different reference frame
(Pearson r = 0.358 between the two sources), precluding direct numerical comparison.
Leave-one-element-out (LOEO) extrapolation over 68 elements partially addresses
generalization, but does not substitute for independent DFT validation.

**2. Point–interval decoupling tension.** The point prediction (stacking ensemble)
and the prediction interval (CQR) are based on independent models, following the
conformal prediction philosophy (Romano et al., 2019). Consequently, the point
prediction falls outside the 80% CQR interval for ~9% of samples. This is
consistent with the 80% marginal coverage guarantee (which applies to the true
value, not the point estimate), but introduces a practical tension: users receive
a point estimate and an interval that may not contain it. We mitigate this by
reporting the CQR interval center alongside the point prediction (Pearson r = 0.959
between the two), and recommend the interval as the primary uncertainty statement.

**3. Candidate validation depth.** The 9 predicted stable candidates lack DFT or
experimental verification. The three-tier validation (database cross-reference,
leave-element-out robustness, Goldschmidt synthesizability) provides indirect
evidence, with 3 candidates (LaTiO3, YCoO3, LaAlO3) present in independent DFT
databases. However, database presence confirms only that these compositions have
been studied, not that our predicted stability values are accurate (due to the
reference-frame mismatch noted above).

**4. Applicability of the SR suitability criterion is empirical.** The criterion
(SNR ≥ 2, linear R² ≥ 0.5 for SR success) was derived from comparison of two target
properties (formation energy and hull energy) only. Whether it generalizes to other
material properties (e.g., band gap, elastic modulus) requires further validation
on additional targets.

**5. Extrapolation performance is limited.** LOEO R² averages 0.70, with several
elements yielding negative R² (e.g., B, Ac). The physics-informed layer did not
improve extrapolation over pure ML in our experiments. Users should restrict
predictions to the applicability domain (σ below median) for reliable results.

**6. Symbolic regression equation non-uniqueness.** The SR equations differ in
structure across folds and seeds (non-unique solutions), though the consensus on
key physical quantities (a_site_en appears in 100% of equations) is stable.
We report consensus frequencies rather than a single definitive equation.
