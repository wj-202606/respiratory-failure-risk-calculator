**Table S2. Sensitivity analysis of the three-variable model using bicarbonate instead of base excess**

**A. Frozen coefficients for the HCO₃⁻ model**

| Term | Beta |
|---|---|
| Intercept | 3.193555 |
| ALB (g/L) | -0.120922 |
| BUN (mmol/L) | 0.037409 |
| HCO₃⁻ (mmol/L) | -0.060383 |

Linear predictor: `LP = 3.193555 − 0.120922·ALB + 0.037409·BUN − 0.060383·HCO₃⁻`

Predicted probability: `p = 1 / (1 + exp(-LP))`.

**B. External validation and recalibration**

| Cohort | n | Events | AUC (95% CI) | Brier (95% CI) | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|
| MIMIC-IV | 684 | 109 | 0.642 (0.588–0.700) | 0.137 (0.129–0.145) | 0.485 | -0.809 |
| eICU-CRD | 2287 | 521 | 0.670 (0.643–0.696) | 0.171 (0.162–0.180) | 0.619 | -0.591 |

**C. Recalibration and bootstrap OOB**

| Cohort | Stage | Brier | Calibration slope | Calibration intercept |
|---|---|---|---|---|
| MIMIC-IV | Original | 0.137 | 0.485 | -0.809 |
| MIMIC-IV | Recalibrated (apparent) | 0.131 | 1.002 | 0.004 |
| MIMIC-IV | Recalibrated (bootstrap OOB) | 0.1321 [0.1240–0.1403] | 1.0276 [0.7398–1.4018] | 0.0330 [-0.4308–0.6338] |
| eICU-CRD | Original | 0.171 | 0.619 | -0.591 |
| eICU-CRD | Recalibrated (apparent) | 0.164 | 1.000 | 0.000 |
| eICU-CRD | Recalibrated (bootstrap OOB) | 0.1638 [0.1598–0.1681] | 0.9895 [0.8790–1.1437] | -0.0005 [-0.1486–0.1647] |

AUC, area under the receiver operating characteristic curve; CI, confidence interval; BE, base excess; HCO₃⁻, bicarbonate; MIMIC-IV, Medical Information Mart for Intensive Care IV; eICU-CRD, eICU Collaborative Research Database; OOB, out-of-bag; —, not applicable. The HCO₃⁻ model substitutes bicarbonate for base excess in the three-variable laboratory model (ALB, BUN, and an acid–base marker). Development-cohort coefficients for the HCO₃⁻ model were frozen and applied to both external cohorts without refitting. For MIMIC-IV, 95% CIs were estimated by outcome-stratified percentile bootstrap resampling (B = 2,000; seed = 20261101); the MIMIC-IV table row is the same one reported in Table 3. For eICU-CRD, 95% CIs are reported by the eICU validation script (also B = 2,000). Bootstrap OOB values are median [IQR] across 1,000 iterations. Recalibration was fitted separately in each external cohort; apparent values were computed on the same cohort (no independent recalibration-validation set). OOB evaluation provides internal support for the recalibration within each cohort; it does not constitute independent external validation of the recalibrated model. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission), whereas the development cohort and MIMIC-IV used hospital admission.
