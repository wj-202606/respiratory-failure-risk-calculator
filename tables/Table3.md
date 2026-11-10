**Table 3. Model discrimination and calibration in the development, internal validation, and external validation cohorts (MIMIC-IV and eICU-CRD)**

| Analysis | n | Events | AUC (95% CI) | Brier score (95% CI) | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|
| Training subset | 1420 | 154 | 0.767 | 0.088 | — | — |
| Internal validation | 608 | 72 | 0.727 (SD 0.006) | 0.098 (SD 0.001) | 0.797 (0.781–0.819)† | -0.356 (-0.385–-0.316)† |
| External validation (MIMIC-IV, BE, original) | 434 | 88 | 0.620 (0.555–0.681) | 0.166 (0.155–0.178) | 0.410 (0.170–0.658) | -0.637 (-1.052–-0.220) |
| External validation (MIMIC-IV, BE, recal) | 434 | 88 | 0.620 (0.555–0.681) | 0.158 | 0.999 | −0.001 |
| External validation (MIMIC-IV, HCO₃⁻, original) | 684 | 109 | 0.642 (0.588–0.700) | 0.137 (0.129–0.145) | 0.485 (0.291–0.714) | -0.809 (-1.139–-0.425) |
| External validation (MIMIC-IV, HCO₃⁻, recal) | 684 | 109 | 0.642 (0.588–0.700) | 0.131 | 1.002 | 0.004 |
| External validation (eICU-CRD, BE, original) | 1458 | 363 | 0.670 (0.638–0.703) | 0.179 (0.167–0.190) | 0.599 | −0.486 |
| External validation (eICU-CRD, BE, recal) | 1458 | 363 | 0.670 (0.638–0.703) | 0.174 | 1.000 | 0.000 |
| External validation (eICU-CRD, HCO₃⁻, original) | 2287 | 521 | 0.670 (0.643–0.696) | 0.171 (0.162–0.180) | 0.619 | −0.591 |
| External validation (eICU-CRD, HCO₃⁻, recal) | 2287 | 521 | 0.670 (0.643–0.696) | 0.164 | 1.000 | 0.000 |

AUC, area under the receiver operating characteristic curve; CI, confidence interval; BE, base excess; HCO₃⁻, bicarbonate; eICU-CRD, eICU Collaborative Research Database; recal, recalibrated; —, not applicable. For MIMIC-IV, 95% CIs for calibration slope and intercept are reported alongside the point estimates; for eICU-CRD, calibration slope and intercept are reported as point estimates because bootstrap CIs for these parameters were not generated in the eICU validation script. Calibration slope and intercept were estimated jointly from a logistic model of the observed outcome on the linear predictor; the calibration intercept should therefore be interpreted conditional on the fitted slope. † Internal-validation calibration metrics are reported as mean (Q25–Q75) across 50 imputed internal-validation datasets. Development-cohort coefficients were frozen and applied to each external cohort without refitting. For MIMIC-IV, 95% CIs for AUC, Brier score, calibration slope, and calibration intercept were estimated by outcome-stratified percentile bootstrap resampling (B = 2,000; seed 20261101). For eICU-CRD, 95% CIs for AUC and Brier score were estimated by bootstrap resampling (B = 2,000). The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission), whereas the development cohort and MIMIC-IV used hospital admission.
