**Table 5. Recalibration and bootstrap validation of the 3-variable model**

| Model version | Stage | n | Events | AUC | Brier score | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|---|
| BE model (MIMIC-IV) | Original | 434 | 88 | 0.620 | 0.166 | 0.410 | −0.637 |
| BE model (MIMIC-IV) | Recalibrated (apparent) | 434 | 88 | 0.620 | 0.158 | 0.999 | −0.001 |
| BE model (MIMIC-IV) | Recalibrated (bootstrap OOB) | — | — | — | 0.1596 [0.1492–0.1696] | 1.0075 [0.6090–1.5430] | -0.0135 [-0.5393–0.7927] |
| HCO₃⁻ model (MIMIC-IV) | Original | 684 | 109 | 0.642 | 0.137 | 0.485 | −0.809 |
| HCO₃⁻ model (MIMIC-IV) | Recalibrated (apparent) | 684 | 109 | 0.642 | 0.131 | 1.002 | 0.004 |
| HCO₃⁻ model (MIMIC-IV) | Recalibrated (bootstrap OOB) | — | — | — | 0.1321 [0.1240–0.1403] | 1.0276 [0.7398–1.4018] | 0.0330 [-0.4308–0.6338] |
| BE model (eICU-CRD) | Original | 1458 | 363 | 0.670 | 0.179 | 0.599 | −0.486 |
| BE model (eICU-CRD) | Recalibrated (apparent) | 1458 | 363 | 0.670 | 0.174 | 1.000 | 0.000 |
| BE model (eICU-CRD) | Recalibrated (bootstrap OOB) | — | — | — | 0.1739 [0.1692–0.1794] | 1.0003 [0.8566–1.1570] | -0.0046 [-0.1718–0.1749] |
| HCO₃⁻ model (eICU-CRD) | Original | 2287 | 521 | 0.670 | 0.171 | 0.619 | −0.591 |
| HCO₃⁻ model (eICU-CRD) | Recalibrated (apparent) | 2287 | 521 | 0.670 | 0.164 | 1.000 | 0.000 |
| HCO₃⁻ model (eICU-CRD) | Recalibrated (bootstrap OOB) | — | — | — | 0.1638 [0.1598–0.1681] | 0.9895 [0.8790–1.1437] | -0.0005 [-0.1486–0.1647] |

AUC, area under the receiver operating characteristic curve; BE, base excess; HCO₃⁻, bicarbonate; MIMIC-IV, Medical Information Mart for Intensive Care IV; eICU-CRD, eICU Collaborative Research Database; OOB, out-of-bag; —, not applicable. Bootstrap OOB values are median [IQR] across 1,000 iterations. Recalibration was fitted separately on each external cohort; apparent values were computed on the same cohort (no independent validation set). AUC is unchanged by recalibration because recalibration is a monotonic transformation of the linear predictor. OOB evaluation provides internal support for the recalibration within each external cohort; it does not constitute independent external validation of the recalibrated model. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission), whereas the development cohort and MIMIC-IV used hospital admission.
