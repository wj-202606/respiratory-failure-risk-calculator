# Supplementary Material

**Manuscript title**: External validation and recalibration of a 3-variable prognostic model for in-hospital mortality in patients with respiratory failure: calibration drift, machine learning comparison, and a web-based risk calculator

**Version**: v6.2.10 (frozen 2026-11-04)
**Data cutoff**: 2026-11-04

---

## Table of Contents

| Item | Description |
|---|---|
| Table S1 | LASSO variable selection frequencies |
| Table S2 | HCO₃⁻ sensitivity analysis |
| Table S4 | DeLong test (pairwise AUC comparisons) |
| Table S5 | Comparison with representative published prognostic models |
| Table S6 | This study — summary row |
| Table S7a | Standardized mean differences: development vs MIMIC-IV |
| Table S7b | Standardized mean differences: development vs eICU-CRD |
| Table S8 | MIMIC-IV included vs excluded |
| Table S9 | MIMIC-IV ALB measured vs not measured |
| Table S10 | ALB fixed at development-cohort mean (two-variable sensitivity) |
| Table S11 | eICU-CRD baseline characteristics |
| Table S12 | eICU-CRD included vs excluded |
| Table S13 | eICU-CRD ALB measured vs not measured |
| Table S14 | Comparison with SOFA and APACHE IVa |
| Figure S1 | Internal validation ROC / calibration |
| Figure S2 | HCO₃⁻ sensitivity (MIMIC-IV) |
| Figure S3 / S3b | SHAP summary and mean |SHAP| |
| Figure S4 | Calibration before / after recalibration (MIMIC-IV) |
| Figure S5 | AUC distribution across 100 random splits |
| Figure S6 | BE model in eICU-CRD |
| Figure S7 | HCO₃⁻ model in eICU-CRD |
| Figure S8 | DCA in eICU-CRD |

Note: Table S3 (Machine learning comparison) has been removed because the full ML comparison is presented as Table 4 in the main manuscript. Tables S6a and S6b (DCA for MIMIC-IV and eICU-CRD) have been incorporated into Tables 6a and 6b of the main manuscript.

---

## Table S1. LASSO variable selection frequencies

| Variable | Selected count (of 50) | Selected frequency |
|---|---|---|
| DD | 50 | 100% |
| APTT | 50 | 100% |
| **ALB** | **50** | **100%** |
| **BUN** | **50** | **100%** |
| **BE** | **50** | **100%** |
| WBC | 49 | 98% |
| Na | 28 | 56% |
| Age | 6 | 12% |
| HCO₃⁻ | 1 | 2% |

Selected count indicates the number of imputed datasets (out of 50) in which each variable was retained at λ.1se. Selected frequency is the corresponding percentage. LASSO regression was performed on z-scored continuous predictors across 50 MICE-imputed training datasets. The final three-variable model (ALB, BUN, BE) was prespecified on the basis of clinical availability and parsimony; variables retained by LASSO but not included in the final model (DD, APTT, WBC) were not subsequently entered. Of note, DD and APTT are not consistently measured at the 24-h landmark in MIMIC-IV or eICU-CRD and could not be externally validated.

---

## Table S2. HCO₃⁻ sensitivity analysis

**A. Frozen coefficients for the HCO₃⁻ model**

| Term | Beta |
|---|---|
| Intercept | 3.193555 |
| ALB (g/L) | −0.120922 |
| BUN (mmol/L) | 0.037409 |
| HCO₃⁻ (mmol/L) | −0.060383 |

Linear predictor: LP = 3.193555 − 0.120922·ALB + 0.037409·BUN − 0.060383·HCO₃⁻
Predicted probability: p = 1 / (1 + exp(−LP)).

**B. External validation**

| Cohort | n | Events | AUC (95% CI) | Brier (95% CI) | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|
| MIMIC-IV | 684 | 109 | 0.642 (0.588–0.700) | 0.137 (0.129–0.145) | 0.485 | −0.809 |
| eICU-CRD | 2,287 | 521 | 0.670 (0.643–0.696) | 0.171 (0.162–0.180) | 0.619 | −0.591 |

**C. Recalibration and bootstrap OOB**

| Cohort | Stage | Brier | Calibration slope | Calibration intercept |
|---|---|---|---|---|
| MIMIC-IV | Original | 0.137 | 0.485 | −0.809 |
| MIMIC-IV | Recalibrated (apparent) | 0.131 | 1.002 | 0.004 |
| MIMIC-IV | Recalibrated (bootstrap OOB) | 0.1321 [0.1240–0.1403] | 1.0276 [0.7398–1.4018] | 0.0330 [−0.4308–0.6338] |
| eICU-CRD | Original | 0.171 | 0.619 | −0.591 |
| eICU-CRD | Recalibrated (apparent) | 0.164 | 1.000 | 0.000 |
| eICU-CRD | Recalibrated (bootstrap OOB) | 0.1638 [0.1598–0.1681] | 0.9895 [0.8790–1.1437] | −0.0005 [−0.1486–0.1647] |

AUC, area under the receiver operating characteristic curve; CI, confidence interval; BE, base excess; HCO₃⁻, bicarbonate; MIMIC-IV, Medical Information Mart for Intensive Care IV; eICU-CRD, eICU Collaborative Research Database; OOB, out-of-bag; —, not applicable. The HCO₃⁻ model substitutes bicarbonate for base excess in the three-variable laboratory model (ALB, BUN, and an acid–base marker). Development-cohort coefficients for the HCO₃⁻ model were frozen and applied to both external cohorts without refitting. For MIMIC-IV, 95% CIs were estimated by outcome-stratified percentile bootstrap resampling (B = 2,000; seed = 20261101). For eICU-CRD, 95% CIs are reported by the eICU validation script (also B = 2,000). Bootstrap OOB values are median [IQR] across 1,000 iterations. Recalibration was fitted separately in each external cohort; apparent values were computed on the same cohort (no independent recalibration-validation set). OOB evaluation provides internal support for the recalibration within each cohort; it does not constitute independent external validation of the recalibrated model. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission), whereas the development cohort and MIMIC-IV used hospital admission.

---

## # Table S4. Pairwise comparisons of model discrimination using DeLong's test

**A. Internal validation cohort (n = 608, 72 events)**

| Cohort | Model 1 | AUC 1 | Model 2 | AUC 2 | Delta AUC | z | P (DeLong) | P (Holm) |
|---|---|---|---|---|---|---|---|---|
| Internal validation | Logistic | 0.7435 | RandomForest | 0.7478 | -0.0044 | -0.3891 | 0.697 | 1.000 |
| Internal validation | Logistic | 0.7435 | XGBoost | 0.7415 | 0.002 | 0.1833 | 0.855 | 1.000 |
| Internal validation | Logistic | 0.7435 | LightGBM | 0.7197 | 0.0238 | 1.2767 | 0.202 | 0.807 |
| Internal validation | Logistic | 0.7435 | SVM | 0.5869 | 0.1566 | 3.5233 | <0.001 | 0.004 |
| Internal validation | Logistic | 0.7435 | MLP | 0.4452 | 0.2983 | 5.6875 | <0.001 | <0.001 |
| Internal validation | RandomForest | 0.7478 | XGBoost | 0.7415 | 0.0064 | 0.8599 | 0.390 | 1.000 |
| Internal validation | RandomForest | 0.7478 | LightGBM | 0.7197 | 0.0281 | 1.8014 | 0.072 | 0.430 |
| Internal validation | RandomForest | 0.7478 | SVM | 0.5869 | 0.161 | 3.6874 | <0.001 | 0.002 |
| Internal validation | RandomForest | 0.7478 | MLP | 0.4452 | 0.3027 | 5.7355 | <0.001 | <0.001 |
| Internal validation | XGBoost | 0.7415 | LightGBM | 0.7197 | 0.0217 | 1.7867 | 0.074 | 0.430 |
| Internal validation | XGBoost | 0.7415 | SVM | 0.5869 | 0.1546 | 3.502 | <0.001 | 0.004 |
| Internal validation | XGBoost | 0.7415 | MLP | 0.4452 | 0.2963 | 5.775 | <0.001 | <0.001 |
| Internal validation | LightGBM | 0.7197 | SVM | 0.5869 | 0.1329 | 2.7637 | 0.006 | 0.046 |
| Internal validation | LightGBM | 0.7197 | MLP | 0.4452 | 0.2746 | 5.3294 | <0.001 | <0.001 |
| Internal validation | SVM | 0.5869 | MLP | 0.4452 | 0.1417 | 2.4148 | 0.016 | 0.110 |


**B. External validation cohort (MIMIC-IV, n = 434, 88 events)**

| Cohort | Model 1 | AUC 1 | Model 2 | AUC 2 | Delta AUC | z | P (DeLong) | P (Holm) |
|---|---|---|---|---|---|---|---|---|
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | Python_Logistic | 0.6199 | 0.0005 | 0.5779 | 0.563 | 1.000 |
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | RandomForest | 0.6085 | 0.0119 | 0.6685 | 0.504 | 1.000 |
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | XGBoost | 0.6028 | 0.0176 | 1.1636 | 0.245 | 1.000 |
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | LightGBM | 0.5866 | 0.0338 | 1.8495 | 0.064 | 0.708 |
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | SVM | 0.5022 | 0.1182 | 2.9492 | 0.003 | 0.061 |
| External validation (MIMIC-IV) | FrozenR_Logistic | 0.6204 | MLP | 0.4387 | 0.1817 | 3.0161 | 0.003 | 0.054 |
| External validation (MIMIC-IV) | Python_Logistic | 0.6199 | RandomForest | 0.6085 | 0.0114 | 0.6437 | 0.520 | 1.000 |
| External validation (MIMIC-IV) | Python_Logistic | 0.6199 | XGBoost | 0.6028 | 0.0171 | 1.1357 | 0.256 | 1.000 |
| External validation (MIMIC-IV) | Python_Logistic | 0.6199 | LightGBM | 0.5866 | 0.0332 | 1.8282 | 0.068 | 0.708 |
| External validation (MIMIC-IV) | Python_Logistic | 0.6199 | SVM | 0.5022 | 0.1177 | 2.933 | 0.003 | 0.061 |
| External validation (MIMIC-IV) | Python_Logistic | 0.6199 | MLP | 0.4387 | 0.1812 | 3.0075 | 0.003 | 0.054 |
| External validation (MIMIC-IV) | RandomForest | 0.6085 | XGBoost | 0.6028 | 0.0057 | 0.6425 | 0.521 | 1.000 |
| External validation (MIMIC-IV) | RandomForest | 0.6085 | LightGBM | 0.5866 | 0.0219 | 1.6874 | 0.092 | 0.824 |
| External validation (MIMIC-IV) | RandomForest | 0.6085 | SVM | 0.5022 | 0.1063 | 2.6686 | 0.008 | 0.129 |
| External validation (MIMIC-IV) | RandomForest | 0.6085 | MLP | 0.4387 | 0.1698 | 2.6655 | 0.008 | 0.129 |
| External validation (MIMIC-IV) | XGBoost | 0.6028 | LightGBM | 0.5866 | 0.0161 | 1.5114 | 0.131 | 1.000 |
| External validation (MIMIC-IV) | XGBoost | 0.6028 | SVM | 0.5022 | 0.1006 | 2.549 | 0.011 | 0.151 |
| External validation (MIMIC-IV) | XGBoost | 0.6028 | MLP | 0.4387 | 0.1641 | 2.6383 | 0.008 | 0.129 |
| External validation (MIMIC-IV) | LightGBM | 0.5866 | SVM | 0.5022 | 0.0845 | 2.0336 | 0.042 | 0.504 |
| External validation (MIMIC-IV) | LightGBM | 0.5866 | MLP | 0.4387 | 0.148 | 2.4429 | 0.015 | 0.189 |
| External validation (MIMIC-IV) | SVM | 0.5022 | MLP | 0.4387 | 0.0635 | 1.0462 | 0.295 | 1.000 |

AUC, area under the receiver operating characteristic curve; DeLong, DeLong's test for two correlated ROC curves; Holm, Holm–Bonferroni adjustment for multiple comparisons (15 tests within internal, 21 within external). Delta AUC is Model 1 minus Model 2. P values below 0.001 are reported as <0.001. The FrozenR_Logistic model uses the R deployment coefficients; Python_Logistic is an independent sklearn implementation and should not be treated as identical. Both the unadjusted and Holm-adjusted p values are reported; interpretation should prioritize the Holm-adjusted values to avoid over-interpretation of nominally significant results from multiple testing.

---

## Table S5. Comparison with representative published prognostic models

| Model / Study | Population | Derivation n | External validation | Reported AUC | Calibration reported | Recalibration performed | Source |
|---|---|---|---|---|---|---|---|
| CURB-65 (Lim 2003) | Community-acquired pneumonia | 1,068 | Yes | 0.73–0.79 | No | No | Thorax 2003;58:377–382 |
| PSI (Fine 1997) | Community-acquired pneumonia | 14,199 | Yes | 0.75–0.85 | Partial | No | N Engl J Med 1997;336:243–250 |
| SOFA (Vincent 1996) | ICU patients (mixed) | 1,449 | Yes | 0.70–0.85 | No | No | Intensive Care Med 1996;22:707–710 |
| APACHE II (Knaus 1985) | ICU patients (mixed) | 5,815 | Yes | 0.75–0.85 | No | No | Crit Care Med 1985;13:818–829 |
| This study | Respiratory failure (J96) | 2,028 | Yes (MIMIC-IV n = 434 BE / 684 HCO₃⁻; eICU-CRD n = 1,458 BE / 2,287 HCO₃⁻) | 0.620 (MIMIC-IV BE); 0.642 (MIMIC-IV HCO₃⁻); 0.670 (eICU BE); 0.670 (eICU HCO₃⁻) | Yes (slope, intercept, bootstrap OOB) | Yes (slope + intercept) | This study |

AUC, area under the receiver operating characteristic curve; PSI, Pneumonia Severity Index; SOFA, Sequential Organ Failure Assessment; APACHE II, Acute Physiology and Chronic Health Evaluation II; BE, base excess; HCO₃⁻, bicarbonate. AUC ranges for the published models are taken verbatim from the original publications and reflect the ranges reported across the derivation and validation cohorts in those papers. They are NOT head-to-head comparisons with the present model: populations, outcome definitions, and prediction time windows differ between studies. The present study's AUC is reported for BE and HCO₃⁻ separately because the two models use different acid-base predictors.

---

## Table S6. This study — summary row

| Model / Study | Population | Derivation n | External validation | Discrimination (AUC) | Calibration reported | Recalibration performed |
|---|---|---|---|---|---|---|
| This study (ALB + BUN + BE; HCO₃⁻ sensitivity) | Respiratory failure (J96) | 2,028 | Yes (MIMIC-IV, n = 434 BE / 684 HCO₃⁻; eICU-CRD, n = 1,458 BE / 2,287 HCO₃⁻) | MIMIC-IV AUC 0.620 (BE) and 0.642 (HCO₃⁻); eICU-CRD 0.670 (BE); 0.670 (HCO₃⁻) | Yes (slope, intercept, bootstrap OOB; see Table 3, Table 5, Figure 2, Figure S2) | Yes (slope + intercept) |

---

## Table S7a. Baseline comparison — development vs MIMIC-IV external validation cohort

| Variable | Development (n = 2,028) | MIMIC-IV (n = 434) | SMD |
|---|---|---|---|
| Age, years, mean (SD) | 76.0 (12.0) | 64.1 (15.4) | 0.94 |
| Male, % | 71.1 | 56.7 | 0.30 |
| ALB, g/L, mean (SD) | 34.5 (5.3) | 32.6 (5.8) | 0.34 |
| BUN, mmol/L, mean (SD) | 8.7 (6.2) | 10.4 (7.9) | 0.26 |
| BE, mmol/L, mean (SD) | 4.2 (7.8) | 0.4 (6.5) | 0.49 |
| In-hospital mortality, % | 11.1 | 20.3 | 0.25 |

SMD, standardized mean difference; an SMD above 0.10 is conventionally considered non-negligible; BE, base excess. The MIMIC-IV cohort here is the same 434-case BE complete-case cohort used in Table 3 and Figure 2. Age and sex were obtained from the MIMIC-IV demographics file matched by hadm_id. All clinical values are used as-is; no imputation is performed. This SMD table is descriptive and is not a formal hypothesis test.

---

## Table S7b. Baseline comparison — development vs eICU-CRD external validation cohorts

| Variable | Development (n = 2,028) | eICU-CRD BE (n = 1,458) | SMD (BE) | eICU-CRD HCO₃⁻ (n = 2,287) | SMD (HCO₃⁻) |
|---|---|---|---|---|---|
| Age, years, mean (SD) | 76.0 (12.0) | 64.1 (16.0) | 0.86 | 64.4 (16.2) | 0.81 |
| Male, % | 71.1 | 53.0 | 0.38 | 51.7 | 0.41 |
| ALB, g/L, mean (SD) | 34.5 (5.3) | 27.4 (6.4) | 1.21 | 27.2 (6.5) | 1.21 |
| BUN, mmol/L, mean (SD) | 8.7 (6.2) | 11.5 (8.3) | 0.39 | 11.4 (8.2) | 0.36 |
| BE, mmol/L, mean (SD) | 4.2 (7.8) | −0.2 (7.3) | 0.58 | — | — |
| HCO₃⁻, mmol/L, mean (SD) | 29.0 (8.0) | — | — | 24.9 (6.2) | 0.58 |
| In-hospital mortality, % | 11.1 | 24.9 | 0.36 | 22.8 | 0.31 |

SMD, standardized mean difference; an SMD above 0.10 is conventionally considered non-negligible; BE, base excess; HCO₃⁻, bicarbonate; eICU-CRD, eICU Collaborative Research Database. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission), whereas the development cohort used hospital admission.

---

## Table S8. MIMIC-IV included vs excluded

| Variable | Included (n = 434) | Excluded (n = 2,022) | SMD | P value |
|---|---|---|---|---|
| Age, years, mean (SD) | 64.1 (15.4) | 65.6 (15.3) | 0.09 | 0.074 |
| Male, n (%) | 246 (56.7) | 987 (48.8) | 0.16 | 0.003 |
| BUN, mmol/L, mean (SD) | 10.39 (7.91) | 8.33 (6.06) | 0.29 | <0.001 |
| BE, mmol/L, mean (SD) | 0.44 (6.48) | 1.35 (5.31) | 0.15 | 0.013 |
| HCO₃⁻, mmol/L, mean (SD) | 24.36 (6.12) | 25.28 (3.87) | 0.18 | 0.016 |
| In-hospital mortality, n (%) | 88 (20.3) | 163 (8.1) | 0.36 | <0.001 |

SMD, standardized mean difference; BE, base excess; HCO₃⁻, bicarbonate; MIMIC-IV, Medical Information Mart for Intensive Care IV. Included patients are those with complete ALB, BUN, BE, and outcome (BE complete-case cohort, n = 434). Excluded patients are the remaining 2,022 admissions in the final MIMIC-IV cohort of 2,456. P values: Welch's t-test for continuous variables; chi-square test for categorical variables.

---

## Table S9. MIMIC-IV ALB measured vs not measured

| Variable | ALB measured (n = 684) | ALB not measured (n = 1,772) | SMD | P value |
|---|---|---|---|---|
| Age, years, mean (SD) | 64.9 (15.5) | 65.5 (15.3) | 0.04 | 0.375 |
| Male, % | 54.8 | 48.4 | 0.13 | 0.005 |
| BUN, mmol/L, mean (SD) | 9.46 (7.12) | 8.41 (6.19) | 0.16 | <0.001 |
| BE, mmol/L, mean (SD) | 0.44 (6.48) | 1.35 (5.31) | 0.15 | 0.013 |
| In-hospital mortality, % | 15.9 | 8.0 | 0.25 | <0.001 |

SMD, standardized mean difference; BE, base excess; MIMIC-IV, Medical Information Mart for Intensive Care IV. Male and in-hospital mortality are reported as percentages only because the raw numerator/denominator are not retained in the underlying CSV. P values: Welch's t-test for continuous variables; chi-square test for categorical variables.

---

## Table S10. ALB fixed at development-cohort mean (two-variable sensitivity analysis)

| Model | n | Events | AUC | Brier | Calibration slope | Calibration intercept |
|---|---|---|---|---|---|---|
| Two-variable (ALB fixed at development-cohort mean 34.44 g/L) | 1,169 | 182 | 0.586 | 0.132 | 0.604 | −0.406 |
| Three-variable (complete cases) | 434 | 88 | 0.620 | 0.166 | 0.410 | −0.637 |

AUC, area under the receiver operating characteristic curve; ALB, albumin. The two rows are drawn from different samples and are not directly comparable. The two-variable analysis fixed ALB at the development-cohort mean (34.44 g/L) and included all patients with BUN and BE (n = 1,169, 182 deaths); the three-variable model included only patients with complete ALB, BUN, and BE (n = 434, 88 deaths). The difference in AUC should therefore not be interpreted as the isolated contribution of ALB. Calibration slope and intercept were estimated jointly from a logistic model of the observed outcome on the linear predictor.

---

## Table S11. eICU-CRD baseline characteristics

| Variable | eICU-CRD BE (n = 1,458) | eICU-CRD HCO₃⁻ (n = 2,287) |
|---|---|---|
| Age, years, mean (SD) | 64.1 (16.0) | 64.4 (16.2) |
| Male, n (%) | 772 (53.0) | 1,183 (51.7) |
| ALB, g/L, mean (SD) | 27.4 (6.4) | 27.2 (6.5) |
| BUN, mmol/L, mean (SD) | 11.5 (8.3) | 11.4 (8.2) |
| BE, mmol/L, mean (SD) | −0.2 (7.3) | — |
| HCO₃⁻, mmol/L, mean (SD) | — | 24.9 (6.2) |
| In-hospital death, n (%) | 363 (24.9) | 521 (22.8) |

BE, base excess; HCO₃⁻, bicarbonate; eICU-CRD, eICU Collaborative Research Database. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission).

---

## Table S12. eICU-CRD included vs excluded

| Version | Variable | Included | Excluded | SMD | P value |
|---|---|---|---|---|---|
| BE | Age, years, mean (SD) | 64.1 (16.0) | 66.0 (15.9) | 0.12 | <0.001 |
| BE | Male, n (%) | 772 (53.0) | 1,579 (48.8) | 0.08 | 0.009 |
| BE | BUN, mmol/L, mean (SD) | 11.5 (8.3) | 10.4 (7.5) | 0.13 | <0.001 |
| BE | In-hospital death, n (%) | 363 (24.9) | 614 (19.0) | 0.14 | <0.001 |
| HCO₃⁻ | Age, years, mean (SD) | 64.4 (16.2) | 66.3 (15.7) | 0.12 | <0.001 |
| HCO₃⁻ | Male, n (%) | 1,183 (51.7) | 1,168 (48.5) | 0.07 | 0.028 |
| HCO₃⁻ | BUN, mmol/L, mean (SD) | 11.4 (8.2) | 10.1 (7.2) | 0.16 | <0.001 |
| HCO₃⁻ | In-hospital death, n (%) | 521 (22.8) | 456 (18.9) | 0.09 | 0.001 |

SMD, standardized mean difference; an SMD above 0.10 is conventionally considered non-negligible; BE, base excess; HCO₃⁻, bicarbonate; eICU-CRD, eICU Collaborative Research Database. P values: Welch's t-test for continuous variables; chi-square test for categorical variables. The eICU-CRD evaluation used ICU admission as the time origin (prediction time 24 h after ICU admission).

---

## Table S13. eICU-CRD ALB measured vs not measured (n = 4,696)

| Variable | ALB measured (n = 2,390) | ALB not measured (n = 2,306) | SMD | P value |
|---|---|---|---|---|
| Age, years, mean (SD) | 64.5 (16.1) | 66.3 (15.7) | 0.11 | <0.001 |
| Male, % | 1,239 (51.8) | 1,112 (48.2) | 0.07 | 0.014 |
| BUN, mmol/L, mean (SD) | 11.4 (8.2) | 10.1 (7.2) | 0.16 | <0.001 |
| BE, mmol/L, mean (SD) | −0.2 (7.4) | 2.0 (7.1) | 0.31 | <0.001 |
| In-hospital death, n (%) | 543 (22.7) | 434 (18.8) | 0.10 | 0.001 |

SMD, standardized mean difference; BE, base excess; eICU-CRD, eICU Collaborative Research Database. P values: Welch's t-test for continuous variables; chi-square test for categorical variables.

---

## Table S14. Comparison of the three-variable prediction models with established ICU severity scores

| Cohort | Model | n | AUC (95% CI) | Difference (95% CI) | DeLong P |
|---|---|---|---|---|---|
| MIMIC-IV BE | 3-variable | 393 | 0.613 (0.547–0.680) | — | — |
| MIMIC-IV BE | SOFA | 393 | 0.638 (0.570–0.705) | −0.025 (−0.092 to 0.044) | 0.485 |
| MIMIC-IV HCO₃⁻ | 3-variable | 467 | 0.612 (0.551–0.671) | — | — |
| MIMIC-IV HCO₃⁻ | SOFA | 467 | 0.642 (0.580–0.701) | −0.029 (−0.102 to 0.035) | 0.394 |
| eICU-CRD BE | 3-variable | 1,136 | 0.671 (0.633–0.709) | — | — |
| eICU-CRD BE | APACHE IVa | 1,136 | 0.727 (0.693–0.761) | −0.056 (−0.098 to −0.015) | 0.009 |
| eICU-CRD HCO₃⁻ | 3-variable | 1,722 | 0.678 (0.647–0.708) | — | — |
| eICU-CRD HCO₃⁻ | APACHE IVa | 1,722 | 0.739 (0.711–0.767) | −0.062 (−0.096 to −0.028) | <0.001 |

SOFA, Sequential Organ Failure Assessment; APACHE IVa, Acute Physiology and Chronic Health Evaluation IVa; AUC, area under the receiver operating characteristic curve; CI, confidence interval; BE, base excess; HCO₃⁻, bicarbonate; —, not applicable. The MIMIC-IV SOFA comparison was restricted to patients in the primary MIMIC-IV BE or HCO₃⁻ complete-case cohort with an identifiable ICU stay and an available first-day SOFA score. The earliest ICU stay was selected for each hospital admission. In the MIMIC-IV first_day_sofa derived concept, missing component scores are treated as 0 (normal) when calculating the total SOFA score. In the eICU-CRD APACHE IVa comparison, records with predictedhospitalmortality = −1 (APACHE prediction not available) were excluded, consistent with the eICU-CRD data dictionary and the MIT-LCP analysis practice; the analysis was restricted to records with valid APACHE IVa predictions in [0, 1]. The primary prediction models used laboratory measurements obtained within the first 24 hours after hospital admission, whereas SOFA and APACHE IVa represented the first 24 hours after ICU admission and the full ICU stay, respectively. Therefore, these comparisons involved subsets of the primary external validation cohorts and different prediction time origins. APACHE IVa achieved significantly higher AUCs than both three-variable models. DeLong's test used the paired covariance of the two AUCs among the same patients; AUCs and ΔAUCs are reported with 95% CIs estimated by paired, outcome-stratified bootstrap resampling (B = 2,000), with both predictions resampled in the same patients to preserve the paired structure. ΔAUC 95% CIs were not derived by subtracting the two independent AUC CIs.

---

## Figure S1. Internal validation ROC and calibration

**Panel A**: Receiver operating characteristic curve. The AUC was 0.727 (IQR 0.724–0.730 across 50 imputations). The curve was drawn from the first imputed dataset.

**Panel B**: Apparent calibration plot across 50 imputations. The estimated calibration slope was 0.797 and the intercept was −0.356. The dashed diagonal line represents perfect calibration.

---

## Figure S2. HCO₃⁻ sensitivity analysis (MIMIC-IV)

**Panel A**: Receiver operating characteristic curve. AUC 0.642 (95% CI 0.588–0.700).

**Panel B**: Calibration plot. Slope 0.485; intercept −0.809.

---

## Figure S3 / S3b. SHAP summary and mean |SHAP|

XGBoost model trained on the development cohort (n = 1,420). ALB shows the largest mean |SHAP| value, followed by BE and BUN. Directions are consistent with the logistic model coefficients (higher ALB and BE associated with lower risk, higher BUN associated with higher risk).

---

## Figure S4. Calibration before and after recalibration (MIMIC-IV)

**Panel A (BE)**: original slope 0.410, intercept −0.637; recalibrated slope 0.999, intercept −0.001.

**Panel B (HCO₃⁻)**: original slope 0.485, intercept −0.809; recalibrated slope 1.002, intercept 0.004.

---

## Figure S5. AUC distribution across 100 random splits

Median AUC 0.751 (IQR 0.731–0.772). Each split repeats the full imputation and estimation pipeline for the prespecified fixed three-variable model; variable selection is not repeated.

---

## Figure S6. BE model in eICU-CRD

n = 1,458, 363 deaths. AUC 0.670 (95% CI 0.638–0.703). Calibration slope 0.599; intercept −0.486. Prediction time: 24 h after ICU admission.

---

## Figure S7. HCO₃⁻ model in eICU-CRD

n = 2,287, 521 deaths. AUC 0.670 (95% CI 0.643–0.696). Calibration slope 0.619; intercept −0.591. Prediction time: 24 h after ICU admission.

---

## Figure S8. Decision curve analysis in eICU-CRD

At 10%: BE recalibrated net benefit 0.1630 vs treat-all 0.1655; HCO₃⁻ 0.1395 vs 0.1420. At 20%–30%: BE recalibrated exceeded treat-all. At 15%–30%: HCO₃⁻ recalibrated exceeded treat-all. The exploratory threshold range is 10%–30%.