# Supplementary Material

**Manuscript title**: Development and external validation of a 3-variable prognostic model for in-hospital mortality in patients with respiratory failure: a retrospective cohort study with machine learning comparison

**Authors**: [待填]

**Corresponding author**: [待填]

**Version**: v1.0
**Date**: 2026-10-11

---

## Table of Contents

| Item | Description |
|---|---|
| Table S1 | LASSO variable selection frequencies across 50 imputations |
| Table S2 | 4-variable sensitivity analysis (DD + ALB + BUN + BE) |
| Table S3 | HCO₃⁻ sensitivity analysis (full results) |
| Table S4 | Machine learning comparison (internal and external) |
| Table S5 | DeLong test (pairwise AUC comparisons) |
| Figure S1 | Internal validation ROC and calibration curves |
| Figure S2 | HCO₃⁻ sensitivity analysis ROC and calibration |
| Figure S3 | SHAP dependence plot — Albumin |
| Figure S4 | SHAP dependence plot — BUN |
| Figure S5 | SHAP dependence plot — Base excess |
| Appendix 1 | TRIPOD+AI checklist |
| Appendix 2 | PostgreSQL extraction code |
| Appendix 3 | Python and R analysis code |

---

## Table S1. LASSO variable selection frequencies

LASSO regression with 10-fold cross-validation (lambda.1se) was applied independently to each of the 50 imputed datasets. Variables selected in ≥60% of datasets were retained (highlighted in bold).

| Variable | Selection frequency |
|---|---|
| **ALB** | **100%** |
| **BUN** | **100%** |
| **BE** | **98%** |
| DD | 92% |
| APTT | 70% |
| PT | 44% |
| CRP | 42% |
| WBC | 38% |
| NEU | 36% |
| LYM | 30% |
| RBC | 28% |
| Hb | 26% |
| RDW | 24% |
| PLT | 22% |
| ALT | 18% |
| AST | 16% |
| TBIL | 14% |
| CREA | 12% |
| K | 10% |
| Na | 8% |
| Cl | 6% |
| Ca | 6% |
| PH | 4% |
| PaO₂ | 4% |
| PaCO₂ | 2% |
| HCO₃⁻ | 2% |
| TCO₂ | 2% |
| SaO₂ | 0% |
| LAC | 0% |
| Age | 0% |
| Sex | 0% |

*Note: Values are based on the LASSO selection frequencies reported in the main text. The full dataset is available as `results/LASSO_vars_frequency.csv`.*

*Abbreviations: ALB, albumin; BUN, blood urea nitrogen; BE, base excess; DD, D-dimer; APTT, activated partial thromboplastin time; PT, prothrombin time; CRP, C-reactive protein; WBC, white blood cell; NEU, neutrophil; LYM, lymphocyte; RBC, red blood cell; Hb, hemoglobin; RDW, red cell distribution width; PLT, platelet; ALT, alanine aminotransferase; AST, aspartate aminotransferase; TBIL, total bilirubin; CREA, creatinine; K, potassium; Na, sodium; Cl, chloride; Ca, calcium; PH, arterial pH; PaO₂, partial pressure of oxygen; PaCO₂, partial pressure of carbon dioxide; HCO₃⁻, bicarbonate; TCO₂, total CO₂; SaO₂, arterial oxygen saturation; LAC, lactate.*

---

## Table S2. Four-variable sensitivity analysis

To assess whether adding D-dimer (DD) improves model performance, we refit the model with four variables (DD + ALB + BUN + BE). The comparison with the 3-variable primary model is shown below.

### Model coefficients (original scale)

| Variable | 3-variable model β | 4-variable model β |
|---|---|---|
| Intercept | −2.3843 | −2.3994 |
| DD (per mg/L) | — | +0.0270 |
| ALB (per g/L) | −0.1282 | −0.1234 |
| BUN (per mmol/L) | +0.0320 | +0.0284 |
| BE (per mmol/L) | −0.0523 | −0.0519 |

### Model performance

| Metric | 3-variable model | 4-variable model | Δ |
|---|---|---|---|
| Training AUC | 0.7717 | 0.7823 | +0.0106 |
| Internal validation AUC | 0.7315 | 0.7344 | +0.0029 |
| Training Brier | 0.0899 | 0.0896 | −0.0003 |
| Internal validation Brier | 0.0918 | 0.0905 | −0.0013 |

### Conclusion

Adding DD improved the internal validation AUC by only 0.0029 and the Brier score by 0.0013 — differences that are clinically negligible. Moreover, the addition of DD did not substantially change the coefficients of the other three predictors, indicating no meaningful collinearity. **We therefore retained the more parsimonious 3-variable model as the primary model.**

*Note: All analyses in this table were performed in Python 3.10. Small numerical differences from the R-based primary analysis (e.g., training AUC 0.7717 vs 0.7852) reflect differences in default regularization settings (R `glm` is unregularized; Python `scikit-learn` uses L2 regularization with C = 1.0) and were not considered clinically meaningful.*

---

## Table S3. HCO₃⁻ sensitivity analysis (full results)

Because BE was available in only 49.0% of MIMIC-IV patients, whereas HCO₃⁻ was available in 93.7%, we repeated the external validation using HCO₃⁻ as a substitute acid-base marker.

### Training cohort coefficients (standardized scale)

| Variable | β |
|---|---|
| Intercept | −2.3860 |
| ALB | −0.6644 |
| BUN | 0.2306 |
| HCO₃⁻ | −0.4287 |

### External validation performance

| Metric | Value | 95% CI |
|---|---|---|
| n | 684 | — |
| Events | 109 (15.9%) | — |
| AUC | 0.6410 | 0.5862–0.6949 |
| Brier score | 0.1367 | 0.1185–0.1561 |
| Calibration slope | 0.4799 | 0.2744–0.6836 |
| Calibration intercept | −0.8043 | −1.2086 to −0.4025 |

### Comparison with BE version

| Metric | BE version | HCO₃⁻ version |
|---|---|---|
| n | 434 | 684 |
| Events | 88 (20.3%) | 109 (15.9%) |
| AUC | 0.6187 | 0.6410 |
| Brier | 0.1663 | 0.1367 |
| Calibration slope | 0.4042 | 0.4799 |
| Calibration intercept | −0.6358 | −0.8043 |

*Note: Because the HCO₃⁻ model was refit in the development cohort, this analysis should be interpreted as a variable-substitution sensitivity analysis rather than an independent external validation.*

*⚠️ Important: In the data file `mimic_wide_hco3.csv`, the column named `be_mmol_l` actually contains HCO₃⁻ values, not BE values. This naming issue is documented in the code comments.*

---

## Table S4. Machine learning comparison

Five machine learning algorithms were trained on the same three predictors (ALB, BUN, BE) and evaluated in internal validation (n = 607, 67 deaths) and external validation (n = 434, 88 deaths).

### Internal validation

| Model | AUC | Brier |
|---|---|---|
| Random forest | 0.7339 | 0.0911 |
| Logistic regression | 0.7315 | 0.0918 |
| LightGBM | 0.7218 | 0.0921 |
| XGBoost | 0.6983 | 0.0985 |
| SVM | 0.5823 | 0.0955 |
| MLP | 0.4597 | 0.1604 |

### External validation

| Model | AUC | Brier |
|---|---|---|
| Random forest | 0.6203 | 0.1627 |
| Logistic regression | 0.6187 | 0.1663 |
| LightGBM | 0.6103 | 0.1649 |
| XGBoost | 0.5757 | 0.1834 |
| SVM | 0.4775 | 0.1704 |
| MLP | 0.4534 | 0.1979 |

*Abbreviations: AUC, area under the receiver operating characteristic curve; SVM, support vector machine; MLP, multilayer perceptron; Brier, Brier score.*

*Hyperparameters*:
- Random forest: 500 trees, max depth 6, min samples per leaf 10
- XGBoost: 1,000 trees, learning rate 0.03, max depth 3, L2 = 1.0, L1 = 0.1, early stopping
- LightGBM: 1,000 trees, learning rate 0.03, max depth 3, num leaves 8, early stopping
- SVM: RBF kernel, C = 1.0, gamma = "scale"
- MLP: hidden layers (32, 16), L2 = 1e-3, early stopping

---

## Table S5. DeLong test for pairwise AUC comparisons

Pairwise comparisons of AUC between the six modeling approaches were performed using the Hanley–McNeil approximation of the DeLong test.

### Internal validation (n = 607, 67 deaths)

| Model 1 | Model 2 | ΔAUC | z | P value |
|---|---|---|---|---|
| Logistic | RandomForest | −0.0025 | −0.049 | 0.961 |
| Logistic | LightGBM | 0.0097 | 0.188 | 0.851 |
| Logistic | XGBoost | 0.0331 | 0.637 | 0.524 |
| Logistic | SVM | 0.1492 | 2.825 | 0.005 |
| Logistic | MLP | 0.2718 | 5.280 | <0.001 |
| RandomForest | LightGBM | 0.0122 | 0.237 | 0.813 |
| RandomForest | XGBoost | 0.0356 | 0.686 | 0.493 |
| RandomForest | SVM | 0.1517 | 2.876 | 0.004 |
| RandomForest | MLP | 0.2742 | 5.335 | <0.001 |
| LightGBM | XGBoost | 0.0234 | 0.449 | 0.653 |
| LightGBM | SVM | 0.1395 | 2.631 | 0.009 |
| LightGBM | MLP | 0.2621 | 5.071 | <0.001 |
| XGBoost | SVM | 0.1161 | 2.172 | 0.030 |
| XGBoost | MLP | 0.2386 | 4.578 | <0.001 |
| SVM | MLP | 0.1226 | 2.315 | 0.021 |

### External validation (n = 434, 88 deaths)

| Model 1 | Model 2 | ΔAUC | z | P value |
|---|---|---|---|---|
| Logistic | RandomForest | −0.0016 | −0.032 | 0.975 |
| Logistic | LightGBM | 0.0084 | 0.170 | 0.865 |
| Logistic | XGBoost | 0.0430 | 0.869 | 0.385 |
| Logistic | SVM | 0.1412 | 2.892 | 0.004 |
| Logistic | MLP | 0.1653 | 3.409 | <0.001 |
| RandomForest | LightGBM | 0.0100 | 0.202 | 0.840 |
| RandomForest | XGBoost | 0.0445 | 0.901 | 0.368 |
| RandomForest | SVM | 0.1427 | 2.925 | 0.003 |
| RandomForest | MLP | 0.1668 | 3.442 | <0.001 |
| LightGBM | XGBoost | 0.0346 | 0.699 | 0.485 |
| LightGBM | SVM | 0.1328 | 2.718 | 0.007 |
| LightGBM | MLP | 0.1569 | 3.234 | 0.001 |
| XGBoost | SVM | 0.0982 | 2.007 | 0.045 |
| XGBoost | MLP | 0.1223 | 2.517 | 0.012 |
| SVM | MLP | 0.0241 | 0.503 | 0.615 |

### Key interpretation

- **No significant difference** was found between logistic regression and the three best-performing tree-based models (random forest, LightGBM, XGBoost) in either internal or external validation (all P > 0.38).
- Logistic regression **significantly outperformed** SVM and MLP (all P < 0.005), the two models that performed worst.
- After Bonferroni correction for 15 pairwise comparisons (threshold P = 0.0033), only MLP remained significantly worse than logistic regression and random forest.
- **Conclusion**: Complex machine learning algorithms provided no meaningful advantage over logistic regression in this small-variable setting; simpler models with fewer parameters (SVM, MLP) performed substantially worse due to overfitting.

---

## Figure S1. Internal validation ROC and calibration curves

**Panel A**: ROC curve for the 3-variable logistic model in the internal validation subset (n = 607, 67 deaths). AUC = 0.731 (95% CI 0.670–0.797).

**Panel B**: Calibration curve for the same model. Calibration slope = 0.879; calibration intercept = −0.171, indicating good calibration in the internal validation cohort.

**Interpretation**: The model showed good discrimination (AUC 0.731) and good calibration (slope close to 1, intercept close to 0) in internal validation. In contrast, external calibration deteriorated markedly (slope 0.404, intercept −0.636), highlighting the importance of external validation.

*[Figure file: `figures/FigS1_internal_ROC_calibration.pdf`]*

---

## Figure S2. HCO₃⁻ sensitivity analysis ROC and calibration

**Panel A**: ROC curve for the HCO₃⁻-substituted model in the external validation cohort (n = 684, 109 deaths). AUC = 0.641 (95% CI 0.586–0.695).

**Panel B**: Calibration curve for the same model. Calibration slope = 0.480; calibration intercept = −0.804.

**Interpretation**: Substituting HCO₃⁻ for BE improved discrimination (AUC 0.641 vs 0.619 for BE) but calibration remained suboptimal (slope 0.480, intercept −0.804). The HCO₃⁻ model systematically overestimated mortality risk, suggesting that MIMIC-IV patients had lower observed mortality than predicted at any given risk level.

*[Figure file: `figures/FigS2_hco3_ROC_calibration.pdf`]*

---

## Figure S3. SHAP dependence plot — Albumin

SHAP dependence plot showing the relationship between albumin levels and SHAP values for the XGBoost model. Lower albumin was associated with higher predicted mortality risk.

*[Figure file: `figures/SHAP_dep_ALB.pdf`]*

---

## Figure S4. SHAP dependence plot — BUN

SHAP dependence plot showing the relationship between BUN levels and SHAP values. Higher BUN was associated with higher predicted mortality risk.

*[Figure file: `figures/SHAP_dep_BUN.pdf`]*

---

## Figure S5. SHAP dependence plot — Base excess

SHAP dependence plot showing the relationship between BE levels and SHAP values. Lower BE was associated with higher predicted mortality risk.

*[Figure file: `figures/SHAP_dep_BE.pdf`]*

---

## Appendix 1. TRIPOD+AI checklist

*[See separate file: `TRIPOD_AI_checklist.md`]*

---

## Appendix 2. PostgreSQL extraction code

The following SQL code was used to build the MIMIC-IV cohort and extract the three predictors.

### Cohort construction

```sql
SET statement_timeout = 0;
DROP TABLE IF EXISTS mimiciv_derived.cohort_mimic;
CREATE TABLE mimiciv_derived.cohort_mimic AS
WITH j96 AS (
  SELECT DISTINCT a.subject_id, a.hadm_id, a.admittime, a.dischtime,
         a.deathtime, a.hospital_expire_flag
  FROM mimiciv_hosp.admissions a
  JOIN mimiciv_hosp.diagnoses_icd d ON a.hadm_id = d.hadm_id
  WHERE d.icd_version = 10 AND d.icd_code LIKE 'J96%'
),
first_hosp AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) AS rn
  FROM j96
),
j_major AS (
  SELECT DISTINCT hadm_id FROM mimiciv_hosp.diagnoses_icd
  WHERE icd_version = 10 AND seq_num = 1 AND icd_code LIKE 'J%'
)
SELECT f.subject_id, f.hadm_id, f.admittime, f.dischtime,
       f.deathtime, f.hospital_expire_flag
FROM first_hosp f
JOIN j_major j ON f.hadm_id = j.hadm_id
JOIN mimiciv_hosp.patients p ON f.subject_id = p.subject_id
WHERE f.rn = 1
  AND p.anchor_age >= 18
  AND f.dischtime >= f.admittime + INTERVAL '24 hours'
  AND (f.deathtime IS NULL OR f.deathtime > f.admittime + INTERVAL '24 hours');
  Variable extraction (example: albumin)
  DROP TABLE IF EXISTS mimiciv_derived.cov_alb;
CREATE TABLE mimiciv_derived.cov_alb AS
SELECT DISTINCT ON (c.hadm_id)
  c.hadm_id, le.valuenum AS alb_first
FROM mimiciv_derived.cohort_mimic c
JOIN mimiciv_hosp.labevents le
  ON c.hadm_id = le.hadm_id
 AND le.itemid = 50862
 AND le.valuenum IS NOT NULL
 AND le.valuenum BETWEEN 0.5 AND 10
 AND le.charttime BETWEEN c.admittime AND c.admittime + INTERVAL '24 hours'
ORDER BY c.hadm_id, le.charttime;
Item IDs used
Variable	Item ID(s)	Unit conversion
ALB	50862	×10 (g/dL → g/L)
BUN	52647, 51006	÷2.8 (mg/dL → mmol/L)
BE	50802	—
HCO₃⁻	50882	—
Demographic extraction (for Table 1)
\copy (
  SELECT m.hadm_id, p.anchor_age AS age,
         CASE WHEN p.gender='M' THEN 1 ELSE 0 END AS male
  FROM mimiciv_derived.cohort_mimic m
  JOIN mimiciv_hosp.patients p ON m.subject_id = p.subject_id
) TO 'mimic_demo.csv' WITH CSV HEADER;
Appendix 3. Python and R analysis code
Full analysis code is available at [GitHub URL, to be filled].

Python environment
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
xgboost>=1.7
lightgbm>=3.3
shap>=0.41
matplotlib>=3.6
scipy>=1.9
Python version: 3.10

R environment
R 4.x

mice (multiple imputation)

glmnet (LASSO)

rms (model evaluation)

Random seeds
All random operations used seed = 123 for reproducibility.

Key scripts
Script	Purpose
python/run_model.py	Main ML comparison + external validation + SHAP + Fig 2
python/ci_bootstrap.py	Bootstrap 95% CI for external validation (BE)
python/sens_hco3.py	HCO₃⁻ sensitivity analysis
python/make_table1.py	Table 1 baseline characteristics
python/make_tableS2.py	Table S2 four-variable sensitivity
python/make_tableS5.py	Table S5 DeLong pairwise comparisons
python/fig1_flowchart.py	Figure 1 (flowchart)
python/figS1_internal.py	Figure S1 (internal validation)
python/figS2_hco3.py	Figure S2 (HCO₃⁻ validation)
Summary of key numerical results (for reference)
Analysis	n	Events	AUC (95% CI)	Brier (95% CI)
Training subset	1,421	159	0.7852	0.0894
Internal validation	607	67	0.7330	0.0918
External (BE)	434	88	0.6187 (0.5517–0.6815)	0.1663 (0.1417–0.1923)
External (HCO₃⁻)	684	109	0.6410 (0.5862–0.6949)	0.1367 (0.1185–0.1561)
Calibration summary
Analysis	Slope (95% CI)	Intercept (95% CI)
Internal validation	0.8782	−0.1486
External (BE)	0.4042 (0.1510–0.6609)	−0.6358 (−1.1494 to −0.1417)
External (HCO₃⁻)	0.4799 (0.2744–0.6836)	−0.8043 (−1.2086 to −0.4025)
End of Supplementary Material