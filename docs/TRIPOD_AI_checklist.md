# TRIPOD+AI Checklist
# Prediction Model Development and Validation with Machine Learning
# 呼吸衰竭预后预测模型

**Manuscript title (working)**: Development and external validation of a 3-variable
prognostic model for in-hospital mortality in patients with respiratory failure:
a retrospective cohort study with machine learning comparison

**Version**: v1.0  
**Date**: 2026-10-10  
**Corresponding author**: [待填]

---

## Section 1: Title and Abstract

| # | Item | Reported on page | Description |
|---|---|---|---|
| 1 | Title | Title page | Identify as a prediction model study, mention target population, outcome, and external validation. |
| 2 | Abstract | Structured abstract | Report: background, objective, design (retrospective cohort), setting (single center + MIMIC-IV), participants (adult inpatients with respiratory failure), predictors (ALB, BUN, BE), outcome (in-hospital all-cause mortality), sample size (development n=2,028; external n=434), model performance (AUC 0.785 development, 0.733 internal validation, 0.619 external), and conclusions. |

---

## Section 2: Introduction

| # | Item | Reported on page | Description |
|---|---|---|---|
| 3a | Background | Introduction ¶1–2 | Respiratory failure is a leading cause of ICU admission with high mortality; early risk stratification can guide treatment intensity. |
| 3b | Objectives | Introduction ¶3 | To develop and externally validate a parsimonious prognostic model for in-hospital mortality in patients with respiratory failure, using routinely available laboratory markers, and to compare its performance with machine learning algorithms. |

---

## Section 3: Methods

### 3.1 Source of data

| # | Item | Reported on page | Description |
|---|---|---|---|
| 4a | Study design | Methods ¶1 | Retrospective cohort study. |
| 4b | Setting | Methods ¶1 | **Development cohort**: single tertiary center in China, admitted [period]. **External validation cohort**: MIMIC-IV v3.1 (Beth Israel Deaconess Medical Center, 2008–2022). |
| 4c | Dates | Methods ¶1 | Development: [admission period, e.g., 2015-01 to 2023-12]. External: MIMIC-IV v3.1 covers 2008–2022. |
| 4d | Data source | Methods ¶2 | Development: electronic health records (EHR) from [hospital name]. External: MIMIC-IV v3.1, a publicly available de-identified critical care database. |

### 3.2 Participants

| # | Item | Reported on page | Description |
|---|---|---|---|
| 5a | Eligibility criteria | Methods ¶3, Figure 1 | **Development cohort**: (1) admission diagnosis of respiratory failure (ICD-10 J96); (2) pulmonary disease as primary reason for admission (ICD-10 J-chapter primary diagnosis); (3) age ≥18 years; (4) hospital stay ≥24 h; (5) survived the first 24 h; (6) arterial blood gas analysis within 24 h of admission; (7) first hospitalization per patient. **External cohort**: (1) discharge diagnosis of J96; (2) J-chapter primary diagnosis; (3) age ≥18 years; (4) hospital stay ≥24 h; (5) survived the first 24 h; (6) first hospitalization per patient. Note: arterial blood gas was NOT required in MIMIC-IV. |
| 5b | Treatments received | Methods ¶3 | Not applicable (observational study); all patients received standard care. |
| 5c | Date of study | Methods ¶1 | See 4c. |

### 3.3 Outcome

| # | Item | Reported on page | Description |
|---|---|---|---|
| 6a | Outcome definition | Methods ¶4 | In-hospital all-cause mortality occurring between 24 h after admission and hospital discharge (landmark time = 24 h). |
| 6b | Outcome measurement | Methods ¶4 | Development: extracted from EHR discharge status. External: extracted from MIMIC-IV `admissions.hospital_expire_flag`. |

### 3.4 Predictors

| # | Item | Reported on page | Description |
|---|---|---|---|
| 7a | Predictor definition | Methods ¶5, Table 1 | **ALB** (albumin, g/L), **BUN** (blood urea nitrogen, mmol/L), **BE** (base excess, mmol/L). |
| 7b | Predictor measurement | Methods ¶5 | All measured within 24 h of admission; first value used (`DISTINCT ON (hadm_id) ORDER BY charttime`). Development: routine lab. External: MIMIC-IV `labevents` (itemid 50862 for ALB, 52647/51006 for BUN, 50802 for BE). |
| 7c | Predictor blinding | Methods ¶5 | Not applicable (retrospective; predictors extracted from routinely collected data). |
| 7d | Predictor handling | Methods ¶6 | **Continuous variables**, no dichotomization. Multiple imputation (m=50) in development cohort; complete-case analysis in external cohort. |

### 3.5 Sample size

| # | Item | Reported on page | Description |
|---|---|---|---|
| 8 | Sample size | Methods ¶7 | Development: 2,028 patients with 226 events (EPV ≈ 75 for 3 variables, sufficient). External: 434 patients with 88 events. |

### 3.6 Missing data

| # | Item | Reported on page | Description |
|---|---|---|---|
| 9 | Missing data | Methods ¶6 | Development: multiple imputation by chained equations (m=50); outcome not imputed. External: complete-case analysis (ALB coverage 27.9%, BUN 93.7%, BE 49.0%). |

### 3.7 Statistical analysis

| # | Item | Reported on page | Description |
|---|---|---|---|
| 10a | Model type | Methods ¶8 | **Primary model**: multivariable logistic regression with 3 predictors. **ML comparison**: random forest, XGBoost, LightGBM, SVM, MLP. |
| 10b | Model building | Methods ¶8 | LASSO (lambda.1se) for variable selection across 50 imputed datasets; variables with selection frequency ≥60% retained. Final 3 variables: ALB, BUN, BE. |
| 10c | Model performance | Methods ¶9 | Discrimination: AUC (95% CI by bootstrap, B=2,000). Calibration: calibration slope, calibration intercept, Brier score. |
| 10d | Model comparison | Methods ¶10 | DeLong test for AUC differences (Logistic vs best ML). |
| 10e | Model interpretation | Methods ¶10 | SHAP values for tree-based models. |
| 10f | Internal validation | Methods ¶9 | Random 7:3 split (seed=123); 100 repeated splits; median AUC reported. |
| 10g | External validation | Methods ¶9 | Applied frozen development coefficients to MIMIC-IV; no refitting. |
| 10h | Sensitivity analyses | Methods ¶11 | (1) 4-variable model (DD + ALB + BUN + BE); (2) HCO₃⁻ substituted for BE. |
| 10i | Software | Methods ¶12 | R 4.x (mice, glmnet), Python 3.10 (scikit-learn, xgboost, lightgbm, shap), PostgreSQL 18. |

---

## Section 4: Results

### 4.1 Participants

| # | Item | Reported on page | Description |
|---|---|---|---|
| 11 | Flow of participants | Results ¶1, Figure 1 | Development: 15,749 J96 records → 2,559 with J-chapter primary diagnosis → 2,559 adults → 2,456 with stay ≥24 h → 2,456 survived 24 h → **2,028 final**. External: J96 records → J-chapter primary → adults → stay ≥24 h → survived 24 h → **2,456**; complete cases for 3 predictors: **434**. |

### 4.2 Model development

| # | Item | Reported on page | Description |
|---|---|---|---|
| 12a | Predictors in final model | Results ¶2, Table 2 | ALB (β=−0.1256), BUN (β=0.0390), BE (β=−0.0387), intercept=1.7933. |
| 12b | Model performance (apparent) | Results ¶3 | Training subset AUC=0.7852, Brier=0.0894. |
| 12c | Model performance (internal) | Results ¶3 | Internal validation subset AUC=0.7330 (n=607, 67 events); calibration slope=0.8782, intercept=−0.1486. |

### 4.3 Model validation

| # | Item | Reported on page | Description |
|---|---|---|---|
| 13a | External validation | Results ¶4, Figure 2 | MIMIC-IV: n=434, 88 deaths. AUC=0.619 (95% CI 0.552–0.682); Brier=0.166 (95% CI 0.142–0.192); calibration slope=0.404 (95% CI 0.151–0.661); calibration intercept=−0.636 (95% CI −1.149 to −0.142). |
| 13b | Sensitivity (HCO₃⁻) | Results ¶5 | MIMIC-IV: n=684, 109 deaths. AUC=0.641 (95% CI 0.586–0.695); Brier=0.137 (95% CI 0.119–0.156); calibration slope=0.480 (95% CI 0.274–0.684); intercept=−0.804 (95% CI −1.209 to −0.403). |
| 13c | ML comparison | Results ¶6, Table 3 | In internal validation (n=607): RF=0.734, Logistic=0.732, LGBM=0.722, XGB=0.698, SVM=0.582, MLP=0.460. In external validation (n=434): RF=0.620, Logistic=0.619, LGBM=0.610, XGB=0.576, SVM=0.478, MLP=0.453. DeLong: Logistic vs RF z=−0.049, p=0.961. |
| 13d | SHAP | Results ¶7, Figure 3 | ALB contributed the largest SHAP magnitude; direction consistent with logistic model (lower ALB/BE → higher risk; higher BUN → higher risk). |

---

## Section 5: Discussion

| # | Item | Reported on page | Description |
|---|---|---|---|
| 14 | Interpretation | Discussion ¶1–3 | 3-variable model has acceptable discrimination (AUC 0.62 external) but suboptimal calibration (slope 0.40, intercept −0.64). HCO₃⁻ substitution improves discrimination (AUC 0.64) and accuracy (Brier 0.137). ML offers no advantage over logistic (DeLong p=0.961). |
| 15 | Limitations | Discussion ¶4 | (1) ALB coverage only 27.9% in MIMIC-IV → complete-case selection bias; (2) MIMIC-IV did not require arterial blood gas, widening case-mix; (3) BE definitions/populations may differ; (4) single-center development → potential overfitting; (5) retrospective design. |
| 16 | Implications | Discussion ¶5 | The model may be useful for early risk stratification, but its suboptimal calibration requires recalibration before clinical use. HCO₃⁻ may be a more robust acid-base marker than BE. |

---

## Section 6: Other Information

| # | Item | Reported on page | Description |
|---|---|---|---|
| 17 | Registration | Methods ¶1 | Not registered (retrospective study). |
| 18 | Data availability | Data availability statement | Development data not publicly available (privacy). MIMIC-IV available at https://physionet.org/content/mimiciv/. |
| 19 | Code availability | Data availability statement | Analysis code available at [GitHub URL, to be filled]. |
| 20 | Funding | Funding statement | [待填] |
| 21 | Conflicts of interest | COI statement | None declared. |
| 22 | Patient consent | Ethics statement | Development cohort: waived by IRB due to retrospective nature. MIMIC-IV: de-identified, no consent required. |
| 23 | Ethics approval | Ethics statement | [IRB number, to be filled]. |

---

## Section 7: TRIPOD+AI Specific Items

| # | Item | Reported on page | Description |
|---|---|---|---|
| AI-1 | Data type | Methods ¶2 | Structured tabular data (laboratory values). Not image/text/signal. |
| AI-2 | Data preprocessing | Methods ¶6 | z-score standardization for logistic/SVM/MLP; tree models used raw values. |
| AI-3 | Model architecture | Methods ¶8 | Logistic regression (linear, no hidden layers); RF (500 trees, max_depth=6); XGBoost (1000 trees, lr=0.03, depth=3); LightGBM (same); SVM (RBF kernel); MLP (32-16 hidden, early stopping). |
| AI-4 | Hyperparameter tuning | Methods ¶8 | Pre-specified; no extensive tuning to avoid overfitting. XGBoost/LightGBM used early stopping on validation set. |
| AI-5 | Handling class imbalance | Methods ¶8 | No resampling; class weights not adjusted. Event rate 11% (development), 20% (external). |
| AI-6 | Validation strategy | Methods ¶9 | 7:3 random split (seed=123); 100 repeated splits; external validation on MIMIC-IV without refitting. |
| AI-7 | Performance metrics | Methods ¶9 | AUC, Brier, calibration slope/intercept. |
| AI-8 | Confidence intervals | Methods ¶9 | Bootstrap (B=2,000) for external metrics. |
| AI-9 | Model interpretability | Methods ¶10 | SHAP (TreeExplainer) for tree models. |
| AI-10 | Fairness/bias assessment | Discussion ¶4 | Not performed; acknowledged as limitation. |
| AI-11 | Reproducibility | Data availability statement | Fixed random seeds (123); code and hyperparameters reported. |
| AI-12 | Computational resources | Methods ¶12 | Standard laptop (Windows 10, Python 3.10, R 4.x). No GPU used. |
| AI-13 | Comparison with non-AI model | Results ¶6 | Yes: logistic regression as baseline. |
| AI-14 | Clinical utility | Discussion ¶5 | Not formally assessed (no decision curve analysis); acknowledged as future work. |
| AI-15 | Model updating | Discussion ¶5 | Not performed; recalibration suggested before clinical use. |

---

## Section 8: Key Numbers Summary (for reference)

### Development cohort (n=2,028)
- Events: 226 (11.1%)
- Training subset: n=1,421, events=159
- Internal validation subset: n=607, events=67
- Training AUC: 0.7852
- Internal validation AUC: 0.7330
- 100-split median AUC: 0.7675
- Calibration slope: 0.8782
- Calibration intercept: −0.1486

### External validation (MIMIC-IV)
**BE version (primary)**
- n=434, events=88 (20.3%)
- AUC: 0.619 (95% CI 0.552–0.682)
- Brier: 0.166 (95% CI 0.142–0.192)
- Calibration slope: 0.404 (95% CI 0.151–0.661)
- Calibration intercept: −0.636 (95% CI −1.149 to −0.142)

**HCO₃⁻ version (sensitivity)**
- n=684, events=109 (15.9%)
- AUC: 0.641 (95% CI 0.586–0.695)
- Brier: 0.137 (95% CI 0.119–0.156)
- Calibration slope: 0.480 (95% CI 0.274–0.684)
- Calibration intercept: −0.804 (95% CI −1.209 to −0.403)

### ML comparison (external, n=434)
| Model | AUC | Brier |
|---|---|---|
| RandomForest | 0.6203 | 0.1627 |
| Logistic | 0.6187 | 0.1663 |
| LightGBM | 0.6103 | 0.1649 |
| XGBoost | 0.5757 | 0.1834 |
| SVM | 0.4775 | 0.1704 |
| MLP | 0.4534 | 0.1979 |

**DeLong (Logistic vs RF)**: z=−0.049, p=0.9613

---

## Section 9: Checklist Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Corresponding author | [待填] | | |
| Statistician | [待填] | | |
| Data manager | [待填] | | |

---

**End of TRIPOD+AI Checklist**