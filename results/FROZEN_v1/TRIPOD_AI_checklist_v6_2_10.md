# TRIPOD+AI Checklist
# Prediction Model Development and Validation with Machine Learning
# Respiratory Failure Prognostic Model (eICU-CRD Plan B)

**Manuscript title**: External validation and recalibration of a 3-variable prognostic model for in-hospital mortality in patients with respiratory failure: calibration drift, machine learning comparison, and a web-based risk calculator

**Version**: v6.2.10 (frozen 2026-11-04)
**Date**: 2026-11-04

---

## Section 1: Title and Abstract

| # | Item | Reported on page | Description |
|---|---|---|---|
| 1 | Title | Title page | Identifies as a prediction model study; mentions target population, outcome, external validation, and recalibration. |
| 2 | Abstract | Structured abstract | Background, objective, design, setting (China, MIMIC-IV, eICU-CRD), participants, predictors, outcome (in-hospital mortality after 24-h landmark), sample size (dev n=2,028; MIMIC n=434 BE / 684 HCO₃⁻; eICU n=1,458 BE / 2,287 HCO₃⁻), performance (internal validation AUC 0.727, SD 0.006 across 50 imputations; external AUC 0.620 BE and 0.642 HCO₃⁻ in MIMIC-IV; 0.670 BE and 0.670 HCO₃⁻ in eICU-CRD; calibration slopes 0.410 BE and 0.599 BE), and conclusions. |

## Section 2: Introduction

| # | Item | Reported on page | Description |
|---|---|---|---|
| 3a | Background | Introduction ¶1–2 | Respiratory failure is a leading cause of ICU admission with high mortality; calibration drift after transport is common and under-reported. |
| 3b | Objectives | Introduction ¶3–5 | Five objectives: (1) develop a parsimonious model; (2) externally validate in MIMIC-IV and eICU-CRD; (3) quantify calibration drift and evaluate two-parameter logistic recalibration with bootstrap OOB; (4) compare logistic regression with five ML algorithms and assess DCA net benefit; (5) deploy a research-use web calculator. |

## Section 3: Methods

### 3.1 Source of data

| # | Item | Reported on page | Description |
|---|---|---|---|
| 4a | Study design | Methods §Study design | Retrospective cohort study with one development and two external cohorts. |
| 4b | Setting | Methods §Study design | Development: single tertiary academic medical center in China (ICU and Respiratory Department). External 1: MIMIC-IV v3.1. External 2: eICU-CRD v2.0 (multicenter, US). |
| 4c | Dates | Methods §Study design | Development: October 2022 – June 2026. MIMIC-IV: 2008–2022. eICU-CRD v2.0: per database release. |
| 4d | Data source | Methods §Study design | Development: EHR from Anhui No. 2 Provincial People's Hospital. MIMIC-IV and eICU-CRD: PhysioNet de-identified databases. |

### 3.2 Participants

| # | Item | Reported on page | Description |
|---|---|---|---|
| 5a | Eligibility criteria | Methods §Participants; Figure 1 | Development and MIMIC-IV: ICD-10 J96 in discharge diagnoses; J-chapter primary diagnosis; age ≥18; hospital stay ≥24 h; survival of first 24 h; first hospitalization; ABG within 24 h (development only). eICU-CRD: ICD-9 518.81 primary diagnosis; prediction time anchored to ICU admission (24 h); same age, ICU 24-h survival, first-ICU-stay criteria. |
| 5b | Treatments received | Methods §Participants | Not applicable (observational); standard care. |
| 5c | Date of study | Methods §Study design | See 4c. |

### 3.3 Outcome

| # | Item | Reported on page | Description |
|---|---|---|---|
| 6a | Outcome definition | Methods §Outcome | In-hospital all-cause mortality occurring after the 24-h landmark and before discharge. |
| 6b | Outcome measurement | Methods §Outcome | Development: discharge record. MIMIC-IV: `hospital_expire_flag`. eICU-CRD: `hospitaldischargestatus`. |

### 3.4 Predictors

| # | Item | Reported on page | Description |
|---|---|---|---|
| 7a | Predictor definition | Methods §Predictors | ALB (g/L), BUN (mmol/L), BE (mmol/L); HCO₃⁻ in the sensitivity analysis. |
| 7b | Predictor measurement | Methods §Predictors | First value within 24 h of the respective time origin. MIMIC-IV item IDs: ALB 50862 (×10), BUN 52647/51006 (÷2.8), BE 50802, HCO₃⁻ 50882. eICU-CRD: first 24 h after ICU admission. |
| 7c | Predictor blinding | Methods §Predictors | Not applicable (retrospective). |
| 7d | Predictor handling | Methods §Predictors | Continuous, no dichotomization. Development: multiple imputation (m=50). External: complete-case analysis. |

### 3.5 Sample size

| # | Item | Reported on page | Description |
|---|---|---|---|
| 8 | Sample size | Methods §Sample size | Development 2,028 (226 events; EPV ≈ 75). MIMIC-IV 434 BE complete-case (88 events) and 684 HCO₃⁻ complete-case (109 events). eICU-CRD 1,458 BE complete-case (363 events) and 2,287 HCO₃⁻ complete-case (521 events). |

### 3.6 Missing data

| # | Item | Reported on page | Description |
|---|---|---|---|
| 9 | Missing data | Methods §Missing data | Development: MICE (m=50), outcome excluded, applied separately within training and validation subsets. External: complete-case analysis. MIMIC-IV coverage: ALB 27.9%, BUN 93.7%, BE 49.0%. eICU-CRD predictor coverage was higher than in MIMIC-IV for BUN and HCO₃⁻ (Tables S11–S13). |

### 3.7 Statistical analysis

| # | Item | Reported on page | Description |
|---|---|---|---|
| 10a | Model type | Methods §Model development | Primary: 3-variable logistic regression (ALB, BUN, BE). Sensitivity: HCO₃⁻ substitution. ML comparison: RF, XGBoost, LightGBM, SVM, MLP. |
| 10b | Model building | Methods §Variable selection | LASSO (λ.1se) across 50 imputed datasets; variables selected in ≥60% of datasets were retained: ALB 100%, BUN 100%, BE 100%, DD 100%, APTT 100%, WBC 98%. The final three-variable model (ALB, BUN, BE) was prespecified on the basis of clinical availability and parsimony; DD and APTT were not included because they are not consistently measured at the 24-h landmark in MIMIC-IV or eICU-CRD. Final BE model: LP = 1.782535 − 0.122710·ALB + 0.034275·BUN − 0.058505·BE. Final HCO₃⁻ model: LP = 3.193555 − 0.120922·ALB + 0.037409·BUN − 0.060383·HCO₃⁻. |
| 10c | Model performance | Methods §Model performance | AUC (bootstrap B=2,000; seed 20261101; outcome-stratified percentile), Brier score, calibration slope and intercept estimated jointly from a logistic model of the observed outcome on the linear predictor. |
| 10d | Model comparison | Methods §Machine learning comparison | DeLong test; five ML algorithms on the same three predictors. |
| 10e | Model interpretation | Methods §Model interpretability | SHAP TreeExplainer on XGBoost. |
| 10f | Internal validation | Methods §Internal validation procedure | 7:3 split on raw data before imputation, seed 123. Frozen model applied to 50 imputed internal-validation datasets. Reported: 50-imputation mean (primary stability estimate) and 100-split median (secondary stability check). |
| 10g | External validation | Methods §External validation | Frozen BE coefficients applied without refitting to MIMIC-IV and eICU-CRD. |
| 10h | Recalibration | Methods §Recalibration | Two-parameter logistic recalibration `logit(p_new) = a·logit(p_original) + b` fitted separately in each external cohort; stability evaluated with 1,000-iteration bootstrap OOB. |
| 10i | Decision curve analysis | Methods §Decision curve analysis | Net benefit across thresholds 0.01–0.50; exploratory range 10%–30%. |
| 10j | Sensitivity analyses | Methods §Sensitivity analyses | (1) HCO₃⁻ substitution; (2) ALB fixed at the development-cohort mean (34.44 g/L). |
| 10k | Software | Methods §Statistical software | R 4.5.1 (mice, glmnet, rms), Python 3.10 (scikit-learn, xgboost, lightgbm, shap), PostgreSQL 18. |

## Section 4: Results

### 4.1 Participants

| # | Item | Reported on page | Description |
|---|---|---|---|
| 11 | Flow of participants | Results §Participant flow; Figure 1 | Development: 4,628 initial → 2,028 final (226 deaths). 7:3 split: training 1,420 (154 deaths), internal validation 608 (72 deaths). MIMIC-IV: 20,780 J96 admissions → 2,456 final; 434 BE complete-case (88 deaths); 684 HCO₃⁻ complete-case (109 deaths). eICU-CRD: 200,859 ICU stays → 4,696 eligible (977 deaths) → 4,693 after BE range filtering and 4,691 after HCO₃⁻ range filtering → 1,458 BE complete-case (363 deaths) and 2,287 HCO₃⁻ complete-case (521 deaths). |

### 4.2 Model development

| # | Item | Reported on page | Description |
|---|---|---|---|
| 12a | Predictors in final model | Results §Model development; Table 2 | Intercept 1.7825; ALB −0.1227 (OR 0.885); BUN 0.0343 (OR 1.035); BE −0.0585 (OR 0.943). |
| 12b | Model performance (apparent) | Results §Model performance | Training AUC 0.767 (SD 0.005 across 50 imputations); Brier 0.088 (SD 0.0004). |
| 12c | Model performance (internal) | Results §Model performance; Table 3; Figure S1 | Internal validation (n=608, 72 events), 50-imputation mean: AUC 0.727 (SD 0.006); Brier 0.098 (SD 0.001); calibration slope 0.797 (Q25–Q75 0.781–0.819); calibration intercept −0.356 (Q25–Q75 −0.385 to −0.316). 100-split median: AUC 0.751 (IQR 0.731–0.772). |

### 4.3 Model validation

| # | Item | Reported on page | Description |
|---|---|---|---|
| 13a | External validation (MIMIC-IV, BE) | Results §External validation in MIMIC-IV; Figure 2 | n=434, 88 deaths. AUC 0.620 (95% CI 0.555–0.681); Brier 0.166 (0.155–0.178); slope 0.410; intercept −0.637. |
| 13b | Recalibration (MIMIC-IV, BE) | Results §Logistic recalibration; Table 5 | a=0.999, b=−0.001. Apparent Brier 0.158; OOB median Brier 0.1596 [IQR 0.1492–0.1696], slope 1.0075 [0.6090–1.5430], intercept −0.0135 [−0.5393–0.7927]. |
| 13c | External validation (MIMIC-IV, HCO₃⁻) | Results §Sensitivity analysis with HCO₃⁻; Figure S2; Table S2 | n=684, 109 deaths. AUC 0.642 (95% CI 0.588–0.700); Brier 0.137 (0.129–0.145); slope 0.485; intercept −0.809. Recalibration a=1.002, b=0.004; apparent Brier 0.131; OOB median Brier 0.1321 [0.1240–0.1403], slope 1.0276 [0.7398–1.4018], intercept 0.0330 [−0.4308–0.6338]. |
| 13d | External validation (eICU-CRD, BE) | Results §eICU-CRD external validation; Figure S6 | n=1,458, 363 deaths. AUC 0.670 (95% CI 0.638–0.703); Brier 0.179 (0.167–0.190); slope 0.599; intercept −0.486. Recalibration a=1.000, b=0.000; apparent Brier 0.174; OOB median Brier 0.1739 [0.1692–0.1794], slope 1.0003 [0.8566–1.1570], intercept −0.0046 [−0.1718–0.1749]. |
| 13e | External validation (eICU-CRD, HCO₃⁻) | Results §eICU-CRD external validation; Figure S7 | n=2,287, 521 deaths. AUC 0.670 (95% CI 0.643–0.696); Brier 0.171 (0.162–0.180); slope 0.619; intercept −0.591. Recalibration a=1.000, b=0.000; apparent Brier 0.164; OOB median Brier 0.1638 [0.1598–0.1681], slope 0.9895 [0.8790–1.1437], intercept −0.0005 [−0.1486–0.1647]. |
| 13f | Decision curve analysis | Results §Decision curve analysis; Tables 6a/6b; Figures 3, S8 | BE in MIMIC-IV: net benefit > treat-all at 10%–30% for the recalibrated model. BE in eICU-CRD: exceeded treat-all at 15%–30% (at 15%, marginally lower: 0.1163 vs 0.1164; at 10%, 0.1630 vs 0.1655). HCO₃⁻ recalibration: exceeded treat-all only at 10% in MIMIC-IV (0.0713 vs 0.0660) and at 15%–30% in eICU-CRD. |
| 13g | ML comparison | Results §Machine learning comparison; Table 4 | Internal validation (n=608): Random forest 0.748, Logistic 0.744, XGBoost 0.742, LightGBM 0.720, SVM 0.587, MLP 0.445. External (MIMIC-IV, n=434): Python Logistic 0.620, Random forest 0.609, XGBoost 0.603, LightGBM 0.587, SVM 0.502, MLP 0.439. No ML algorithm exceeded Python logistic regression externally. |
| 13h | Severity-score benchmark | Results §Supplementary comparison; Table S14 | MIMIC-IV BE: 3-variable 0.613 vs SOFA 0.638 (n=393, P=0.485); HCO₃⁻: 0.612 vs 0.642 (n=467, P=0.394). eICU-CRD BE: 3-variable 0.671 vs APACHE IVa 0.727 (n=1,136, ΔAUC −0.056, 95% CI −0.098 to −0.015, P=0.009); HCO₃⁻: 0.678 vs 0.739 (n=1,722, ΔAUC −0.062, 95% CI −0.096 to −0.028, P<0.001). APACHE IVa achieved significantly higher AUCs than both three-variable models. |
| 13i | SHAP | Results §SHAP interpretability; Figures S3/S3b | ALB shows the largest mean |SHAP|, BE intermediate, BUN smallest. Directions consistent with logistic coefficients: higher ALB and BE associated with lower risk, higher BUN with higher risk. |

## Section 5: Discussion

| # | Item | Reported on page | Description |
|---|---|---|---|
| 14 | Interpretation | Discussion §Summary; §Comparison | Moderate external discrimination (AUC 0.620 BE and 0.642 HCO₃⁻ in MIMIC-IV; 0.670 BE and 0.670 HCO₃⁻ in eICU-CRD) with substantial calibration drift (slopes 0.410 BE and 0.599 BE in MIMIC-IV and eICU-CRD, respectively). Two-parameter recalibration improved apparent Brier; bootstrap OOB provided internal support. APACHE IVa significantly outperformed both three-variable models in eICU-CRD, confirming that parsimonious models do not replace established ICU severity scores. |
| 15 | Limitations | Discussion §Limitations | Single-center retrospective development; complete-case selection in external cohorts; recalibration not independently validated; DCA not OOB-validated; differing time origins in eICU-CRD (ICU admission rather than hospital admission); HCO₃⁻ analysis is a variable-substitution sensitivity analysis; APACHE IVa (a comprehensive ICU severity score with >20 physiological and laboratory variables) achieved significantly higher AUCs than the three-variable model; DD and APTT retained by LASSO were not carried forward because they are not consistently measured at the 24-h landmark in external cohorts. |
| 16 | Implications | Discussion §Clinical practice; §Research | Calibration should be assessed when transporting models; local recalibration may be considered; the web calculator is a research-use tool; the model should be interpreted as a parsimonious initial risk stratification tool rather than a replacement for established ICU severity scores. |

## Section 6: Other Information

| # | Item | Reported on page | Description |
|---|---|---|---|
| 17 | Registration | Methods §Study design | Not registered (retrospective study). |
| 18 | Data availability | Data availability statement | Development data not publicly available. MIMIC-IV: https://physionet.org/content/mimiciv/. eICU-CRD: https://physionet.org/content/eicu-crd/. |
| 19 | Code availability | Data availability statement | GitHub: https://github.com/wj-202606/respiratory-failure-risk-calculator. Concept DOI: 10.5281/zenodo.23300381. Latest version: 10.5281/zenodo.23734774 (v1.3.1, 2026-10-28), to be updated to v6.2.10 at submission. |
| 20 | Funding | Funding statement | No funding received for this study. |
| 21 | Conflicts of interest | COI statement | None declared. |
| 22 | Patient consent | Ethics statement | Development: informed consent waived by IRB. MIMIC-IV and eICU-CRD: de-identified, no additional consent required. |
| 23 | Ethics approval | Ethics statement | Development: [IRB number]. MIMIC-IV: BIDMC/MIT IRB. eICU-CRD: per PhysioNet access agreement. |

## Section 7: TRIPOD+AI Specific Items

| # | Item | Reported on page | Description |
|---|---|---|---|
| AI-1 | Data type | Methods §Predictors | Structured tabular data (laboratory values). |
| AI-2 | Data preprocessing | Methods §Machine learning comparison | z-score for logistic/SVM/MLP; raw values for tree models. |
| AI-3 | Model architecture | Methods §Machine learning comparison | Logistic; RF (500 trees, depth 6); XGBoost (1,000 trees, lr 0.03, depth 3); LightGBM (1,000 trees, lr 0.03, depth 3, 8 leaves); SVM (RBF, C=1.0); MLP (32-16, L2=1e-3). |
| AI-4 | Hyperparameter tuning | Methods §Machine learning comparison | Pre-specified; early stopping on the internal-validation split. |
| AI-5 | Class imbalance | Methods §Machine learning comparison | No resampling; event rates 11.1% (development), 20.3% (MIMIC-IV BE), 24.9% (eICU-CRD BE), 22.8% (eICU-CRD HCO₃⁻). |
| AI-6 | Validation strategy | Methods §Internal validation procedure; §External validation | 7:3 split (seed 123) before imputation; 100-split stability check; frozen coefficients applied to MIMIC-IV and eICU-CRD. |
| AI-7 | Performance metrics | Methods §Model performance | AUC, Brier score, calibration slope, calibration intercept. |
| AI-8 | Confidence intervals | Methods §Model performance | Bootstrap B=2,000 (outcome-stratified percentile; seed 20261101) for AUC/Brier; B=1,000 for OOB recalibration. |
| AI-9 | Interpretability | Methods §Model interpretability | SHAP TreeExplainer on XGBoost. |
| AI-10 | Fairness/bias assessment | Discussion §Limitations | Not performed due to limited sample size. |
| AI-11 | Reproducibility | Data availability statement | Fixed seeds; scripts and hyperparameters on GitHub; MIMIC-IV and eICU-CRD are publicly available through PhysioNet. |
| AI-12 | Computational resources | Methods §Statistical software | Standard laptop; R 4.5.1 and Python 3.10; no GPU required. |
| AI-13 | Comparison with non-AI model | Results §Machine learning comparison; Table 4; Table S4 | Logistic regression as baseline; no ML algorithm outperformed it externally (Python Logistic external AUC 0.620; RF 0.609; XGBoost 0.603; LightGBM 0.587). |
| AI-14 | Clinical utility | Results §Decision curve analysis; Tables 6a/6b | DCA performed; net benefit is apparent and exploratory. |
| AI-15 | Model updating | Methods §Recalibration; Results §Logistic recalibration | Two-parameter logistic recalibration with bootstrap OOB evaluation; independent external validation of the recalibrated model remains necessary. |

## Section 8: Key Numbers Summary (v6.2.10)

**Development cohort**
- Training subset (n=1,420, 154 events): AUC 0.767 (SD 0.005 across 50 imputations); Brier 0.088 (SD 0.0004).
- Internal validation (n=608, 72 events), 50-imputation mean: AUC 0.727 (SD 0.006); Brier 0.098 (SD 0.001); calibration slope 0.797 (Q25–Q75 0.781–0.819); calibration intercept −0.356 (Q25–Q75 −0.385 to −0.316).
- 100-split median: AUC 0.751 (IQR 0.731–0.772).

**MIMIC-IV (BE)**: n=434, events=88. AUC 0.620 (95% CI 0.555–0.681); Brier 0.166 (0.155–0.178); calibration slope 0.410; intercept −0.637. Recalibration a=0.999, b=−0.001; apparent Brier 0.158; OOB Brier 0.1596 [0.1492–0.1696], slope 1.0075 [0.6090–1.5430], intercept −0.0135 [−0.5393–0.7927].

**MIMIC-IV (HCO₃⁻)**: n=684, events=109. AUC 0.642 (95% CI 0.588–0.700); Brier 0.137 (0.129–0.145); calibration slope 0.485; intercept −0.809. Recalibration a=1.002, b=0.004; apparent Brier 0.131; OOB Brier 0.1321 [0.1240–0.1403], slope 1.0276 [0.7398–1.4018], intercept 0.0330 [−0.4308–0.6338].

**eICU-CRD (BE, ICU-admission time origin)**: n=1,458, events=363. AUC 0.670 (95% CI 0.638–0.703); Brier 0.179 (0.167–0.190); calibration slope 0.599; intercept −0.486. Recalibration a=1.000, b=0.000; apparent Brier 0.174; OOB Brier 0.1739 [0.1692–0.1794], slope 1.0003 [0.8566–1.1570], intercept −0.0046 [−0.1718–0.1749].

**eICU-CRD (HCO₃⁻, ICU-admission time origin)**: n=2,287, events=521. AUC 0.670 (95% CI 0.643–0.696); Brier 0.171 (0.162–0.180); calibration slope 0.619; intercept −0.591. Recalibration a=1.000, b=0.000; apparent Brier 0.164; OOB Brier 0.1638 [0.1598–0.1681], slope 0.9895 [0.8790–1.1437], intercept −0.0005 [−0.1486–0.1647].

**SOFA comparison (MIMIC-IV)**:
- BE: 3-variable 0.613 (95% CI 0.547–0.680) vs SOFA 0.638 (0.570–0.705); ΔAUC −0.025 (95% CI −0.092 to 0.044); DeLong P=0.485; n=393.
- HCO₃⁻: 3-variable 0.612 (0.551–0.671) vs SOFA 0.642 (0.580–0.701); ΔAUC −0.029 (95% CI −0.102 to 0.035); DeLong P=0.394; n=467.

**APACHE IVa comparison (eICU-CRD)**:
- BE: 3-variable 0.671 (95% CI 0.633–0.709) vs APACHE IVa 0.727 (0.693–0.761); ΔAUC −0.056 (95% CI −0.098 to −0.015); DeLong P=0.009; n=1,136.
- HCO₃⁻: 3-variable 0.678 (0.647–0.708) vs APACHE IVa 0.739 (0.711–0.767); ΔAUC −0.062 (95% CI −0.096 to −0.028); DeLong P<0.001; n=1,722.

**Machine learning comparison (Table 4)**:

| Model | Internal AUC | External AUC |
|---|---|---|
| Logistic regression (Python) | 0.744 | 0.620 |
| Random forest | 0.748 | 0.609 |
| XGBoost | 0.742 | 0.603 |
| LightGBM | 0.720 | 0.587 |
| SVM | 0.587 | 0.502 |
| MLP | 0.445 | 0.439 |

**LASSO variable selection (50 imputed datasets, λ.1se)**:
- Selected in 100%: ALB, BUN, BE, DD, APTT.
- WBC 98%; Na 56%; Age 12%; HCO₃⁻ 2%.
- Final model: ALB + BUN + BE (prespecified on clinical availability and parsimony).

---

**End of TRIPOD+AI Checklist**