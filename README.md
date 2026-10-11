# Respiratory Failure In-Hospital Mortality Risk Calculator

A 3-variable logistic regression model for predicting in-hospital mortality in adults with respiratory failure, developed at a single tertiary center in China and externally validated in MIMIC-IV v3.1.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://respiratory-failure-risk-calculator-8bdtjwjbpz4arjfskuptqp.streamlit.app)

## Overview

This repository contains the analysis code and web-based risk calculator for a study on respiratory failure prognosis. The study:

1. Developed a parsimonious 3-variable model (albumin, blood urea nitrogen, base excess) using LASSO across 50 imputed datasets (n = 2,028, 226 deaths).
2. Externally validated the model in MIMIC-IV v3.1 (n = 434, 88 deaths; AUC 0.619).
3. Quantified calibration drift (slope 0.404, intercept -0.636) and demonstrated that a single logistic recalibration restores calibration (slope 1.000, intercept 0.000) without changing discrimination.
4. Validated the recalibration benefit using 1,000-iteration bootstrap with out-of-bag evaluation.
5. Assessed clinical utility by decision curve analysis, showing higher net benefit than treat-all across threshold probabilities of 10%-30%.
6. Compared logistic regression with five machine learning algorithms (random forest, XGBoost, LightGBM, SVM, MLP); no ML algorithm outperformed logistic regression (DeLong p = 0.961).
7. Deployed an open-access web calculator with both original and recalibrated probabilities displayed.

## Web calculator

The calculator is publicly accessible at:

**https://respiratory-failure-risk-calculator-8bdtjwjbpz4arjfskuptqp.streamlit.app**

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
Model
Primary (BE version):

text
LP = 1.7933 - 0.1256 * ALB + 0.0390 * BUN - 0.0387 * BE
where ALB is in g/L, BUN in mmol/L, and BE in mmol/L.

Recalibration:

text
logit(p_new) = 0.4042 * logit(p_original) - 0.6358
Sensitivity (HCO3- version):

text
LP = 3.2028 - 0.1277 * ALB + 0.0353 * BUN - 0.0523 * HCO3_
Recalibration:

text
logit(p_new) = 0.4808 * logit(p_original) - 0.1779
Repository structure
text
.
├── app.py                   # Web calculator (Streamlit)
├── analysis/                # Analysis scripts
│   ├── run_model.py         # Primary analysis: ML comparison + external validation + SHAP
│   ├── run_model_hco3.py    # HCO3- sensitivity analysis
│   ├── recal_bootstrap.py   # Recalibration + bootstrap + DCA
│   ├── make_table1.py       # Table 1 (baseline characteristics)
│   ├── make_tableS5.py      # Table S5 (DeLong pairwise comparisons)
│   ├── fig1_flowchart.py    # Figure 1 (flowchart)
│   ├── figS1_internal.py    # Figure S1 (internal validation)
│   └── figS2_hco3.py        # Figure S2 (HCO3- sensitivity)
├── docs/
│   ├── TRIPOD_AI_checklist.md
│   └── Supplementary_Material.md
├── requirements.txt
├── LICENSE
└── README.md
Data availability
Development cohort: Not publicly available due to patient privacy regulations.

External validation cohort: MIMIC-IV v3.1 is publicly available at https://physionet.org/content/mimiciv/ after credentialing.

The analysis scripts expect local copies of:

results/traindata.csv (n = 1,421)

results/valdata.csv (n = 607)

/path/to/mimic_wide.csv (BE version)

/path/to/mimic_wide_hco3.csv (HCO3- version)

Path placeholders above should be replaced with your local data directory. These files are not included in the repository due to data use agreements.

Requirements
Python 3.10+, with:

text
streamlit>=1.30
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
shap>=0.44
matplotlib>=3.7
scipy>=1.11
Citation
If you use this code or the web calculator, please cite:

[Authors]. External validation and recalibration of a 3-variable prognostic model for in-hospital mortality in patients with respiratory failure. [Journal]. [Year]. DOI: [to be filled]

License
This project is licensed under the MIT License - see LICENSE for details.

Contact
[Corresponding author email, to be filled]

Disclaimer
This web calculator is a research tool and is not a validated clinical device. The recalibration parameters were estimated and evaluated on the same external cohort (apparent recalibration), and although bootstrap out-of-bag validation supports the benefit, prospective external validation is still required. The calculator should not be used as the sole basis for clinical decisions.

text

## 保存并提交

1. VS Code 里 `Ctrl+S` 保存
2. 回到 CMD：

```cmd
git add README.md
git commit -m "Fix app.py filename and path placeholders in README"
git push origin main