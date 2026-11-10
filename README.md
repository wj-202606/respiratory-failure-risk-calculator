# AIC Respiratory Failure Prediction Model — Analysis Pipeline

**Version**: v6.2.10 (frozen 2026-11-04)
**Manuscript**: External validation and recalibration of a 3-variable prognostic model for in-hospital mortality in patients with respiratory failure: calibration drift, machine learning comparison, and a web-based risk calculator
**Target journal**: AIC

---

## Overview

A parsimonious 3-variable logistic regression model (ALB + BUN + BE) for
in-hospital mortality in patients with respiratory failure, developed in a
single-center Chinese cohort and externally validated in two large public
databases (MIMIC-IV and eICU-CRD).

A prespecified sensitivity model substitutes bicarbonate (HCO₃⁻) for base
excess (BE). A two-parameter logistic recalibration was fitted separately in
each external cohort. The model is deployed as a research-use web calculator.

---

## Environment

| Component | Version |
|---|---|
| R | 4.5.1 |
| Python | 3.10.4 |
| PostgreSQL | 18 |
| OS | Windows 11 |

R packages: `naniar`, `mice`, `tableone`, `ggplot2`, `lattice`, `caret`,
`autoReg`, `dplyr`, `Matrix`, `glmnet`, `Boruta`, `Hmisc`, `rms`, `car`,
`pROC`, `e1071`, `boot`, `dcurves`.

Python packages: see `requirements_v6.2.txt`.

---

## Directory structure
.
├── R_analysis_v5_4_1.R # Development cohort: MICE, LASSO, Rubin
├── run_model.py # MIMIC predictions + 5 ML + SHAP
├── make_external_stats.py # Single source of truth for external CIs
├── fig2_be.py # Figure 2 (BE)
├── figS2_hco3.py # Figure S2 (HCO3)
├── figS1_internal.py # Figure S1 (internal)
├── recal_bootstrap.py # Recalibration + OOB + DCA
├── run_mimic_sofa_all.py # SOFA comparison
├── eicu_build_wide.py # eICU cohort construction
├── eicu_validate.py # eICU external validation
├── compare_eicu_apache.py # APACHE IVa comparison
├── make_eicu_supp_tables.py # Supplementary eICU tables
├── make_tables.py # Main tables + Table S14
├── make_table1.py # Table 1 (rebuilt)
├── make_table2.py # Table 2 (LASSO + coefficients)
├── make_tableS2.py # Table S2 (HCO3 sensitivity)
├── make_tableS4.py # Table S4 (DeLong pairs)
├── make_tableS5_representative.py # Table S5 (published models)
├── make_tableS7a.py # Table S7a (dev vs MIMIC)
├── make_tableS8.py # Table S8 (MIMIC incl/excl)
├── alb_missing_analysis.py # Table S9 (MIMIC ALB missing)
├── two_var_model.py # Table S10 (ALB fixed)
├── make_closure_report.py # HCO3 source closure report
├── make_submission_audit.py # Submission audit package
├── check_v6_2_1.py # Verification script
├── ci_config.py # Unified bootstrap parameters
├── shared_calib.py # Calibration/OOB/bootstrap utilities
├── run_pipeline_v6_2_10.ps1 # Full pipeline driver
├── requirements_v6.2.txt # Python dependencies
├── VERSION_v6_2_10.txt # Full version history
├── README.md # This file
├── Supplementary_v6_2_10.md # Supplementary Material
├── TRIPOD_AI_checklist_v6_2_10.md # TRIPOD+AI checklist
├── results/ # Frozen analysis outputs
│ └── FROZEN_v1/ # Final frozen snapshot (122 active files)
├── tables/ # Formatted tables (Table1-6b, S2-S14)
├── 论文图/ # Publication figures (Fig1-3, S1-S8)
└── submission_audit/ # Submission audit package (10 files)

text

---

## Data sources (not redistributed)

| Cohort | Source | Access |
|---|---|---|
| Development | Anhui No. 2 Provincial People's Hospital | Not publicly available |
| MIMIC-IV v3.1 | PhysioNet | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD v2.0 | PhysioNet | https://physionet.org/content/eicu-crd/2.0/ |

Both MIMIC-IV and eICU-CRD are de-identified public databases requiring
credentialed PhysioNet access. Raw patient-level data are not redistributed
with this repository.

### MIMIC-IV wide table

- Path: `D:\mimic_wide_with_hco3.csv`
- Size: 2,456 rows × 8 columns (684 rows with HCO₃⁻)
- SHA256: `b151612f75c568fce52f5a472206608682daaecd1a8ae2302595c45a0b3a72eb`

### eICU-CRD APACHE IVa

- Path: `D:\eicu2.0\apachePatientResult.csv.gz`
- Source: eICU-CRD v2.0 PhysioNet release
- Not redistributed; SHA256 recorded in `results/apache_input_provenance.csv`

---

## Run order (v6.2.10)
R_analysis_v5_4_1.R (RUN_100_SPLITS = FALSE)

eicu_build_wide.py

eicu_validate.py

compare_eicu_apache.py

run_model.py

make_external_stats.py <- must run before fig2_be / figS2 / make_tables

fig2_be.py

figS2_hco3.py

recal_bootstrap.py

run_mimic_sofa_all.py

make_eicu_supp_tables.py

make_tables.py

check_v6_2_1.py

make_closure_report.py

text

Steps 1–4 can be skipped when `results/` already contains valid
`external_validation_coef*.csv`, `eicu_*.csv`, and `apache_*.csv`.

Steps 5–14 are the minimum required to regenerate predictions, statistics,
Figure 2, Figure S2, Table 3, and the closure report.

### One-shot PowerShell driver

```powershell
.\run_pipeline_v6_2_10.ps1 -SkipSofa -SkipClosureReport
Or without skipping (re-runs SOFA comparison + generates the closure
report in the same manifest):

powershell
.\run_pipeline_v6_2_10.ps1
Frozen reference values (v6.2.10)
Model	n	events	AUC (95% CI)	Brier (95% CI)	slope	intercept
MIMIC-IV BE	434	88	0.620 (0.555–0.681)	0.166 (0.155–0.178)	0.410	−0.637
MIMIC-IV HCO₃⁻	684	109	0.642 (0.588–0.700)	0.137 (0.129–0.145)	0.485	−0.809
eICU-CRD BE	1,458	363	0.670 (0.638–0.703)	0.179 (0.167–0.190)	0.599	−0.486
eICU-CRD HCO₃⁻	2,287	521	0.670 (0.643–0.696)	0.171 (0.162–0.180)	0.619	−0.591
Additional reference values:

Internal validation AUC: 0.727 (SD 0.006 across 50 imputations)

Internal calibration slope: 0.797; intercept: −0.356

100-split median AUC: 0.751 (IQR 0.731–0.772)

100-split direction audit: 5,000 AUC calculations, 0 flips

Bootstrap: outcome-stratified percentile, B = 2,000, seed = 20261101

Key deliverables
File	Purpose
VERSION_v6_2_10.txt	Full version history and fix log
Supplementary_v6_2_10.md	Supplementary Material (Tables S1–S14, Figures S1–S8)
TRIPOD_AI_checklist_v6_2_10.md	TRIPOD+AI checklist (Sections 1–8)
submission_audit/	Submission audit package (10 files)
results/FROZEN_v1/	Frozen snapshot (122 active files); SHA256SUMS.txt is in submission_audit/
results/mimic_hco3_closure_report.txt	HCO₃⁻ source closure report (Status: CLOSED)
results/pipeline_run_manifest_*.txt	Per-run manifest (hashes + exit codes)
Reproducibility notes
Fixed seeds: All analyses use fixed seeds (R global = 123; bootstrap
CI = 20261101; 100-split = 1:100).

AUC direction: pROC::auc direction fixed to <
(positive-probability-positive) at all three AUC sites. The
direction_audit file records 5,000 calculations with 0 flips.

SHA256 guards: run_model.py, recal_bootstrap.py, and
make_external_stats.py abort with FATAL if the MIMIC wide table or
the prediction files do not match the expected hashes.

Provenance: All manifests (mimic_prediction_manifest.csv,
mimic_recal_input_manifest.csv, apache_input_provenance.csv,
eicu_supp_input_provenance.csv, mimic_sofa_run_manifest.txt) record
input file SHA256 and version.

Bootstrap replicates: The full replicate arrays for BE and HCO₃⁻
are saved to results/external_bootstrap_replicates_*.npz for audit.

Per-patient predictions: Dumped to
results/ml_predictions_internal_v6.2.10.csv and
results/ml_predictions_external_v6.2.10.csv.

Code repository
GitHub: https://github.com/wj-202606/respiratory-failure-risk-calculator

Concept DOI: 10.5281/zenodo.23300381

Latest version DOI (v1.3.1, 2026-10-28): 10.5281/zenodo.23734774

to be updated to v6.2.10 at submission

Web calculator
URL: https://respiratory-failure-risk-calculator-8bdtjwjbpz4arjfskuptqp.streamlit.app

Research-use only. Not a medical device.

License
To be determined.

Contact
Corresponding author: [to be filled]

Citation
If you use this code, please cite the manuscript and the Zenodo DOI
(to be assigned at submission).

text

## 保存 + 归档

```powershell
Set-Location "C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"

# 两个文件保存后（VSCode，UTF-8 无 BOM）

# 验证
Get-Item .\requirements_v6.2.txt, .\README.md | Select-Object Name, Length, LastWriteTime
Get-Content .\requirements_v6.2.txt
Get-Content .\README.md | Select-Object -First 20

# 归档
Copy-Item .\requirements_v6.2.txt .\results\FROZEN_v1\ -Force
Copy-Item .\README.md .\results\FROZEN_v1\ -Force
Copy-Item .\check_v6_2_1.py .\results\FROZEN_v1\ -Force

## Historical files

The `results/FROZEN_v1/archive_old/` subdirectory contains 10
historical files from earlier analysis rounds (identified by the
absence of the "v5" suffix or by the ".bak_" suffix). These files
are preserved for traceability but are NOT part of the current
v6.2.10 pipeline and are excluded from `submission_audit/SHA256SUMS.txt`.

The active `results/FROZEN_v1/` tree (122 files) is verified by
`submission_audit/SHA256SUMS.txt` (122 OK, 0 FAIL on 2026-11-06).
`SHA256SUMS.txt` itself is stored only in `submission_audit/` and
is not duplicated into `FROZEN_v1/`, to avoid a checksum file
containing its own hash.

This sensitivity analysis is independent of the frozen v6.2.10 primary analysis and did not overwrite any primary outputs. The outcome was included only as an auxiliary predictor in the training-set imputation models; validation predictors were imputed using the primary outcome-excluded procedure.