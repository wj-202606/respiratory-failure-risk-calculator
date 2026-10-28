@echo off
chcp 65001 >nul
cd /d "C:\Users\24339\Desktop\respiratory-failure-risk-calculator"

git config --global http.proxy http://127.0.0.1:8001
git config --global https.proxy http://127.0.0.1:8001

echo [branch]
git branch --show-current

git add reproduce/
git add analysis/run_model.py
git add analysis/recal_bootstrap.py
git add analysis/verify_frozen.py
git add app.py
git add "论文图/Figure2_external_ROC_calibration.pdf"
git add "论文图/Figure3_DCA.pdf"
git add "论文图/FigureS3_SHAP_summary.pdf"
git add "论文图/FigureS3b_SHAP_bar.pdf"
git add "论文图/FigureS4_recalibration.pdf"
git add "论文图/FigureS6_eicu_ROC_calibration.pdf"
git add "论文图/FigureS7_eicu_hco3_ROC_calibration.pdf"
git add "论文图/FigureS8_eicu_DCA.pdf"
git add results/FROZEN_v1/dca_results_dual.csv
git add results/FROZEN_v1/external_predictions.csv
git add results/FROZEN_v1/recal_bootstrap_summary_v2.csv
git add results/FROZEN_v1/recal_summary_dual.csv
git add results/FROZEN_v1/internal_validation_50imp.csv
git add verify_report.csv
git add verify_report.md

echo [staged]
git status --short

git commit -m "Reproduce all figures/tables end-to-end; align Figure 2 AUC 0.622 and Figure S4 intercept -0.001 with manuscript; rebuild internal-validation FROZEN (verify 97/97)"

git push origin main

echo [after]
git status --short
pause