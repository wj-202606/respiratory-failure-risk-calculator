# Table S4. Pairwise comparisons of model discrimination using DeLong's test

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
