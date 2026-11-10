**Table 4. Comparison of logistic regression with five machine learning algorithms**

| Model | Internal AUC | Internal Brier | External AUC | External Brier |
|---|---|---|---|---|
| Logistic regression (Python) | 0.744 | 0.098 | 0.620 | 0.166 |
| Random forest | 0.748 | 0.096 | 0.609 | 0.164 |
| XGBoost | 0.742 | 0.096 | 0.603 | 0.164 |
| LightGBM | 0.720 | 0.098 | 0.587 | 0.166 |
| Support vector machine | 0.587 | 0.103 | 0.502 | 0.170 |
| Multilayer perceptron | 0.445 | 0.162 | 0.439 | 0.198 |

AUC, area under the receiver operating characteristic curve; Brier, Brier score. Internal validation: 608 patients (72 events) from the prespecified 7:3 split of the development cohort. External validation: MIMIC-IV complete-case cohort (n = 434, 88 events). Logistic regression in this table was implemented in Python (scikit-learn, default L2 regularization, C = 1.0) for consistency with the other algorithms; it is therefore an independent implementation and not an exact reproduction of the primary R model (FrozenR_Logistic, external AUC 0.620). XGBoost and LightGBM used inner 5-fold cross-validation for early stopping; the final number of trees was the median of the five folds (XGBoost 93, LightGBM 90). No machine-learning algorithm achieved a higher external AUC than Python logistic regression (0.620).
