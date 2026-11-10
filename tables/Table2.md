**Table 2. LASSO variable selection and final model coefficients**

**Panel A. LASSO variable selection frequencies (50 imputed datasets, λ.1se)**

| Variable | Selected count (of 50) | Frequency (%) |
|---|---|---|
| **ALB** | 50 | 100% |
| **APTT** | 50 | 100% |
| **BE** | 50 | 100% |
| **BUN** | 50 | 100% |
| **DD** | 50 | 100% |
| WBC | 49 | 98% |
| Na | 28 | 56% |
| age | 6 | 12% |
| HCO3 | 1 | 2% |

Variables selected in 100% of imputed datasets are shown in bold. LASSO (glmnet, α = 1) was applied to z-scored continuous predictors across 50 MICE-imputed training datasets; the λ.1se penalty was used. The final three-variable model (ALB, BUN, BE) was prespecified on clinical availability and parsimony; variables selected by LASSO but not included in the final model (DD, APTT, WBC) were not carried forward. DD and APTT are not consistently measured at the 24-h landmark in MIMIC-IV or eICU-CRD, so they could not be externally validated.

**Panel B. Final BE model (ALB + BUN + BE) — Rubin pooled coefficients**

| Term | β | SE | OR (95% CI) | P |
|---|---|---|---|---|
| Intercept | 1.7825 | 0.6111 | 5.945 (1.792–19.717) | 0.004 |
| ALB | -0.1227 | 0.0180 | 0.885 (0.854–0.916) | <0.001 |
| BUN | 0.0343 | 0.0117 | 1.035 (1.011–1.059) | 0.003 |
| BE | -0.0585 | 0.0129 | 0.943 (0.920–0.967) | <0.001 |

Linear predictor: `LP = 1.7825 − 0.1227·ALB + 0.0343·BUN − 0.0585·BE`.

**Panel C. HCO₃⁻ sensitivity model (ALB + BUN + HCO₃⁻) — Rubin pooled coefficients**

| Term | β | SE | OR (95% CI) | P |
|---|---|---|---|---|
| Intercept | 3.1936 | 0.7233 | 24.375 (5.898–100.743) | <0.001 |
| ALB | -0.1209 | 0.0179 | 0.886 (0.855–0.918) | <0.001 |
| BUN | 0.0374 | 0.0114 | 1.038 (1.015–1.062) | 0.001 |
| HCO₃⁻ | -0.0604 | 0.0140 | 0.941 (0.916–0.968) | <0.001 |

Linear predictor: `LP = 3.1936 − 0.1209·ALB + 0.0374·BUN − 0.0604·HCO₃⁻`.

ALB, albumin; BUN, blood urea nitrogen; BE, base excess; HCO₃⁻, bicarbonate; β, regression coefficient; SE, standard error; OR, odds ratio; CI, confidence interval. Coefficients were estimated by multivariable logistic regression in each of 50 MICE-imputed training datasets and combined using Rubin's rules. ORs are exp(β). P values are two-sided. The BE model is the primary model; the HCO₃⁻ model is a prespecified sensitivity analysis substituting bicarbonate for base excess.
