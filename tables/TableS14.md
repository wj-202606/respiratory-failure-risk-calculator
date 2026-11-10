**Table S14. Comparison of the three-variable prediction models with established ICU severity scores in external validation cohorts**

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

SOFA, Sequential Organ Failure Assessment; APACHE IVa, Acute Physiology and Chronic Health Evaluation IVa; AUC, area under the receiver operating characteristic curve; CI, confidence interval; —, not applicable. The MIMIC-IV SOFA comparison was restricted to patients in the primary MIMIC-IV BE or HCO₃⁻ complete-case cohort with an identifiable ICU stay and an available first-day SOFA score. The earliest ICU stay was selected for each hospital admission. In the MIMIC-IV `first_day_sofa` derived concept, missing component scores are treated as 0 (normal) when calculating the total SOFA score. In the eICU-CRD APACHE IVa comparison, records with `predictedhospitalmortality = -1` (APACHE prediction not available) were excluded, consistent with the eICU-CRD data dictionary and the MIT-LCP analysis practice; the analysis was restricted to records with valid APACHE IVa predictions in [0, 1]. The primary prediction models used laboratory measurements obtained within the first 24 hours after hospital admission, whereas SOFA and APACHE IVa represented the first 24 hours after ICU admission and the full ICU stay, respectively. Therefore, these comparisons involved subsets of the primary external validation cohorts and different prediction time origins. APACHE IVa achieved significantly higher AUCs than both three-variable models. DeLong's test used the paired covariance of the two AUCs among the same patients; AUCs and ΔAUCs are reported with 95% CIs estimated by paired, outcome-stratified bootstrap resampling (B = 2,000), with both predictions resampled in the same patients to preserve the paired structure. ΔAUC 95% CIs were not derived by subtracting the two independent AUC CIs.
