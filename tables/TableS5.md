**Table S5. Comparison of the proposed model with representative published prognostic models**

| Model / Study | Population | Derivation n | External validation | Reported AUC | Calibration reported | Recalibration performed | Source |
|---|---|---|---|---|---|---|---|
| CURB-65 (Lim 2003) | Community-acquired pneumonia | 1,068 | Yes | 0.73–0.79 | No | No | Thorax 2003;58:377–382 |
| PSI (Fine 1997) | Community-acquired pneumonia | 14,199 | Yes | 0.75–0.85 | Partial | No | N Engl J Med 1997;336:243–250 |
| SOFA (Vincent 1996) | ICU patients (mixed) | 1,449 | Yes | 0.70–0.85 | No | No | Intensive Care Med 1996;22:707–710 |
| APACHE II (Knaus 1985) | ICU patients (mixed) | 5,815 | Yes | 0.75–0.85 | No | No | Crit Care Med 1985;13:818–829 |
| This study | Respiratory failure (J96) | 2,028 | Yes (MIMIC-IV n = 434 BE / 684 HCO3; eICU-CRD n = 1,458 BE / 2,287 HCO3) | 0.620 (MIMIC-IV BE); 0.642 (MIMIC-IV HCO3); 0.670 (eICU BE); 0.670 (eICU HCO3) | Yes (slope, intercept, bootstrap OOB) | Yes (slope + intercept) | This study |

AUC, area under the receiver operating characteristic curve; PSI, Pneumonia Severity Index; SOFA, Sequential Organ Failure Assessment; APACHE II, Acute Physiology and Chronic Health Evaluation II; BE, base excess; HCO3, bicarbonate. AUC ranges for the published models are taken verbatim from the original publications and reflect the ranges reported across the derivation and validation cohorts in those papers. They are NOT head-to-head comparisons with the present model: the populations, outcome definitions, and prediction time windows differ between studies. The present study's MIMIC-IV AUC is reported for BE and HCO3 separately because the two models use different acid-base predictors. Calibration and recalibration columns describe whether the original model's performance was reported or updated in the source paper; the present study performs and reports both.