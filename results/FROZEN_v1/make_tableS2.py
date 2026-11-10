# -*- coding: utf-8 -*-
"""
make_tableS2.py v2.0 (v6.2.10)
Table S2. Sensitivity analysis of the three-variable model using
bicarbonate (HCO3-) instead of base excess (BE).

This script replaces the previous DD 4-variable sensitivity output.
The DD 4-variable analysis is NOT included in S2; it should be
discussed in the manuscript's Discussion section.

Data sources (all from the frozen v6.2.10 pipeline):
  results/external_validation_coef_hco3.csv
  results/external_auc_stats.csv           (HCO3 row)
  results/recal_summary_dual.csv           (HCO3 rows)
  results/recal_bootstrap_summary.csv      (HCO3- row)
  results/eicu_hco3_summary.csv

Writes:
  results/TableS2_hco3_sensitivity.csv
  tables/TableS2.md
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "tables")
os.makedirs(OUTDIR, exist_ok=True)

COEF_HCO3    = os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv")
EXT_STATS    = os.path.join(RESULTS_DIR, "external_auc_stats.csv")
RECAL_DUAL   = os.path.join(RESULTS_DIR, "recal_summary_dual.csv")
RECAL_BOOT   = os.path.join(RESULTS_DIR, "recal_bootstrap_summary.csv")
EICU_HCO3    = os.path.join(RESULTS_DIR, "eicu_hco3_summary.csv")


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def fmt_ci(point, lo, hi, nd=3):
    if any(v is None or (isinstance(v, float) and np.isnan(v))
           for v in (point, lo, hi)):
        return fmt(point, nd)
    return f"{point:.{nd}f} ({lo:.{nd}f}–{hi:.{nd}f})"


def fmt_signed(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if abs(x) < 0.5 * 10 ** (-nd):
        x = 0.0
    return f"{x:.{nd}f}".replace("-", "−")


# ------------------------------------------------------------
# A. HCO3 frozen coefficients and model formula
# ------------------------------------------------------------
coef = require_file(COEF_HCO3, "external_validation_coef_hco3.csv")
term_col = "Term" if "Term" in coef.columns else coef.columns[0]
beta_col = "Beta" if "Beta" in coef.columns else coef.columns[1]
coef_map = dict(zip(coef[term_col].astype(str), coef[beta_col].astype(float)))
if "(Intercept)" in coef_map and "Intercept" not in coef_map:
    coef_map["Intercept"] = coef_map.pop("(Intercept)")

b0   = coef_map["Intercept"]
b_al = coef_map["ALB"]
b_bu = coef_map["BUN"]
b_hc = coef_map["HCO3"]

lp_formula = (
    f"LP = {b0:.6f} "
    f"{'+' if b_al >= 0 else '−'} {abs(b_al):.6f}·ALB "
    f"{'+' if b_bu >= 0 else '−'} {abs(b_bu):.6f}·BUN "
    f"{'+' if b_hc >= 0 else '−'} {abs(b_hc):.6f}·HCO₃⁻"
)

coef_table = pd.DataFrame([
    {"Term": "Intercept", "Beta": round(b0,   6)},
    {"Term": "ALB (g/L)", "Beta": round(b_al, 6)},
    {"Term": "BUN (mmol/L)", "Beta": round(b_bu, 6)},
    {"Term": "HCO₃⁻ (mmol/L)", "Beta": round(b_hc, 6)},
])

# ------------------------------------------------------------
# B. MIMIC-IV external validation (HCO3 row)
# ------------------------------------------------------------
ext_stats = require_file(EXT_STATS, "external_auc_stats.csv")
ext_h = ext_stats[ext_stats["Version"].astype(str).str.upper() == "HCO3"]
if ext_h.empty:
    raise KeyError("external_auc_stats.csv does not contain an HCO3 row")
ext_h = ext_h.iloc[0]

mimic_n      = int(ext_h["n"])
mimic_events = int(ext_h["events"])
mimic_auc    = float(ext_h["AUC"])
mimic_auc_lo = float(ext_h["AUC_lo"])
mimic_auc_hi = float(ext_h["AUC_hi"])
mimic_br     = float(ext_h["Brier"])
mimic_br_lo  = float(ext_h["Brier_lo"])
mimic_br_hi  = float(ext_h["Brier_hi"])
mimic_slope  = float(ext_h["slope"])
mimic_int    = float(ext_h["intercept"])

# ------------------------------------------------------------
# C. MIMIC-IV recalibration + bootstrap OOB (HCO3)
# ------------------------------------------------------------
recal = require_file(RECAL_DUAL, "recal_summary_dual.csv")
recal["Version_norm"] = (recal["Version"].astype(str).str.upper()
                         .str.replace("HCO3-", "HCO3")
                         .str.replace("HCO3⁻", "HCO3"))
recal_h = recal[recal["Version_norm"] == "HCO3"]
rec_orig = recal_h[recal_h["Stage"].astype(str).str.lower() == "original"]
rec_recal = recal_h[recal_h["Stage"].astype(str).str.lower() == "recalibrated"]

if rec_orig.empty or rec_recal.empty:
    raise KeyError("recal_summary_dual.csv missing HCO3 original or recal rows")

mimic_recal_brier = float(rec_recal.iloc[0]["Brier"])
mimic_recal_slope = float(rec_recal.iloc[0]["slope"])
mimic_recal_int   = float(rec_recal.iloc[0]["intercept"])

boot = require_file(RECAL_BOOT, "recal_bootstrap_summary.csv")
boot["Version_norm"] = (boot["Version"].astype(str).str.upper()
                        .str.replace("HCO3-", "HCO3")
                        .str.replace("HCO3⁻", "HCO3"))
boot_h = boot[boot["Version_norm"] == "HCO3"]
mimic_oob_brier = "—"
mimic_oob_slope = "—"
mimic_oob_int   = "—"
if not boot_h.empty:
    b = boot_h.iloc[0]

    def _fmt_iqr(row, med, q25, q75, nd=4):
        try:
            m, lo, hi = float(row[med]), float(row[q25]), float(row[q75])
            return f"{m:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}]"
        except Exception:
            return "—"

    mimic_oob_brier = _fmt_iqr(b, "Brier_OOB_median", "Brier_OOB_q25", "Brier_OOB_q75")
    mimic_oob_slope = _fmt_iqr(b, "slope_OOB_median", "slope_OOB_q25", "slope_OOB_q75")
    mimic_oob_int   = _fmt_iqr(b, "intercept_OOB_median", "intercept_OOB_q25", "intercept_OOB_q75")

# ------------------------------------------------------------
# D. eICU-CRD HCO3 external validation
# ------------------------------------------------------------
eicu = require_file(EICU_HCO3, "eicu_hco3_summary.csv")
if eicu.empty:
    raise KeyError("eicu_hco3_summary.csv is empty")
s = eicu.iloc[0]

eicu_n      = int(s["n"])
eicu_events = int(s["events"])
eicu_auc    = float(s["AUC"])
eicu_auc_lo = float(s["AUC_lo"])
eicu_auc_hi = float(s["AUC_hi"])
eicu_br     = float(s["Brier"])
eicu_br_lo  = float(s["Brier_lo"])
eicu_br_hi  = float(s["Brier_hi"])
eicu_slope  = float(s["cal_slope"])
eicu_int    = float(s["cal_intercept"])
eicu_recal_br    = float(s["Brier_recal"])
eicu_recal_slope = float(s["slope_recal"])
eicu_recal_int   = float(s["intercept_recal"])

def _fmt_eicu_iqr(prefix):
    try:
        m  = float(s[f"{prefix}_median"])
        lo = float(s[f"{prefix}_q25"])
        hi = float(s[f"{prefix}_q75"])
        return f"{m:.4f} [{lo:.4f}–{hi:.4f}]"
    except Exception:
        return "—"

eicu_oob_brier = _fmt_eicu_iqr("OOB_Brier")
eicu_oob_slope = _fmt_eicu_iqr("OOB_slope")
eicu_oob_int   = _fmt_eicu_iqr("OOB_intercept")

# ------------------------------------------------------------
# Assemble outputs
# ------------------------------------------------------------
summary = pd.DataFrame([
    {"Cohort": "MIMIC-IV", "n": mimic_n, "events": mimic_events,
     "AUC": round(mimic_auc, 4),
     "AUC 95% CI": f"{mimic_auc_lo:.3f}–{mimic_auc_hi:.3f}",
     "Brier": round(mimic_br, 4),
     "Brier 95% CI": f"{mimic_br_lo:.3f}–{mimic_br_hi:.3f}",
     "Calibration slope": round(mimic_slope, 4),
     "Calibration intercept": round(mimic_int, 4)},

    {"Cohort": "eICU-CRD", "n": eicu_n, "events": eicu_events,
     "AUC": round(eicu_auc, 4),
     "AUC 95% CI": f"{eicu_auc_lo:.3f}–{eicu_auc_hi:.3f}",
     "Brier": round(eicu_br, 4),
     "Brier 95% CI": f"{eicu_br_lo:.3f}–{eicu_br_hi:.3f}",
     "Calibration slope": round(eicu_slope, 4),
     "Calibration intercept": round(eicu_int, 4)},
])
summary.to_csv(os.path.join(RESULTS_DIR, "TableS2_hco3_sensitivity.csv"),
               index=False)
coef_table.to_csv(os.path.join(RESULTS_DIR, "TableS2_hco3_coefficients.csv"),
                  index=False)

# ------------------------------------------------------------
# Markdown
# ------------------------------------------------------------
md = []
md.append("**Table S2. Sensitivity analysis of the three-variable model "
          "using bicarbonate instead of base excess**")
md.append("")

md.append("**A. Frozen coefficients for the HCO₃⁻ model**")
md.append("")
md.append("| Term | Beta |")
md.append("|---|---|")
for _, r in coef_table.iterrows():
    md.append(f"| {r['Term']} | {r['Beta']:.6f} |")
md.append("")
md.append(f"Linear predictor: `{lp_formula}`")
md.append("")
md.append("Predicted probability: `p = 1 / (1 + exp(-LP))`.")
md.append("")

md.append("**B. External validation and recalibration**")
md.append("")
md.append("| Cohort | n | Events | AUC (95% CI) | Brier (95% CI) | "
          "Calibration slope | Calibration intercept |")
md.append("|---|---|---|---|---|---|---|")
for _, r in summary.iterrows():
    md.append(
        f"| {r['Cohort']} | {r['n']} | {r['events']} | "
        f"{r['AUC']:.3f} ({r['AUC 95% CI']}) | "
        f"{r['Brier']:.3f} ({r['Brier 95% CI']}) | "
        f"{r['Calibration slope']:.3f} | "
        f"{r['Calibration intercept']:.3f} |")
md.append("")

md.append("**C. Recalibration and bootstrap OOB**")
md.append("")
md.append("| Cohort | Stage | Brier | Calibration slope | Calibration intercept |")
md.append("|---|---|---|---|---|")
md.append(f"| MIMIC-IV | Original | {mimic_br:.3f} | "
          f"{mimic_slope:.3f} | {mimic_int:.3f} |")
md.append(f"| MIMIC-IV | Recalibrated (apparent) | {mimic_recal_brier:.3f} | "
          f"{mimic_recal_slope:.3f} | {fmt_signed(mimic_recal_int, 3)} |")
md.append(f"| MIMIC-IV | Recalibrated (bootstrap OOB) | {mimic_oob_brier} | "
          f"{mimic_oob_slope} | {mimic_oob_int} |")
md.append(f"| eICU-CRD | Original | {eicu_br:.3f} | "
          f"{eicu_slope:.3f} | {eicu_int:.3f} |")
md.append(f"| eICU-CRD | Recalibrated (apparent) | {eicu_recal_br:.3f} | "
          f"{eicu_recal_slope:.3f} | {fmt_signed(eicu_recal_int, 3)} |")
md.append(f"| eICU-CRD | Recalibrated (bootstrap OOB) | {eicu_oob_brier} | "
          f"{eicu_oob_slope} | {eicu_oob_int} |")
md.append("")

md.append(
    "AUC, area under the receiver operating characteristic curve; "
    "CI, confidence interval; BE, base excess; HCO₃⁻, bicarbonate; "
    "MIMIC-IV, Medical Information Mart for Intensive Care IV; "
    "eICU-CRD, eICU Collaborative Research Database; OOB, out-of-bag; "
    "—, not applicable. The HCO₃⁻ model substitutes bicarbonate for "
    "base excess in the three-variable laboratory model (ALB, BUN, "
    "and an acid–base marker). Development-cohort coefficients for "
    "the HCO₃⁻ model were frozen and applied to both external "
    "cohorts without refitting. For MIMIC-IV, 95% CIs were estimated "
    "by outcome-stratified percentile bootstrap resampling "
    "(B = 2,000; seed = 20261101); the MIMIC-IV table row is the "
    "same one reported in Table 3. For eICU-CRD, 95% CIs are "
    "reported by the eICU validation script (also B = 2,000). "
    "Bootstrap OOB values are median [IQR] across 1,000 iterations. "
    "Recalibration was fitted separately in each external cohort; "
    "apparent values were computed on the same cohort (no "
    "independent recalibration-validation set). OOB evaluation "
    "provides internal support for the recalibration within each "
    "cohort; it does not constitute independent external validation "
    "of the recalibrated model. The eICU-CRD evaluation used ICU "
    "admission as the time origin (prediction time 24 h after ICU "
    "admission), whereas the development cohort and MIMIC-IV used "
    "hospital admission.")

with open(os.path.join(OUTDIR, "TableS2.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"[Table S2] written: {os.path.join(OUTDIR, 'TableS2.md')}")
print(f"[Table S2] {os.path.join(RESULTS_DIR, 'TableS2_hco3_sensitivity.csv')}")
print(f"[Table S2] {os.path.join(RESULTS_DIR, 'TableS2_hco3_coefficients.csv')}")
print("\nHCO3 coefficients:")
print(coef_table.to_string(index=False))
print("\nExternal validation:")
print(summary.to_string(index=False))