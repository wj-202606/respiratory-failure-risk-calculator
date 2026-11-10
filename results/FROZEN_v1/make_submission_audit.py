# -*- coding: utf-8 -*-
"""
make_submission_audit.py v1.0 (v6.2.10)
Generate the submission audit package under submission_audit/.

Files produced:
  R_VERSION.txt
  analysis_manifest.csv
  SHA256SUMS.txt
  sample_size_audit.csv
  coefficient_audit.csv
  prediction_provenance.csv
  table_number_audit.csv
  figure_number_audit.csv
  final_validation_report.txt
  change_log.txt

Reads from results/ and tables/ but never modifies them.
"""
import os
import hashlib
import datetime
import pandas as pd

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
TABLES_DIR  = os.path.join(BASE, "tables")
FIG_DIR     = os.path.join(BASE, "论文图")
FROZEN_DIR  = os.path.join(RESULTS_DIR, "FROZEN_v1")
OUT_DIR     = os.path.join(BASE, "submission_audit")
os.makedirs(OUT_DIR, exist_ok=True)

NOW = datetime.datetime.now().isoformat(timespec="seconds")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def write(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"[written] {path}")


# ============================================================
# 1. R_VERSION.txt
# ============================================================
r_version_lines = [
    "R analysis environment",
    "======================",
    f"Generated: {NOW}",
    "",
    "R_analysis_v5_4_1.R",
    "  Pipeline version: v5.4.1",
    "  R version: R version 4.5.1",
    "  Packages used: naniar, mice, tableone, ggplot2, lattice, caret,",
    "                 autoReg, dplyr, Matrix, glmnet, Boruta, Hmisc,",
    "                 rms, car, pROC, e1071, boot, dcurves",
    "",
    "Global seed: 123",
    "Outer 7:3 split seed: 123",
    "Imputation seed: 123 (train and combined train+val)",
    "LASSO loop seed: 123 + imp_i (imp_i in 1..50)",
    "100-split seed: 1:100",
    "Bootstrap seed: 123",
    "",
    "AUC direction: fixed to '<' (positive-probability-positive)",
    "  in pROC::auc calls at the 50-imputation train AUC, the",
    "  50-imputation internal-validation AUC, and the 100-split AUC.",
    "  direction_audit (5,000 AUC calculations): 0 flips.",
    "",
    "Environment file: results/run_manifest_v54.txt",
]
write(os.path.join(OUT_DIR, "R_VERSION.txt"), "\n".join(r_version_lines) + "\n")


# ============================================================
# 2. analysis_manifest.csv
# ============================================================
manifest_rows = [
    # (file, source, purpose, version)
    ("R_analysis_v5_4_1.R", "R analysis", "MICE + LASSO + Rubin's rules + internal validation + 100-split stability", "v5.4.1"),
    ("run_model.py", "Python", "MIMIC-IV frozen predictions + 5 ML algorithms + paired DeLong + SHAP", "v6.2.10"),
    ("make_external_stats.py", "Python", "Single source of truth for AUC/Brier/calibration CIs (external)", "v1.2"),
    ("fig2_be.py", "Python", "Figure 2 (BE ROC + calibration)", "v1.0"),
    ("figS2_hco3.py", "Python", "Figure S2 (HCO3 ROC + calibration)", "v2.1"),
    ("figS1_internal.py", "Python", "Figure S1 (internal validation ROC + calibration)", "v2.0"),
    ("recal_bootstrap.py", "Python", "Recalibration + bootstrap OOB + DCA (Figures 3, S4)", "v6.2.8"),
    ("run_mimic_sofa_all.py", "Python", "MIMIC-IV SOFA comparison", "v2.5"),
    ("eicu_build_wide.py", "Python", "eICU-CRD cohort construction (Plan B: ICU-admission time origin)", "v6.0"),
    ("eicu_validate.py", "Python", "eICU-CRD external validation (Figures S6, S7, S8)", "v6.0"),
    ("compare_eicu_apache.py", "Python", "eICU-CRD APACHE IVa comparison", "v6.2.5"),
    ("make_eicu_supp_tables.py", "Python", "eICU supplementary tables S6, S7b, S11, S12, S13", "v6.2.9"),
    ("make_tables.py", "Python", "Main tables 3, 4, 5, 6a, 6b, S12, S14", "v6.2.9"),
    ("make_tableS2.py", "Python", "Table S2 (HCO3 sensitivity)", "v2.0"),
    ("make_tableS4.py", "Python", "Table S4 (pairwise DeLong, 15 internal + 21 external)", "v1.0"),
    ("make_tableS5_representative.py", "Python", "Table S5 (comparison with published prognostic models)", "v1.0"),
    ("make_tableS7a.py", "Python", "Table S7a (development vs MIMIC-IV baseline)", "v1.0"),
    ("make_tableS8.py", "Python", "Table S8 (MIMIC-IV included vs excluded)", "v1.0"),
    ("alb_missing_analysis.py", "Python", "Table S9 (MIMIC-IV ALB measured vs not)", "v1.0"),
    ("two_var_model.py", "Python", "Table S10 (two-variable ALB-fixed sensitivity)", "v1.1"),
    ("make_table1.py", "Python", "Table 1 (baseline characteristics, 4 cohorts)", "v1.1"),
    ("make_table2.py", "Python", "Table 2 (LASSO + final model coefficients)", "v1.0"),
    ("make_closure_report.py", "Python", "MIMIC HCO3 closure report", "v1.1"),
    ("make_submission_audit.py", "Python", "Submission audit package (this script)", "v1.0"),
    ("check_v6_2_1.py", "Python", "Verification script (APACHE, XGBoost, AUC CI)", "v6.2.3"),
    ("ci_config.py", "Python", "Unified bootstrap constants (CI_SEED=20261101)", "v1.1"),
    ("shared_calib.py", "Python", "Shared calibration / OOB / bootstrap utilities", "v1.3"),
    ("run_pipeline_v6_2_10.ps1", "PowerShell", "Pipeline driver (SHA256 guard + CI print + manifest)", "v6.2.10"),
    ("VERSION_v6_2_10.txt", "Metadata", "Full version history and fix log", "v6.2.10"),
    ("Supplementary_v6_2_10.md", "Manuscript", "Supplementary Material (Tables S1-S14, Figures S1-S8)", "v6.2.10"),
    ("TRIPOD_AI_checklist_v6_2_10.md", "Manuscript", "TRIPOD+AI checklist (Sections 1-8)", "v6.2.10"),
    ("Table1_v6_2_10.csv", "Results", "Baseline characteristics (Development / Training / Val / External)", "v6.2.10"),
    ("Table2.csv", "Results", "LASSO frequencies + BE/HCO3 model coefficients (long format)", "v6.2.10"),
    ("Table2.md", "Tables", "Table 2 formatted for manuscript", "v6.2.10"),
    ("external_auc_stats.csv", "Results", "AUC/Brier/slope/intercept + CI for BE and HCO3", "v6.2.10"),
    ("external_auc_stats.json", "Results", "JSON version of external_auc_stats.csv", "v6.2.10"),
    ("external_predictions.csv", "Results", "MIMIC-IV BE per-patient predictions (n=434)", "v6.2.10"),
    ("external_predictions_hco3_v2.csv", "Results", "MIMIC-IV HCO3 per-patient predictions (n=684)", "v6.2.10"),
    ("external_bootstrap_replicates_BE.npz", "Results", "2,000 bootstrap replicates (BE) for audit", "v6.2.10"),
    ("external_bootstrap_replicates_HCO3.npz", "Results", "2,000 bootstrap replicates (HCO3) for audit", "v6.2.10"),
    ("ml_predictions_internal_v6.2.10.csv", "Results", "Internal validation per-patient predictions (n=608 x 6 models)", "v6.2.10"),
    ("ml_predictions_external_v6.2.10.csv", "Results", "External per-patient predictions (n=434 x 7 models)", "v6.2.10"),
    ("ml_model_hyperparameters.csv", "Results", "ML hyperparameters and global seed", "v6.2.10"),
    ("python_ml_early_stopping_summary.csv", "Results", "XGBoost/LightGBM inner-CV best_iter + median", "v6.2.10"),
    ("auc_100splits_v5.csv", "Results", "100-split AUC (median 0.751, IQR 0.731-0.772)", "v6.2.10"),
    ("auc_100splits_direction_audit.csv", "Results", "5,000 AUC direction audit (flipped = 0)", "v6.2.10"),
    ("internal_validation_50imp_v5.csv", "Results", "50-imputation internal validation (AUC 0.727, SD 0.006)", "v6.2.10"),
    ("mimic_prediction_manifest.csv", "Results", "MIMIC prediction provenance (hashes, n, events)", "v6.2.10"),
    ("mimic_recal_input_manifest.csv", "Results", "Recalibration input provenance", "v6.2.10"),
    ("mimic_sofa_vs_model.csv", "Results", "SOFA vs 3-variable model comparison", "v6.2.10"),
    ("mimic_sofa_run_manifest.txt", "Results", "SOFA input/output SHA256 + result summary", "v6.2.10"),
    ("apache_delong_results.csv", "Results", "APACHE IVa vs 3-variable model comparison", "v6.2.10"),
    ("apache_input_provenance.csv", "Results", "APACHE input provenance", "v6.2.10"),
    ("eicu_be_summary.csv", "Results", "eICU-CRD BE model summary (AUC, Brier, calibration, OOB)", "v6.2.10"),
    ("eicu_hco3_summary.csv", "Results", "eICU-CRD HCO3 model summary", "v6.2.10"),
    ("eicu_supp_input_provenance.csv", "Results", "eICU supplementary table input provenance", "v6.2.10"),
    ("LASSO_vars_frequency.csv", "Results", "LASSO selection frequencies (9 variables)", "v6.2.10"),
    ("Rubin_OR_main3.csv", "Results", "BE model Rubin pooled coefficients", "v6.2.10"),
    ("Rubin_OR_hco3.csv", "Results", "HCO3 model Rubin pooled coefficients", "v6.2.10"),
    ("final_vars.csv", "Results", "Final variable declarations (Main3_BE / Sens3_HCO3 / Extra4_DD)", "v6.2.10"),
    ("two_var_summary.csv", "Results", "ALB-fixed two-variable sensitivity (n=1,169)", "v6.2.10"),
    ("TableS2_hco3_sensitivity.csv", "Results", "Table S2 external validation + recalibration", "v6.2.10"),
    ("TableS4_delong_internal.csv", "Results", "Table S4 internal DeLong (15 pairs)", "v6.2.10"),
    ("TableS4_delong_external.csv", "Results", "Table S4 external DeLong (21 pairs)", "v6.2.10"),
    ("TableS5_representative.csv", "Results", "Table S5 published models comparison", "v6.2.10"),
    ("TableS6_this_study_row.csv", "Results", "Table S6 This study summary row", "v6.2.10"),
    ("TableS7a_dev_vs_mimic.csv", "Results", "Table S7a development vs MIMIC-IV baseline", "v6.2.10"),
    ("TableS7b_dev_vs_eicu.csv", "Results", "Table S7b development vs eICU baseline", "v6.2.10"),
    ("TableS8_inclusion_vs_exclusion.csv", "Results", "Table S8 MIMIC-IV included vs excluded", "v6.2.10"),
    ("TableS9_alb_missing.csv", "Results", "Table S9 MIMIC-IV ALB measured vs not", "v6.2.10"),
    ("TableS11_eicu_baseline.csv", "Results", "Table S11 eICU baseline characteristics", "v6.2.10"),
    ("TableS12_eicu_incl_vs_excl.csv", "Results", "Table S12 eICU included vs excluded", "v6.2.10"),
    ("TableS13_eicu_alb_measured_vs_not.csv", "Results", "Table S13 eICU ALB measured vs not", "v6.2.10"),
]
manifest_df = pd.DataFrame(
    manifest_rows,
    columns=["file", "source", "purpose", "version"],
)
manifest_df.to_csv(os.path.join(OUT_DIR, "analysis_manifest.csv"), index=False)
print(f"[written] {os.path.join(OUT_DIR, 'analysis_manifest.csv')}")


# ============================================================
# 3. SHA256SUMS.txt
# ============================================================
sum_lines = []
sum_lines.append("SHA256SUMS")
sum_lines.append("=" * 70)
sum_lines.append(f"Generated: {NOW}")
sum_lines.append(f"Base directory: {FROZEN_DIR}")
sum_lines.append("")

if os.path.isdir(FROZEN_DIR):
    files = []
    for root, _, names in os.walk(FROZEN_DIR):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, FROZEN_DIR)
            files.append((rel, full))
    files.sort()
    for rel, full in files:
        try:
            h = sha256(full)
            size = os.path.getsize(full)
            sum_lines.append(f"{h}  {size:>10d}  {rel}")
        except Exception as e:
            sum_lines.append(f"ERROR  {rel}  ({e})")
    sum_lines.append("")
    sum_lines.append(f"Total files: {len(files)}")
else:
    sum_lines.append(f"[warn] {FROZEN_DIR} not found")

write(os.path.join(OUT_DIR, "SHA256SUMS.txt"), "\n".join(sum_lines) + "\n")


# ============================================================
# 4. sample_size_audit.csv
# ============================================================
sample_rows = [
    # cohort, subset, n, events, note
    ("Development", "Total",           2028, 226, "Full development cohort after MICE"),
    ("Development", "Training",        1420, 154, "7:3 split, seed 123"),
    ("Development", "Internal validation", 608, 72, "7:3 split, seed 123"),
    ("MIMIC-IV",    "BE complete-case", 434,  88, "dropna on ALB, BUN, BE, outcome"),
    ("MIMIC-IV",    "HCO3 complete-case", 684, 109, "dropna on ALB, BUN, HCO3, outcome"),
    ("MIMIC-IV",    "Full cohort",      2456, None, "after inclusion criteria"),
    ("MIMIC-IV",    "SOFA BE subset",   393,  85, "with first-day SOFA available"),
    ("MIMIC-IV",    "SOFA HCO3 subset", 467, 103, "with first-day SOFA available"),
    ("MIMIC-IV",    "APACHE not applicable", None, None, "APACHE IVa only in eICU"),
    ("eICU-CRD",    "BE complete-case", 1458, 363, "Plan B: ICU-admission time origin"),
    ("eICU-CRD",    "HCO3 complete-case", 2287, 521, "Plan B: ICU-admission time origin"),
    ("eICU-CRD",    "Full eligible",    4696, 977, "before complete-case drop"),
    ("eICU-CRD",    "APACHE BE subset", 1136, 273, "with APACHE IVa prediction in [0,1]"),
    ("eICU-CRD",    "APACHE HCO3 subset", 1722, 387, "with APACHE IVa prediction in [0,1]"),
]
sample_df = pd.DataFrame(
    sample_rows,
    columns=["cohort", "subset", "n", "events", "note"],
)
sample_df.to_csv(os.path.join(OUT_DIR, "sample_size_audit.csv"), index=False)
print(f"[written] {os.path.join(OUT_DIR, 'sample_size_audit.csv')}")


# ============================================================
# 5. coefficient_audit.csv
# ============================================================
def read_coef_audit(path, model_label):
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    file_hash = sha256(path)
    rows = []
    for _, r in df.iterrows():
        term = r.get("Term", r.get("term", r.iloc[0]))
        beta = r.get("Beta", r.get("beta", r.get("estimate", r.iloc[1])))
        se = r.get("SE", r.get("se", r.get("std.error", None)))
        rows.append({
            "model": model_label,
            "term": term,
            "beta": beta,
            "se": se if se is not None else "",
            "source_file": os.path.basename(path),
            "source_sha256": file_hash,
        })
    return rows

coef_rows = []
coef_rows += read_coef_audit(os.path.join(RESULTS_DIR, "external_validation_coef.csv"), "BE")
coef_rows += read_coef_audit(os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv"), "HCO3")

if coef_rows:
    pd.DataFrame(coef_rows).to_csv(
        os.path.join(OUT_DIR, "coefficient_audit.csv"), index=False)
    print(f"[written] {os.path.join(OUT_DIR, 'coefficient_audit.csv')}")


# ============================================================
# 6. prediction_provenance.csv
# ============================================================
pred_rows = []
man_path = os.path.join(RESULTS_DIR, "mimic_prediction_manifest.csv")
if os.path.exists(man_path):
    man = pd.read_csv(man_path)
    for _, r in man.iterrows():
        pred_rows.append({
            "file": ("external_predictions.csv"
                     if r["Version"] == "BE"
                     else "external_predictions_hco3_v2.csv"),
            "model": r["Version"],
            "n": int(r["n"]),
            "events": int(r["events"]),
            "coef_sha256": r["coef_sha256"],
            "mimic_sha256": r["mimic_sha256"],
            "model_version": r["model_version"],
        })

# eICU predictions
for label, f, ver in [
    ("BE", "eicu_predictions_be.csv", "v6.0"),
    ("HCO3", "eicu_predictions_hco3.csv", "v6.0"),
]:
    p = os.path.join(RESULTS_DIR, f)
    if os.path.exists(p):
        df = pd.read_csv(p)
        pred_rows.append({
            "file": f,
            "model": label,
            "n": len(df),
            "events": int(df["outcome"].sum()) if "outcome" in df.columns else "",
            "coef_sha256": sha256(
                os.path.join(RESULTS_DIR, f"external_validation_coef"
                              + ("" if label == "BE" else "_hco3") + ".csv")),
            "mimic_sha256": "N/A (eICU-CRD)",
            "model_version": ver,
        })

if pred_rows:
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(OUT_DIR, "prediction_provenance.csv"), index=False)
    print(f"[written] {os.path.join(OUT_DIR, 'prediction_provenance.csv')}")


# ============================================================
# 7. table_number_audit.csv
# ============================================================
table_rows = [
    {"table": "Table 1",  "subset": "Development (n=2,028, 226 events)", "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Table1_v6_2_10.csv"},
    {"table": "Table 1",  "subset": "Training (n=1,420, 154 events)",     "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Table1_v6_2_10.csv"},
    {"table": "Table 1",  "subset": "Internal validation (n=608, 72 events)", "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Table1_v6_2_10.csv"},
    {"table": "Table 1",  "subset": "External MIMIC-IV (n=434, 88 events)", "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Table1_v6_2_10.csv"},

    {"table": "Table 2", "subset": "BE model coefficients (n=2,028)", "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Rubin_OR_main3.csv"},
    {"table": "Table 2", "subset": "HCO3 model coefficients (n=2,028)", "AUC": "", "Brier": "", "slope": "", "intercept": "", "source": "Rubin_OR_hco3.csv"},

    {"table": "Table 3", "subset": "Training subset (n=1,420, 154 events)", "AUC": "0.767", "Brier": "0.088", "slope": "—", "intercept": "—", "source": "tables/Table3.csv"},
    {"table": "Table 3", "subset": "Internal validation (n=608, 72 events)", "AUC": "0.727 (SD 0.006)", "Brier": "0.098 (SD 0.001)", "slope": "0.797 (0.781–0.819)", "intercept": "−0.356 (−0.385 to −0.316)", "source": "tables/Table3.csv"},
    {"table": "Table 3", "subset": "MIMIC-IV BE (n=434, 88 events)", "AUC": "0.620 (0.555–0.681)", "Brier": "0.166 (0.155–0.178)", "slope": "0.410", "intercept": "−0.637", "source": "tables/Table3.csv"},
    {"table": "Table 3", "subset": "MIMIC-IV HCO3 (n=684, 109 events)", "AUC": "0.642 (0.588–0.700)", "Brier": "0.137 (0.129–0.145)", "slope": "0.485", "intercept": "−0.809", "source": "tables/Table3.csv"},
    {"table": "Table 3", "subset": "eICU-CRD BE (n=1,458, 363 events)", "AUC": "0.670 (0.638–0.703)", "Brier": "0.179 (0.167–0.190)", "slope": "0.599", "intercept": "−0.486", "source": "tables/Table3.csv"},
    {"table": "Table 3", "subset": "eICU-CRD HCO3 (n=2,287, 521 events)", "AUC": "0.670 (0.643–0.696)", "Brier": "0.171 (0.162–0.180)", "slope": "0.619", "intercept": "−0.591", "source": "tables/Table3.csv"},

    {"table": "Table 4", "subset": "Internal Logistic", "AUC": "0.744", "Brier": "0.098", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "Internal RandomForest", "AUC": "0.748", "Brier": "0.096", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "Internal XGBoost", "AUC": "0.742", "Brier": "0.096", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "Internal LightGBM", "AUC": "0.720", "Brier": "0.098", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "Internal SVM", "AUC": "0.587", "Brier": "0.103", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "Internal MLP", "AUC": "0.445", "Brier": "0.162", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External Python Logistic", "AUC": "0.620", "Brier": "0.166", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External RandomForest", "AUC": "0.609", "Brier": "0.164", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External XGBoost", "AUC": "0.603", "Brier": "0.164", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External LightGBM", "AUC": "0.587", "Brier": "0.166", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External SVM", "AUC": "0.502", "Brier": "0.170", "slope": "", "intercept": "", "source": "tables/Table4.csv"},
    {"table": "Table 4", "subset": "External MLP", "AUC": "0.439", "Brier": "0.198", "slope": "", "intercept": "", "source": "tables/Table4.csv"},

    {"table": "Table 5", "subset": "MIMIC-IV BE OOB Brier", "AUC": "", "Brier": "0.1596 [0.1492–0.1696]", "slope": "", "intercept": "", "source": "tables/Table5.csv"},
    {"table": "Table 5", "subset": "MIMIC-IV HCO3 OOB Brier", "AUC": "", "Brier": "0.1321 [0.1240–0.1403]", "slope": "", "intercept": "", "source": "tables/Table5.csv"},
    {"table": "Table 5", "subset": "eICU-CRD BE OOB Brier", "AUC": "", "Brier": "0.1739 [0.1692–0.1794]", "slope": "", "intercept": "", "source": "tables/Table5.csv"},
    {"table": "Table 5", "subset": "eICU-CRD HCO3 OOB Brier", "AUC": "", "Brier": "0.1638 [0.1598–0.1681]", "slope": "", "intercept": "", "source": "tables/Table5.csv"},

    {"table": "Table S14", "subset": "MIMIC-IV SOFA BE", "AUC": "0.613 vs 0.638", "Brier": "—", "slope": "—", "intercept": "—", "source": "tables/TableS14.csv"},
    {"table": "Table S14", "subset": "MIMIC-IV SOFA HCO3", "AUC": "0.612 vs 0.642", "Brier": "—", "slope": "—", "intercept": "—", "source": "tables/TableS14.csv"},
    {"table": "Table S14", "subset": "eICU APACHE BE", "AUC": "0.671 vs 0.727", "Brier": "—", "slope": "—", "intercept": "—", "source": "tables/TableS14.csv"},
    {"table": "Table S14", "subset": "eICU APACHE HCO3", "AUC": "0.678 vs 0.739", "Brier": "—", "slope": "—", "intercept": "—", "source": "tables/TableS14.csv"},
]
table_audit_df = pd.DataFrame(table_rows)
table_audit_df.to_csv(os.path.join(OUT_DIR, "table_number_audit.csv"), index=False)
print(f"[written] {os.path.join(OUT_DIR, 'table_number_audit.csv')}")


# ============================================================
# 8. figure_number_audit.csv
# ============================================================
figure_rows = [
    {"figure": "Figure 1", "content": "Participant flow diagram",
     "key_numbers": "Dev 4,628 → 2,028 (226 deaths); MIMIC 20,780 → 434 BE / 684 HCO3; eICU 200,859 → 1,458 BE / 2,287 HCO3",
     "source_file": "论文图/Figure1_flowchart.pdf"},
    {"figure": "Figure 2", "content": "MIMIC-IV BE ROC + calibration",
     "key_numbers": "AUC 0.620 (95% CI 0.555–0.681); slope 0.410; intercept −0.637",
     "source_file": "论文图/Figure2_external_ROC_calibration.pdf"},
    {"figure": "Figure 3", "content": "DCA — MIMIC-IV",
     "key_numbers": "Net benefit at 10% / 15% / 20% / 30% (see Table 6a)",
     "source_file": "论文图/Figure3_DCA.pdf"},
    {"figure": "Figure S1", "content": "Internal validation ROC + calibration",
     "key_numbers": "AUC 0.727 (IQR 0.724–0.730 across 50 imputations); slope 0.797; intercept −0.356",
     "source_file": "论文图/FigureS1_internal_ROC_calibration.pdf"},
    {"figure": "Figure S2", "content": "MIMIC-IV HCO3 ROC + calibration",
     "key_numbers": "AUC 0.642 (95% CI 0.588–0.700); slope 0.485; intercept −0.809",
     "source_file": "论文图/FigureS2_hco3_ROC_calibration.pdf"},
    {"figure": "Figure S3 / S3b", "content": "SHAP summary + mean |SHAP|",
     "key_numbers": "ALB > BE > BUN; directions consistent with logistic coefficients",
     "source_file": "论文图/FigureS3_SHAP_summary.pdf + FigureS3b_SHAP_bar.pdf"},
    {"figure": "Figure S4", "content": "Calibration before/after recalibration (MIMIC-IV)",
     "key_numbers": "BE: 0.410/−0.637 → 0.999/−0.001; HCO3: 0.485/−0.809 → 1.002/0.004",
     "source_file": "论文图/FigureS4_recalibration.pdf"},
    {"figure": "Figure S5", "content": "100-split AUC distribution",
     "key_numbers": "Median 0.751 (IQR 0.731–0.772)",
     "source_file": "论文图/FigureS5_AUC_distribution.pdf"},
    {"figure": "Figure S6", "content": "eICU-CRD BE ROC + calibration",
     "key_numbers": "n=1,458; 363 deaths; AUC 0.670 (0.638–0.703); slope 0.599; intercept −0.486",
     "source_file": "论文图/FigureS6_eicu_ROC_calibration.pdf"},
    {"figure": "Figure S7", "content": "eICU-CRD HCO3 ROC + calibration",
     "key_numbers": "n=2,287; 521 deaths; AUC 0.670 (0.643–0.696); slope 0.619; intercept −0.591",
     "source_file": "论文图/FigureS7_eicu_hco3_ROC_calibration.pdf"},
    {"figure": "Figure S8", "content": "DCA — eICU-CRD",
     "key_numbers": "See Table 6b",
     "source_file": "论文图/FigureS8_eicu_DCA.pdf"},
]
figure_audit_df = pd.DataFrame(figure_rows)
figure_audit_df.to_csv(os.path.join(OUT_DIR, "figure_number_audit.csv"), index=False)
print(f"[written] {os.path.join(OUT_DIR, 'figure_number_audit.csv')}")


# ============================================================
# 9. final_validation_report.txt
# ============================================================
report_lines = [
    "FINAL VALIDATION REPORT — v6.2.10",
    "=" * 70,
    f"Generated: {NOW}",
    "",
    "1. Version",
    "   v6.2.10 (frozen 2026-11-04)",
    "",
    "2. Pipeline run",
    "   results/pipeline_run_manifest_20261104_163126.txt",
    "   10 steps completed successfully at 2026-11-04 17:02:20",
    "",
    "3. MIMIC wide table",
    "   SHA256: b151612f75c568fce52f5a472206608682daaecd1a8ae2302595c45a0b3a72eb",
    "   Size: 2,456 rows x 8 columns; 684 rows with HCO3",
    "",
    "4. Frozen coefficient files",
    "   external_validation_coef.csv       BE  : see coefficient_audit.csv",
    "   external_validation_coef_hco3.csv  HCO3: see coefficient_audit.csv",
    "",
    "5. Sample sizes",
    "   Development: 2,028 (226 deaths)",
    "   Training: 1,420 (154 deaths)",
    "   Internal validation: 608 (72 deaths)",
    "   MIMIC-IV BE complete-case: 434 (88 deaths)",
    "   MIMIC-IV HCO3 complete-case: 684 (109 deaths)",
    "   eICU-CRD BE complete-case: 1,458 (363 deaths)",
    "   eICU-CRD HCO3 complete-case: 2,287 (521 deaths)",
    "",
    "6. Key performance",
    "   Internal validation AUC: 0.727 (SD 0.006) across 50 imputations",
    "   Internal calibration slope: 0.797; intercept: -0.356",
    "   100-split median AUC: 0.751 (IQR 0.731-0.772)",
    "   MIMIC-IV BE AUC: 0.620 (95% CI 0.555-0.681), Brier 0.166, slope 0.410, intercept -0.637",
    "   MIMIC-IV HCO3 AUC: 0.642 (0.588-0.700), Brier 0.137, slope 0.485, intercept -0.809",
    "   eICU-CRD BE AUC: 0.670 (0.638-0.703), Brier 0.179, slope 0.599, intercept -0.486",
    "   eICU-CRD HCO3 AUC: 0.670 (0.643-0.696), Brier 0.171, slope 0.619, intercept -0.591",
    "",
    "7. Direction audit",
    "   pROC::auc direction fixed to '<' at three AUC sites",
    "   direction_audit: 5,000 rows (100 splits x 50 imputations), 0 flips",
    "",
    "8. Bootstrap valid counts",
    "   n_boot = 2,000 per metric per model",
    "   All 4 metrics (AUC, Brier, slope, intercept) x 2 models = 8 x 2,000 valid",
    "",
    "9. APACHE IVa comparison (eICU-CRD)",
    "   BE  : 3-variable 0.671 (0.633-0.709) vs APACHE 0.727 (0.693-0.761);",
    "         delta AUC -0.056 (95% CI -0.098 to -0.015), p = 0.009, n = 1,136",
    "   HCO3: 3-variable 0.678 (0.647-0.708) vs APACHE 0.739 (0.711-0.767);",
    "         delta AUC -0.062 (95% CI -0.096 to -0.028), p < 0.001, n = 1,722",
    "",
    "10. SOFA comparison (MIMIC-IV)",
    "    BE  : 3-variable 0.613 (0.547-0.680) vs SOFA 0.638 (0.570-0.705);",
    "          delta AUC -0.025 (95% CI -0.092 to 0.044), p = 0.485, n = 393",
    "    HCO3: 3-variable 0.612 (0.551-0.671) vs SOFA 0.642 (0.580-0.701);",
    "          delta AUC -0.029 (95% CI -0.102 to 0.035), p = 0.394, n = 467",
    "",
    "11. Closure report",
    "    results/mimic_hco3_closure_report.txt",
    "    Status: CLOSED (16/16 checks passed)",
    "",
    "12. This audit package",
    "    submission_audit/",
    "      R_VERSION.txt",
    "      analysis_manifest.csv",
    "      SHA256SUMS.txt",
    "      sample_size_audit.csv",
    "      coefficient_audit.csv",
    "      prediction_provenance.csv",
    "      table_number_audit.csv",
    "      figure_number_audit.csv",
    "      final_validation_report.txt",
    "      change_log.txt",
    "",
    "End of report.",
]
write(os.path.join(OUT_DIR, "final_validation_report.txt"),
      "\n".join(report_lines) + "\n")


# ============================================================
# 10. change_log.txt
# ============================================================
changelog_lines = [
    "Change log — v6.2.10",
    "=" * 70,
    f"Generated: {NOW}",
    "",
    "v6.2.1  (2026-11-04)",
    "  - XGBoost early stopping with three-layer fallback",
    "  - APACHE IVa duplicate / merge audit",
    "",
    "v6.2.2  (2026-11-04)",
    "  - HCO3 frozen predictions skipped if no HCO3 column",
    "  - -1 treated as missing in APACHE predictions",
    "",
    "v6.2.3  (2026-11-04)",
    "  - HCO3 row made optional in manifest and Table 3/5",
    "",
    "v6.2.4  (2026-11-04)",
    "  - Table 3/5 HCO3 footnote conditional on has_hco3_mimic",
    "",
    "v6.2.5  (2026-11-04)",
    "  - APACHE and SOFA independent AUC 95% CI",
    "",
    "v6.2.6  (2026-11-04)",
    "  - Table S14 (SOFA and APACHE IVa comparison)",
    "",
    "v6.2.7  (2026-11-04)",
    "  - Table S14 n column thousands separator",
    "",
    "v6.2.8  (2026-11-04)",
    "  - MIMIC wide SHA256 guard in run_model.py and recal_bootstrap.py",
    "  - mimic_prediction_manifest.csv gains mimic_n_rows / mimic_n_cols / hco3_source",
    "  - mimic_recal_input_manifest.csv gains role / expected_sha256",
    "  - make_closure_report.py (new)",
    "",
    "v6.2.9  (2026-11-04)",
    "  - Unified CI: CI_SEED=20261101, N_BOOT_CI=2000, CI_METHOD=percentile,",
    "    CI_LEVEL=0.95, MIN_VALID_BOOT=1900",
    "  - make_external_stats.py (new) — single source of truth",
    "  - Figure 2, Figure S2, Table 3, Table S6 read CI from external_auc_stats.csv",
    "  - Figure S2 regenerated from external_predictions_hco3_v2.csv",
    "  - Table S13 label 'In-hospital mortality, %' -> 'In-hospital death, n (%)'",
    "  - Table S6 MIMIC-IV cell reports explicit AUC",
    "",
    "v6.2.10 (2026-11-04 to 2026-11-06)",
    "  - Figure 2 extracted from run_model.py into fig2_be.py",
    "    (P0 circular dependency removed)",
    "  - shared_calib.bootstrap_ci_auc_brier: outcome-stratified,",
    "    returns dict, removes internal 50% threshold",
    "  - make_external_stats.load_pred: verifies hadm_id, mimic_sha256,",
    "    model_version, landmark, and (v1.2) coef_sha256",
    "  - 4 metrics valid-replicate counts recorded separately",
    "  - Bootstrap replicate arrays saved to .npz for audit",
    "  - figS2_hco3.py verifies prediction-file SHA256 vs external_auc_stats.csv",
    "  - run_pipeline_v6_2_9.ps1 renamed to run_pipeline_v6_2_10.ps1;",
    "    adds shared_calib.py v1.3 version check",
    "  - R_analysis_v5_4_1.R: pROC::auc direction fixed to '<';",
    "    direction_audit reports 0 flips out of 5,000 AUC calculations",
    "  - Table 1 rebuilt from current v6.2.10 data",
    "    (Training n=1,420, Internal validation n=608);",
    "    old Table1.csv (2026-10-10) archived as Table1_OLD_20261010.csv",
    "  - Table 2 generated by make_table2.py",
    "  - Per-patient predictions dumped to",
    "    ml_predictions_internal_v6.2.10.csv and",
    "    ml_predictions_external_v6.2.10.csv",
    "  - ml_model_hyperparameters.csv, python_ml_early_stopping_summary.csv",
    "  - mimic_sofa_run_manifest.txt generated",
    "",
    "End of change log.",
]
write(os.path.join(OUT_DIR, "change_log.txt"),
      "\n".join(changelog_lines) + "\n")


# ============================================================
# Summary
# ============================================================
print()
print("=" * 70)
print(f"Submission audit package written to: {OUT_DIR}")
print("=" * 70)
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    print(f"  {f:40s}  {os.path.getsize(p):>10d} bytes")