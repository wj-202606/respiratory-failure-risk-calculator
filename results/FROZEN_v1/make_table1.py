# -*- coding: utf-8 -*-
"""
make_table1.py v1.1 (v6.2.10)
Table 1: Baseline characteristics of the development, training,
internal validation, and external validation cohorts.

v1.1: Removed the missingness rows (External is BE complete-case by
      construction; the missingness of the full MIMIC-IV cohort is
      already reported in Table S8/S9). Removed the old footnote
      about Training n=1421 / Internal val n=607 (no longer applies).

Reads:
  results/traindata_descriptive.csv  (n=1420, 154 events)
  results/valdata_descriptive.csv    (n=608, 72 events)
  results/external_predictions.csv   (hadm_id list, n=434)
  D:\\mimic_wide_with_hco3.csv       (external labs)
  D:\\mimic_wide_with_demo.csv       (external age, sex)

Writes:
  results/Table1_v6_2_10.csv
  tables/Table1.md
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "tables")
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_CSV = os.path.join(RESULTS_DIR, "traindata_descriptive.csv")
VAL_CSV   = os.path.join(RESULTS_DIR, "valdata_descriptive.csv")
EXT_PRED  = os.path.join(RESULTS_DIR, "external_predictions.csv")
MIMIC_WIDE = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide_with_hco3.csv")
MIMIC_DEMO = r"D:\mimic_wide_with_demo.csv"


def fmt_mean_sd(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"


def welch_p(x, y):
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return "—"
    p = stats.ttest_ind(x, y, equal_var=False).pvalue
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def chi2_p(a, b, c, d):
    table = np.array([[a, b], [c, d]])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return "—"
    p = stats.chi2_contingency(table, correction=True).pvalue
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------- Development cohort ----------
train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
common_cols = [c for c in train.columns if c in val.columns]
dev = pd.concat([train[common_cols], val[common_cols]], ignore_index=True)
print(f"[dev] train n={len(train)}, events={int(train['Outcome'].sum())}")
print(f"[dev] val   n={len(val)}, events={int(val['Outcome'].sum())}")
print(f"[dev] total n={len(dev)}, events={int(dev['Outcome'].sum())}")

# ---------- External cohort ----------
ext_pred = pd.read_csv(EXT_PRED)
keep_hadm = set(ext_pred["hadm_id"].astype(int).tolist())
print(f"[ext] n={len(ext_pred)}, events={int(ext_pred['outcome'].sum())}")

wide = pd.read_csv(MIMIC_WIDE)
wide = wide[wide["hadm_id"].astype(int).isin(keep_hadm)].copy()
demo = pd.read_csv(MIMIC_DEMO)[["hadm_id", "age", "male"]]
ext = wide.merge(demo, on="hadm_id", how="left", validate="one_to_one")
print(f"[ext] matched {len(ext)} of {len(keep_hadm)} hadm_id")


def agg_col(name, col_dev, col_train, col_val, col_ext):
    return {
        "Variable": name,
        "Development (n=2028)":  fmt_mean_sd(dev[col_dev]),
        "Training (n=1420)":     fmt_mean_sd(train[col_train]),
        "Internal val (n=608)":  fmt_mean_sd(val[col_val]),
        "External (n=434)":      fmt_mean_sd(ext[col_ext]),
        "P value": welch_p(dev[col_dev], ext[col_ext]),
    }


def agg_male():
    def count_male(s):
        s = s.astype(str).str.lower()
        return int(s.isin(["male", "m", "1", "男"]).sum())
    dev_m   = count_male(dev["sex"]) if "sex" in dev.columns else 0
    train_m = count_male(train["sex"]) if "sex" in train.columns else 0
    val_m   = count_male(val["sex"]) if "sex" in val.columns else 0
    ext_m   = int(pd.to_numeric(ext["male"], errors="coerce").fillna(0).astype(int).sum())
    return {
        "Variable": "Male, n (%)",
        "Development (n=2028)":  f"{dev_m} ({100*dev_m/len(dev):.1f})",
        "Training (n=1420)":     f"{train_m} ({100*train_m/len(train):.1f})",
        "Internal val (n=608)":  f"{val_m} ({100*val_m/len(val):.1f})",
        "External (n=434)":      f"{ext_m} ({100*ext_m/len(ext):.1f})",
        "P value": chi2_p(dev_m, len(dev)-dev_m, ext_m, len(ext)-ext_m),
    }


def agg_mortality():
    dev_e   = int(pd.to_numeric(dev["Outcome"], errors="coerce").fillna(0).astype(int).sum())
    train_e = int(pd.to_numeric(train["Outcome"], errors="coerce").fillna(0).astype(int).sum())
    val_e   = int(pd.to_numeric(val["Outcome"], errors="coerce").fillna(0).astype(int).sum())
    ext_e   = int(pd.to_numeric(ext_pred["outcome"], errors="coerce").fillna(0).astype(int).sum())
    return {
        "Variable": "In-hospital mortality, n (%)",
        "Development (n=2028)":  f"{dev_e} ({100*dev_e/len(dev):.1f})",
        "Training (n=1420)":     f"{train_e} ({100*train_e/len(train):.1f})",
        "Internal val (n=608)":  f"{val_e} ({100*val_e/len(val):.1f})",
        "External (n=434)":      f"{ext_e} ({100*ext_e/len(ext):.1f})",
        "P value": chi2_p(dev_e, len(dev)-dev_e, ext_e, len(ext)-ext_e),
    }


rows = []
if "age" in dev.columns:
    rows.append(agg_col("Age, years, mean (SD)", "age", "age", "age", "age"))
rows.append(agg_male())
if "ALB" in dev.columns:
    rows.append(agg_col("ALB, g/L, mean (SD)", "ALB", "ALB", "ALB", "alb_g_l"))
if "BUN" in dev.columns:
    rows.append(agg_col("BUN, mmol/L, mean (SD)", "BUN", "BUN", "BUN", "bun_mmol_l"))
if "BE" in dev.columns:
    rows.append(agg_col("BE, mmol/L, mean (SD)", "BE", "BE", "BE", "be_mmol_l"))
if "HCO3" in dev.columns:
    rows.append(agg_col("HCO3, mmol/L, mean (SD)", "HCO3", "HCO3", "HCO3", "hco3_mmol_l"))
rows.append(agg_mortality())

out = pd.DataFrame(rows)
cols = ["Variable", "Development (n=2028)", "Training (n=1420)",
        "Internal val (n=608)", "External (n=434)", "P value"]
out = out[cols]
out.to_csv(os.path.join(RESULTS_DIR, "Table1_v6_2_10.csv"), index=False)

md = []
md.append("**Table 1. Baseline characteristics of the development, "
          "internal validation, and external validation cohorts**")
md.append("")
md.append("| " + " | ".join(cols) + " |")
md.append("|" + "|".join(["---"] * len(cols)) + "|")
for _, r in out.iterrows():
    md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
md.append("")
md.append(
    "SD, standard deviation; ALB, albumin; BUN, blood urea nitrogen; "
    "BE, base excess; HCO₃⁻, bicarbonate. Development cohort = Training "
    "+ Internal validation. Training (n=1,420) and Internal validation "
    "(n=608) match Table 3 exactly. External cohort = MIMIC-IV BE "
    "complete-case cohort (n=434); by construction no predictor is "
    "missing in this subset, so missingness rows are not shown. "
    "External age and sex were obtained from the MIMIC-IV demographics "
    "file matched by hadm_id. P values compare the development cohort "
    "with the external cohort: Welch's t-test for continuous variables; "
    "chi-square test for categorical variables.")

with open(os.path.join(OUTDIR, "Table1.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"[Table 1] {os.path.join(RESULTS_DIR, 'Table1_v6_2_10.csv')}")
print(f"[Table 1] {os.path.join(OUTDIR, 'Table1.md')}")
print()
print(out.to_string(index=False))