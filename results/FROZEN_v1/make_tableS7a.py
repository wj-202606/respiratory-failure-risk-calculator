# -*- coding: utf-8 -*-
"""
make_tableS7a.py v1.0 (v6.2.10)
Table S7a: SMDs development vs MIMIC-IV external validation cohort.

Data sources:
  - Development baseline: results/dev_full_baseline.csv
  - MIMIC-IV raw values:  D:\\mimic_wide_with_hco3.csv
    restricted to the 434 hadm_id in results/external_predictions.csv
  - Demographics (age, sex): joined from D:\\mimic_wide_with_demo.csv
    (same hadm_id set; the demo file is used ONLY for age and sex)

The MIMIC-IV external cohort here is the SAME 434-case BE complete-case
cohort used in Table 3 / Figure 2. Mortality is computed on the same
434 cases. No missing values are imputed.

Writes:
  results/TableS7a_dev_vs_mimic.csv
  tables/TableS7a.md
"""
import os
import hashlib
import numpy as np
import pandas as pd

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "tables")
os.makedirs(OUTDIR, exist_ok=True)

MIMIC_WIDE = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide_with_hco3.csv")
MIMIC_DEMO = r"D:\mimic_wide_with_demo.csv"
DEV_BASE   = os.path.join(RESULTS_DIR, "dev_full_baseline.csv")
EXT_PRED   = os.path.join(RESULTS_DIR, "external_predictions.csv")

if not os.path.exists(DEV_BASE):
    raise FileNotFoundError(f"Missing: {DEV_BASE}")
if not os.path.exists(EXT_PRED):
    raise FileNotFoundError(f"Missing: {EXT_PRED}")
if not os.path.exists(MIMIC_WIDE):
    raise FileNotFoundError(f"Missing: {MIMIC_WIDE}")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def smd_continuous(m1, sd1, n1, m2, sd2, n2):
    pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return abs(m1 - m2) / pooled


def smd_binary(p1, n1, p2, n2):
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if denom == 0:
        return np.nan
    return abs(p1 - p2) / denom


# ---- dev ----
dev = pd.read_csv(DEV_BASE)
dev_d = dict(zip(dev["Variable"].astype(str), dev["Value"]))
dev_n = int(dev_d["n"])
dev_age_m, dev_age_sd = float(dev_d["age_mean"]), float(dev_d["age_sd"])
dev_male_pct = float(dev_d["male_pct"])
dev_alb_m, dev_alb_sd = float(dev_d["alb_mean"]), float(dev_d["alb_sd"])
dev_bun_m, dev_bun_sd = float(dev_d["bun_mean"]), float(dev_d["bun_sd"])
dev_be_m,  dev_be_sd  = float(dev_d["be_mean"]),  float(dev_d["be_sd"])
dev_mort_pct = float(dev_d["mort_pct"])

# ---- MIMIC-IV: 434 BE complete-case cohort ----
ext = pd.read_csv(EXT_PRED)
keep_hadm = set(ext["hadm_id"].astype(int).tolist())
y_ext = ext.set_index("hadm_id")["outcome"].astype(int)

wide = pd.read_csv(MIMIC_WIDE)
wide = wide[wide["hadm_id"].astype(int).isin(keep_hadm)].copy()

if not os.path.exists(MIMIC_DEMO):
    raise FileNotFoundError(f"Missing demographics file: {MIMIC_DEMO} "
                            f"(needed for age and sex)")
demo = pd.read_csv(MIMIC_DEMO)[["hadm_id", "age", "male"]]
merged = wide.merge(demo, on="hadm_id", how="left", validate="one_to_one")
print(f"[S7a] matched {len(merged)} of {len(keep_hadm)} hadm_id")

# use only available values; do not impute
mimic_n = int(len(merged))
mimic_age  = pd.to_numeric(merged["age"],  errors="coerce")
mimic_male = pd.to_numeric(merged["male"], errors="coerce")
mimic_alb  = pd.to_numeric(merged["alb_g_l"], errors="coerce")
mimic_bun  = pd.to_numeric(merged["bun_mmol_l"], errors="coerce")
mimic_be   = pd.to_numeric(merged["be_mmol_l"], errors="coerce")
mimic_out  = merged["hadm_id"].map(y_ext).astype(int)

mimic_age_m, mimic_age_sd = mimic_age.mean(), mimic_age.std(ddof=1)
mimic_male_pct = 100 * mimic_male.mean()
mimic_alb_m, mimic_alb_sd = mimic_alb.mean(), mimic_alb.std(ddof=1)
mimic_bun_m, mimic_bun_sd = mimic_bun.mean(), mimic_bun.std(ddof=1)
mimic_be_m,  mimic_be_sd  = mimic_be.mean(),  mimic_be.std(ddof=1)
mimic_mort_pct = 100 * mimic_out.mean()

rows = [
    {"Variable": "Age, years, mean (SD)",
     "Development": f"{dev_age_m:.1f} ({dev_age_sd:.1f})",
     "MIMIC-IV":    f"{mimic_age_m:.1f} ({mimic_age_sd:.1f})",
     "SMD": f"{smd_continuous(dev_age_m, dev_age_sd, dev_n, mimic_age_m, mimic_age_sd, mimic_n):.2f}"},

    {"Variable": "Male, %",
     "Development": f"{dev_male_pct:.1f}",
     "MIMIC-IV":    f"{mimic_male_pct:.1f}",
     "SMD": f"{smd_binary(dev_male_pct/100, dev_n, mimic_male_pct/100, mimic_n):.2f}"},

    {"Variable": "ALB, g/L, mean (SD)",
     "Development": f"{dev_alb_m:.1f} ({dev_alb_sd:.1f})",
     "MIMIC-IV":    f"{mimic_alb_m:.1f} ({mimic_alb_sd:.1f})",
     "SMD": f"{smd_continuous(dev_alb_m, dev_alb_sd, dev_n, mimic_alb_m, mimic_alb_sd, mimic_n):.2f}"},

    {"Variable": "BUN, mmol/L, mean (SD)",
     "Development": f"{dev_bun_m:.1f} ({dev_bun_sd:.1f})",
     "MIMIC-IV":    f"{mimic_bun_m:.1f} ({mimic_bun_sd:.1f})",
     "SMD": f"{smd_continuous(dev_bun_m, dev_bun_sd, dev_n, mimic_bun_m, mimic_bun_sd, mimic_n):.2f}"},

    {"Variable": "BE, mmol/L, mean (SD)",
     "Development": f"{dev_be_m:.1f} ({dev_be_sd:.1f})",
     "MIMIC-IV":    f"{mimic_be_m:.1f} ({mimic_be_sd:.1f})",
     "SMD": f"{smd_continuous(dev_be_m, dev_be_sd, dev_n, mimic_be_m, mimic_be_sd, mimic_n):.2f}"},

    {"Variable": "In-hospital mortality, %",
     "Development": f"{dev_mort_pct:.1f}",
     "MIMIC-IV":    f"{mimic_mort_pct:.1f}",
     "SMD": f"{smd_binary(dev_mort_pct/100, dev_n, mimic_mort_pct/100, mimic_n):.2f}"},
]

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS_DIR, "TableS7a_dev_vs_mimic.csv"), index=False)

md = []
md.append("**Table S7a. Baseline comparison — development vs MIMIC-IV "
          "external validation cohort**")
md.append("")
md.append("| Variable | Development (n = {}) | MIMIC-IV (n = {}) | SMD |".format(dev_n, mimic_n))
md.append("|---|---|---|---|")
for _, r in out.iterrows():
    md.append(f"| {r['Variable']} | {r['Development']} | {r['MIMIC-IV']} | {r['SMD']} |")
md.append("")
md.append(
    "SMD, standardized mean difference; an SMD above 0.10 is conventionally "
    "considered non-negligible; BE, base excess; NA, not applicable. "
    f"The MIMIC-IV cohort here is the same {mimic_n}-case BE complete-case "
    "cohort used in Table 3 and Figure 2. Age and sex were obtained from "
    "D:\\mimic_wide_with_demo.csv matched by hadm_id. All clinical values "
    "are used as-is; no imputation is performed. This SMD table is "
    "descriptive and is not a formal hypothesis test.")

with open(os.path.join(OUTDIR, "TableS7a.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"[S7a] written: {os.path.join(OUTDIR, 'TableS7a.md')}")
print(out.to_string(index=False))
print(f"[S7a] dev n = {dev_n};  MIMIC n = {mimic_n}")
print(f"[S7a] source SHA256:")
print(f"  dev_full_baseline.csv : {sha256(DEV_BASE)}")
print(f"  mimic_wide_with_hco3  : {sha256(MIMIC_WIDE)}")
print(f"  external_predictions  : {sha256(EXT_PRED)}")