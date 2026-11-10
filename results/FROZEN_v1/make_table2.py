# -*- coding: utf-8 -*-
"""
make_table2.py v1.0 (v6.2.10)
Table 2: Variable selection and final model coefficients.

Panel A: LASSO variable selection frequencies across 50 imputed
         datasets (lambda.1se).
Panel B: Final BE model (ALB + BUN + BE) - Rubin pooled coefficients.
Panel C: HCO3 sensitivity model (ALB + BUN + HCO3) - Rubin pooled
         coefficients.

Reads:
  results/LASSO_vars_frequency.csv
  results/final_vars.csv
  results/Rubin_OR_main3.csv
  results/Rubin_OR_hco3.csv

Writes:
  results/Table2.csv
  tables/Table2.md
"""
import os
import numpy as np
import pandas as pd

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "tables")
os.makedirs(OUTDIR, exist_ok=True)

LASSO_CSV = os.path.join(RESULTS_DIR, "LASSO_vars_frequency.csv")
VARS_CSV  = os.path.join(RESULTS_DIR, "final_vars.csv")
BE_CSV    = os.path.join(RESULTS_DIR, "Rubin_OR_main3.csv")
HCO3_CSV  = os.path.join(RESULTS_DIR, "Rubin_OR_hco3.csv")


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path)


# ---------- Panel A: LASSO ----------
lasso = require_file(LASSO_CSV, "LASSO_vars_frequency.csv")
lasso = lasso.rename(columns={
    "Variable":       "Variable",
    "Selected_Count": "Selected count (of 50)",
    "Selected_Pct":   "Frequency (%)",
})
# 排序：按 Selected count 降序，再按变量名
lasso = lasso.sort_values(
    by=["Selected count (of 50)", "Variable"],
    ascending=[False, True],
).reset_index(drop=True)

# 把 100% 的 5 个变量用加粗标记（可选）
def bold_if_100(row):
    return f"**{row['Variable']}**" if row["Selected count (of 50)"] == 50 else row["Variable"]
lasso_display = lasso.copy()
lasso_display["Variable"] = lasso.apply(bold_if_100, axis=1)

# ---------- Panel B: BE model ----------
be = require_file(BE_CSV, "Rubin_OR_main3.csv")

def fmt_p(p):
    if isinstance(p, str):
        return p
    if pd.isna(p):
        return "—"
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def fmt_or_ci(row):
    return f"{row['OR']:.3f} ({row['Lower_CI']:.3f}–{row['Upper_CI']:.3f})"

be_display = be.copy()
be_display["Term"] = be_display["Variable"].replace({
    "(Intercept)": "Intercept",
})
be_display["Beta"] = be_display["Beta"].apply(lambda x: f"{x:.4f}")
be_display["SE"]   = be_display["SE"].apply(lambda x: f"{x:.4f}")
be_display["OR (95% CI)"] = be_display.apply(fmt_or_ci, axis=1)
be_display["P"]    = be_display["p_value"].apply(fmt_p)

# ---------- Panel C: HCO3 model ----------
hco3 = require_file(HCO3_CSV, "Rubin_OR_hco3.csv")

hco3_display = hco3.copy()
hco3_display["Term"] = hco3_display["Variable"].replace({
    "(Intercept)": "Intercept",
    "HCO3": "HCO₃⁻",
})
hco3_display["Beta"] = hco3_display["Beta"].apply(lambda x: f"{x:.4f}")
hco3_display["SE"]   = hco3_display["SE"].apply(lambda x: f"{x:.4f}")
hco3_display["OR (95% CI)"] = hco3_display.apply(fmt_or_ci, axis=1)
hco3_display["P"]    = hco3_display["p_value"].apply(fmt_p)

# ---------- Model formulas ----------
def build_lp_formula(df, acid_label):
    b0 = df.loc[df["Variable"] == "(Intercept)", "Beta"].iloc[0]
    alb = df.loc[df["Variable"] == "ALB", "Beta"].iloc[0]
    bun = df.loc[df["Variable"] == "BUN", "Beta"].iloc[0]
    acid_col = "BE" if "BE" in df["Variable"].values else "HCO3"
    acid = df.loc[df["Variable"] == acid_col, "Beta"].iloc[0]
    def signed(v):
        return f"+ {v:.4f}" if v >= 0 else f"− {abs(v):.4f}"
    return (f"LP = {b0:.4f} {signed(alb)}·ALB {signed(bun)}·BUN "
            f"{signed(acid)}·{acid_label}")

lp_be   = build_lp_formula(be,   "BE")
lp_hco3 = build_lp_formula(hco3, "HCO₃⁻")

# ---------- Build full CSV (long format for archival) ----------
rows_csv = []
for _, r in lasso.iterrows():
    rows_csv.append({
        "Panel": "A",
        "Term": r["Variable"],
        "Value": f"{int(r['Selected count (of 50)'])}/50",
        "Pct": f"{r['Frequency (%)']:.0f}%",
    })
for _, r in be.iterrows():
    rows_csv.append({
        "Panel": "B",
        "Term": r["Variable"],
        "Beta": f"{r['Beta']:.4f}",
        "SE": f"{r['SE']:.4f}",
        "OR (95% CI)": fmt_or_ci(r),
        "P": fmt_p(r["p_value"]),
    })
for _, r in hco3.iterrows():
    rows_csv.append({
        "Panel": "C",
        "Term": r["Variable"],
        "Beta": f"{r['Beta']:.4f}",
        "SE": f"{r['SE']:.4f}",
        "OR (95% CI)": fmt_or_ci(r),
        "P": fmt_p(r["p_value"]),
    })
csv_out = pd.DataFrame(rows_csv)
csv_out.to_csv(os.path.join(RESULTS_DIR, "Table2.csv"), index=False)

# ---------- Build Markdown ----------
md = []
md.append("**Table 2. LASSO variable selection and final model coefficients**")
md.append("")

# Panel A
md.append("**Panel A. LASSO variable selection frequencies (50 imputed "
          "datasets, λ.1se)**")
md.append("")
md.append("| Variable | Selected count (of 50) | Frequency (%) |")
md.append("|---|---|---|")
for _, r in lasso_display.iterrows():
    md.append(f"| {r['Variable']} | {int(r['Selected count (of 50)'])} "
              f"| {r['Frequency (%)']:.0f}% |")
md.append("")
md.append("Variables selected in 100% of imputed datasets are shown in "
          "bold. LASSO (glmnet, α = 1) was applied to z-scored continuous "
          "predictors across 50 MICE-imputed training datasets; the "
          "λ.1se penalty was used. The final three-variable model "
          "(ALB, BUN, BE) was prespecified on clinical availability and "
          "parsimony; variables selected by LASSO but not included in "
          "the final model (DD, APTT, WBC) were not carried forward. "
          "DD and APTT are not consistently measured at the 24-h landmark "
          "in MIMIC-IV or eICU-CRD, so they could not be externally "
          "validated.")
md.append("")

# Panel B
md.append("**Panel B. Final BE model (ALB + BUN + BE) — Rubin pooled "
          "coefficients**")
md.append("")
md.append("| Term | β | SE | OR (95% CI) | P |")
md.append("|---|---|---|---|---|")
for _, r in be_display.iterrows():
    md.append(f"| {r['Term']} | {r['Beta']} | {r['SE']} "
              f"| {r['OR (95% CI)']} | {r['P']} |")
md.append("")
md.append(f"Linear predictor: `{lp_be}`.")
md.append("")

# Panel C
md.append("**Panel C. HCO₃⁻ sensitivity model (ALB + BUN + HCO₃⁻) — "
          "Rubin pooled coefficients**")
md.append("")
md.append("| Term | β | SE | OR (95% CI) | P |")
md.append("|---|---|---|---|---|")
for _, r in hco3_display.iterrows():
    md.append(f"| {r['Term']} | {r['Beta']} | {r['SE']} "
              f"| {r['OR (95% CI)']} | {r['P']} |")
md.append("")
md.append(f"Linear predictor: `{lp_hco3}`.")
md.append("")

md.append(
    "ALB, albumin; BUN, blood urea nitrogen; BE, base excess; "
    "HCO₃⁻, bicarbonate; β, regression coefficient; SE, standard error; "
    "OR, odds ratio; CI, confidence interval. "
    "Coefficients were estimated by multivariable logistic regression "
    "in each of 50 MICE-imputed training datasets and combined using "
    "Rubin's rules. ORs are exp(β). P values are two-sided. "
    "The BE model is the primary model; the HCO₃⁻ model is a "
    "prespecified sensitivity analysis substituting bicarbonate for "
    "base excess.")

with open(os.path.join(OUTDIR, "Table2.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")

print(f"[Table 2] {os.path.join(RESULTS_DIR, 'Table2.csv')}")
print(f"[Table 2] {os.path.join(OUTDIR, 'Table2.md')}")
print()
print("Panel A (LASSO):")
print(lasso.to_string(index=False))
print()
print("Panel B (BE model):")
print(be.to_string(index=False))
print()
print("Panel C (HCO3 model):")
print(hco3.to_string(index=False))