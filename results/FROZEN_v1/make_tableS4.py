# -*- coding: utf-8 -*-
"""
make_tableS4.py v1.0 (v6.2.10)
Pairwise DeLong comparisons of model discrimination.

Reads:
  results/ml_predictions_internal_v6.2.10.csv  (n=608, 6 models)
  results/ml_predictions_external_v6.2.10.csv  (n=434, 7 models)

Writes:
  results/TableS4_delong_internal.csv
  results/TableS4_delong_external.csv
  tables/TableS4.md

Pairwise comparisons:
  Internal : C(6,2) = 15 pairs
  External : C(7,2) = 21 pairs

The external table includes both FrozenR_Logistic and Python_Logistic
as distinct models (the frozen R implementation vs the sklearn
re-implementation).

Holm-adjusted p-values are added as a supplementary column
(for the 15 / 21 families separately).
"""
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
OUTDIR = os.path.join(BASE, "tables")
os.makedirs(OUTDIR, exist_ok=True)

INT_CSV = os.path.join(RESULTS_DIR, "ml_predictions_internal_v6.2.10.csv")
EXT_CSV = os.path.join(RESULTS_DIR, "ml_predictions_external_v6.2.10.csv")


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, p1, p2):
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    y_sorted = y_true[order]
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    m = int(y_sorted.sum())
    aucs, cov = _fastDeLong(preds, m)
    var = np.diag(cov)
    if var[0] + var[1] - 2 * cov[0, 1] <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0] + var[1] - 2 * cov[0, 1])
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, pvals[idx] * (n - i))
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def build_delong_table(df, models, cohort_label):
    y = df["outcome"].astype(int).values
    rows = []
    for m1, m2 in itertools.combinations(models, 2):
        a1, a2, z, p = delong_roc_test(y, df[m1].values, df[m2].values)
        rows.append({
            "Cohort":     cohort_label,
            "Model 1":    m1,
            "AUC 1":      round(a1, 4),
            "Model 2":    m2,
            "AUC 2":      round(a2, 4),
            "Delta AUC":  round(a1 - a2, 4),
            "z":          round(z, 4),
            "P (DeLong)": p,
        })
    out = pd.DataFrame(rows)
    out["P (Holm)"] = holm_adjust(out["P (DeLong)"].values)
    out["P (DeLong)"] = out["P (DeLong)"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    out["P (Holm)"] = out["P (Holm)"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")
    return out


def fmt_md_table(df, title, footnote):
    cols = list(df.columns)
    lines = [f"**{title}**", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append(footnote)
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Internal
# ============================================================
int_df = pd.read_csv(INT_CSV, comment="#")
int_models = ["Logistic", "RandomForest", "XGBoost", "LightGBM", "SVM", "MLP"]
int_models = [m for m in int_models if m in int_df.columns]
t_int = build_delong_table(int_df, int_models, "Internal validation")
t_int.to_csv(os.path.join(RESULTS_DIR, "TableS4_delong_internal.csv"), index=False)
print(f"[Table S4] internal: {len(t_int)} pairs")

# ============================================================
# External
# ============================================================
ext_df = pd.read_csv(EXT_CSV, comment="#")
ext_models = ["FrozenR_Logistic", "Python_Logistic",
              "RandomForest", "XGBoost", "LightGBM", "SVM", "MLP"]
ext_models = [m for m in ext_models if m in ext_df.columns]
t_ext = build_delong_table(ext_df, ext_models, "External validation (MIMIC-IV)")
t_ext.to_csv(os.path.join(RESULTS_DIR, "TableS4_delong_external.csv"), index=False)
print(f"[Table S4] external: {len(t_ext)} pairs")

# ============================================================
# Write Markdown
# ============================================================
footnote = (
    "AUC, area under the receiver operating characteristic curve; "
    "DeLong, DeLong's test for two correlated ROC curves; "
    "Holm, Holm–Bonferroni adjustment for multiple comparisons "
    "(15 tests within internal, 21 within external). "
    "Delta AUC is Model 1 minus Model 2. P values below 0.001 are "
    "reported as <0.001. The FrozenR_Logistic model uses the R "
    "deployment coefficients; Python_Logistic is an independent "
    "sklearn implementation and should not be treated as identical. "
    "Both the unadjusted and Holm-adjusted p values are reported; "
    "interpretation should prioritize the Holm-adjusted values to "
    "avoid over-interpretation of nominally significant results "
    "from multiple testing."
)

md = []
md.append("# Table S4. Pairwise comparisons of model discrimination using DeLong's test")
md.append("")
md.append(fmt_md_table(t_int,
                       "A. Internal validation cohort (n = 608, 72 events)",
                       ""))
md.append(fmt_md_table(t_ext,
                       "B. External validation cohort (MIMIC-IV, n = 434, 88 events)",
                       footnote))

with open(os.path.join(OUTDIR, "TableS4.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"[Table S4] written: {os.path.join(OUTDIR, 'TableS4.md')}")