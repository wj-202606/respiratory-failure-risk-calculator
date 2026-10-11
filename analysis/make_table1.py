# -*- coding: utf-8 -*-
"""
Table 1: 基线特征表（最终版）
- 4 栏：Development / Training / Internal val / External
- 含 P value（Development vs External）
- 输出 CSV + TXT
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================
# 路径
# ============================================================
BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(RESULTS_DIR, "traindata.csv")
VAL_CSV   = os.path.join(RESULTS_DIR, "valdata.csv")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_DEMO_CSV", r"D:\mimic_wide_with_demo.csv")

# ============================================================
# 读数据
# ============================================================
print("=" * 70)
print("生成 Table 1：基线特征")
print("=" * 70)

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
mimic = pd.read_csv(MIMIC_CSV)

dev = pd.concat([train, val], ignore_index=True)

print(f"train: {train.shape}")
print(f"val  : {val.shape}")
print(f"dev  : {dev.shape}")
print(f"mimic: {mimic.shape}")

# ============================================================
# 列名兼容
# ============================================================
def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_age_dev = get_col(dev, ["age", "Age", "AGE"])
col_sex_dev = get_col(dev, ["sex", "Sex", "SEX", "gender"])
col_alb_dev = get_col(dev, ["ALB", "alb"])
col_bun_dev = get_col(dev, ["BUN", "bun"])
col_be_dev  = get_col(dev, ["BE", "be"])
col_out_dev = get_col(dev, ["Outcome", "outcome"])

col_alb_mi  = get_col(mimic, ["alb_g_l", "ALB"])
col_bun_mi  = get_col(mimic, ["bun_mmol_l", "BUN"])
col_be_mi   = get_col(mimic, ["be_mmol_l", "BE"])
col_out_mi  = get_col(mimic, ["outcome", "Outcome"])
col_age_mi  = get_col(mimic, ["age", "Age"])
col_sex_mi  = get_col(mimic, ["male", "sex", "gender"])

# ============================================================
# 外部只保留三变量完整案例
# ============================================================
mimic_cc = mimic.dropna(subset=[col_alb_mi, col_bun_mi, col_be_mi, col_out_mi]).copy()

# ============================================================
# 辅助函数
# ============================================================
def fmt_mean_sd(x, decimals=2):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "N/A"
    return f"{x.mean():.{decimals}f} ({x.std():.{decimals}f})"

def fmt_median_iqr(x, decimals=1):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "N/A"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.{decimals}f} ({q1:.{decimals}f}–{q3:.{decimals}f})"

def fmt_n_pct(n, total, decimals=1):
    if total == 0:
        return "N/A"
    return f"{int(n)} ({100 * n / total:.{decimals}f})"

def p_value_continuous(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return "N/A"
    t, p = stats.ttest_ind(a, b, equal_var=False)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def p_value_categorical(a, b):
    """a, b 是 0/1 序列"""
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return "N/A"
    table = pd.crosstab(
        pd.Series(["A"] * len(a) + ["B"] * len(b)),
        pd.Series(list(a) + list(b))
    )
    chi2, p, dof, exp = stats.chi2_contingency(table)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# ============================================================
# 提取变量
# ============================================================
tr_alb = train[col_alb_dev]
tr_bun = train[col_bun_dev]
tr_be  = train[col_be_dev]
tr_out = train[col_out_dev]

va_alb = val[col_alb_dev]
va_bun = val[col_bun_dev]
va_be  = val[col_be_dev]
va_out = val[col_out_dev]

mi_alb = mimic_cc[col_alb_mi]
mi_bun = mimic_cc[col_bun_mi]
mi_be  = mimic_cc[col_be_mi]
mi_out = mimic_cc[col_out_mi]

# ============================================================
# 构建表格
# ============================================================
rows = []

# --- n ---
rows.append({
    "Variable": "n",
    "Development": str(len(dev)),
    "Training": str(len(train)),
    "Internal val": str(len(val)),
    "External": str(len(mimic_cc)),
    "P value": "",
})

# --- Age ---
if col_age_dev and col_age_mi:
    p_age = p_value_continuous(dev[col_age_dev], mimic_cc[col_age_mi])
    rows.append({
        "Variable": "Age, years, median (IQR)",
        "Development": fmt_median_iqr(dev[col_age_dev]),
        "Training": fmt_median_iqr(train[col_age_dev]),
        "Internal val": fmt_median_iqr(val[col_age_dev]),
        "External": fmt_median_iqr(mimic_cc[col_age_mi]),
        "P value": p_age,
    })

# --- Sex ---
def sex_binary(df, col):
    s = df[col]
    if s.dtype == object:
        return s.astype(str).str.upper().isin(["M", "MALE", "1"]).astype(int)
    return (s == 1).astype(int)

if col_sex_dev and col_sex_mi:
    dev_m = sex_binary(dev, col_sex_dev)
    tr_m  = sex_binary(train, col_sex_dev)
    va_m  = sex_binary(val, col_sex_dev)
    mi_m  = sex_binary(mimic_cc, col_sex_mi)
    p_sex = p_value_categorical(dev_m, mi_m)
    rows.append({
        "Variable": "Male, n (%)",
        "Development": fmt_n_pct(dev_m.sum(), len(dev)),
        "Training": fmt_n_pct(tr_m.sum(), len(train)),
        "Internal val": fmt_n_pct(va_m.sum(), len(val)),
        "External": fmt_n_pct(mi_m.sum(), len(mimic_cc)),
        "P value": p_sex,
    })

# --- ALB ---
p_alb = p_value_continuous(dev[col_alb_dev], mimic_cc[col_alb_mi])
rows.append({
    "Variable": "ALB, g/L, mean (SD)",
    "Development": fmt_mean_sd(dev[col_alb_dev]),
    "Training": fmt_mean_sd(tr_alb),
    "Internal val": fmt_mean_sd(va_alb),
    "External": fmt_mean_sd(mi_alb),
    "P value": p_alb,
})

# --- BUN ---
p_bun = p_value_continuous(dev[col_bun_dev], mimic_cc[col_bun_mi])
rows.append({
    "Variable": "BUN, mmol/L, mean (SD)",
    "Development": fmt_mean_sd(dev[col_bun_dev]),
    "Training": fmt_mean_sd(tr_bun),
    "Internal val": fmt_mean_sd(va_bun),
    "External": fmt_mean_sd(mi_bun),
    "P value": p_bun,
})

# --- BE ---
p_be = p_value_continuous(dev[col_be_dev], mimic_cc[col_be_mi])
rows.append({
    "Variable": "BE, mmol/L, mean (SD)",
    "Development": fmt_mean_sd(dev[col_be_dev]),
    "Training": fmt_mean_sd(tr_be),
    "Internal val": fmt_mean_sd(va_be),
    "External": fmt_mean_sd(mi_be),
    "P value": p_be,
})

# --- 缺失 ---
rows.append({
    "Variable": "ALB missing, n (%)",
    "Development": fmt_n_pct(dev[col_alb_dev].isnull().sum(), len(dev)),
    "Training": fmt_n_pct(tr_alb.isnull().sum(), len(train)),
    "Internal val": fmt_n_pct(va_alb.isnull().sum(), len(val)),
    "External": fmt_n_pct(mimic[col_alb_mi].isnull().sum(), len(mimic)),
    "P value": "",
})

rows.append({
    "Variable": "BUN missing, n (%)",
    "Development": fmt_n_pct(dev[col_bun_dev].isnull().sum(), len(dev)),
    "Training": fmt_n_pct(tr_bun.isnull().sum(), len(train)),
    "Internal val": fmt_n_pct(va_bun.isnull().sum(), len(val)),
    "External": fmt_n_pct(mimic[col_bun_mi].isnull().sum(), len(mimic)),
    "P value": "",
})

rows.append({
    "Variable": "BE missing, n (%)",
    "Development": fmt_n_pct(dev[col_be_dev].isnull().sum(), len(dev)),
    "Training": fmt_n_pct(tr_be.isnull().sum(), len(train)),
    "Internal val": fmt_n_pct(va_be.isnull().sum(), len(val)),
    "External": fmt_n_pct(mimic[col_be_mi].isnull().sum(), len(mimic)),
    "P value": "",
})

# --- 结局 ---
p_out = p_value_categorical(dev[col_out_dev], mimic_cc[col_out_mi])
rows.append({
    "Variable": "In-hospital death, n (%)",
    "Development": fmt_n_pct(dev[col_out_dev].sum(), len(dev)),
    "Training": fmt_n_pct(tr_out.sum(), len(train)),
    "Internal val": fmt_n_pct(va_out.sum(), len(val)),
    "External": fmt_n_pct(mi_out.sum(), len(mimic_cc)),
    "P value": p_out,
})

# ============================================================
# 输出
# ============================================================
table1 = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("Table 1. Baseline characteristics")
print("=" * 70)
print(table1.to_string(index=False))

# 保存 CSV
out_csv = os.path.join(RESULTS_DIR, "Table1.csv")
table1.to_csv(out_csv, index=False, encoding="utf-8-sig")

# 保存 TXT（含脚注）
out_txt = os.path.join(RESULTS_DIR, "Table1.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("Table 1. Baseline characteristics of the study cohorts\n")
    f.write("=" * 110 + "\n")
    f.write(table1.to_string(index=False))
    f.write("\n\n")
    f.write("Abbreviations: ALB, albumin; BUN, blood urea nitrogen; BE, base excess; "
            "IQR, interquartile range; SD, standard deviation.\n")
    f.write("Continuous variables are presented as mean (SD) or median (IQR); "
            "categorical variables as n (%).\n")
    f.write("P values compare the development cohort and the external "
            "validation cohort (Welch's t-test for continuous, chi-square test "
            "for categorical).\n")
    f.write("The external validation cohort includes only patients with "
            "complete data on all three predictors.\n")

print(f"\n✅ 已保存:")
print(f"   {out_csv}")
print(f"   {out_txt}")
print("\n完成！")