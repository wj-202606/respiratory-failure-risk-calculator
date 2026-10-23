# -*- coding: utf-8 -*-
"""
Table S9: ALB measured vs not measured in the full MIMIC-IV cohort (n=2,456)
- 比较 ALB 测量组 vs 未测量组的死亡率、BUN、BE、年龄、性别
- 输出 results/TableS9_alb_missing.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 读全队列（含 ALB 缺失记录，不能 dropna）
MIMIC_CSV = r"D:\mimic_wide_with_demo.csv"
print(f"读 {MIMIC_CSV} ...")
d = pd.read_csv(MIMIC_CSV)
print(f"总行数: {len(d)}")
print(f"列: {d.columns.tolist()}")

# 确定列名
col_alb = "alb_g_l" if "alb_g_l" in d.columns else "ALB"
col_bun = "bun_mmol_l" if "bun_mmol_l" in d.columns else "BUN"
col_be  = "be_mmol_l" if "be_mmol_l" in d.columns else "BE"
col_out = "outcome" if "outcome" in d.columns else "Outcome"
col_age = "age" if "age" in d.columns else None
col_male = "male" if "male" in d.columns else ("sex" if "sex" in d.columns else None)

print(f"\n用列: ALB={col_alb}, BUN={col_bun}, BE={col_be}, outcome={col_out}, age={col_age}, male={col_male}")

# 只保留入组标准后的 2,456 例（如果 d 已经是入组后的，直接用）
# 如果 d 是原始 MIMIC-IV 全库，需先应用入组标准。这里假设 d 已经是 2,456 例的宽表。
# 如不是，请告知，我改。

# 分组：ALB 测量 vs 未测量
d["alb_measured"] = d[col_alb].notna()
n_meas = d["alb_measured"].sum()
n_not  = (~d["alb_measured"]).sum()
print(f"\nALB 测量: {n_meas}, 未测量: {n_not}")

# 统计函数
def smd_continuous(x1, x2):
    x1 = pd.to_numeric(x1, errors="coerce").dropna()
    x2 = pd.to_numeric(x2, errors="coerce").dropna()
    if len(x1) == 0 or len(x2) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    s_pool = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    smd = abs(x1.mean() - x2.mean()) / s_pool if s_pool > 0 else 0.0
    return smd, x1.mean(), x1.std(), x2.mean(), x2.std()

def smd_binary(p1, p2):
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return abs(p1 - p2) / denom if denom > 0 else 0.0

# 构建表
rows = []
meas = d[d["alb_measured"]]
notm = d[~d["alb_measured"]]

# n
rows.append({"Variable": "n", "ALB measured": str(len(meas)), "ALB not measured": str(len(notm)), "SMD": "", "P value": ""})

# Age
if col_age:
    smd, m1, s1, m2, s2 = smd_continuous(meas[col_age], notm[col_age])
    t, p = stats.ttest_ind(pd.to_numeric(meas[col_age], errors="coerce").dropna(),
                           pd.to_numeric(notm[col_age], errors="coerce").dropna(), equal_var=False)
    rows.append({"Variable": "Age, years, mean (SD)",
                 "ALB measured": f"{m1:.1f} ({s1:.1f})",
                 "ALB not measured": f"{m2:.1f} ({s2:.1f})",
                 "SMD": f"{smd:.2f}", "P value": "<0.001" if p < 0.001 else f"{p:.3f}"})

# Male
if col_male:
    def to_bin(s):
        if s.dtype == object:
            return s.astype(str).str.upper().isin(["M", "MALE", "1", "TRUE"]).astype(int)
        return (s == 1).astype(int)
    m1 = to_bin(meas[col_male]).mean()
    m2 = to_bin(notm[col_male]).mean()
    smd = smd_binary(m1, m2)
    # chi-square
    ct = pd.crosstab(pd.Series(["A"]*len(meas) + ["B"]*len(notm)),
                     pd.Series(list(to_bin(meas[col_male])) + list(to_bin(notm[col_male]))))
    chi2, p, _, _ = stats.chi2_contingency(ct)
    rows.append({"Variable": "Male, %",
                 "ALB measured": f"{m1*100:.1f}",
                 "ALB not measured": f"{m2*100:.1f}",
                 "SMD": f"{smd:.2f}", "P value": "<0.001" if p < 0.001 else f"{p:.3f}"})

# BUN
smd, m1, s1, m2, s2 = smd_continuous(meas[col_bun], notm[col_bun])
t, p = stats.ttest_ind(pd.to_numeric(meas[col_bun], errors="coerce").dropna(),
                       pd.to_numeric(notm[col_bun], errors="coerce").dropna(), equal_var=False)
rows.append({"Variable": "BUN, mmol/L, mean (SD)",
             "ALB measured": f"{m1:.2f} ({s1:.2f})",
             "ALB not measured": f"{m2:.2f} ({s2:.2f})",
             "SMD": f"{smd:.2f}", "P value": "<0.001" if p < 0.001 else f"{p:.3f}"})

# BE
smd, m1, s1, m2, s2 = smd_continuous(meas[col_be], notm[col_be])
t, p = stats.ttest_ind(pd.to_numeric(meas[col_be], errors="coerce").dropna(),
                       pd.to_numeric(notm[col_be], errors="coerce").dropna(), equal_var=False)
rows.append({"Variable": "BE, mmol/L, mean (SD)",
             "ALB measured": f"{m1:.2f} ({s1:.2f})",
             "ALB not measured": f"{m2:.2f} ({s2:.2f})",
             "SMD": f"{smd:.2f}", "P value": "<0.001" if p < 0.001 else f"{p:.3f}"})

# Outcome
p1 = meas[col_out].mean()
p2 = notm[col_out].mean()
smd = smd_binary(p1, p2)
ct = pd.crosstab(pd.Series(["A"]*len(meas) + ["B"]*len(notm)),
                 pd.Series(list(meas[col_out]) + list(notm[col_out])))
chi2, p, _, _ = stats.chi2_contingency(ct)
rows.append({"Variable": "In-hospital mortality, %",
             "ALB measured": f"{p1*100:.1f}",
             "ALB not measured": f"{p2*100:.1f}",
             "SMD": f"{smd:.2f}", "P value": "<0.001" if p < 0.001 else f"{p:.3f}"})

df = pd.DataFrame(rows)
print("\n" + "=" * 90)
print("Table S9. ALB measured vs not measured in the full MIMIC-IV cohort")
print("=" * 90)
print(df.to_string(index=False))

out = os.path.join(RESULTS_DIR, "TableS9_alb_missing.csv")
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存: {out}")