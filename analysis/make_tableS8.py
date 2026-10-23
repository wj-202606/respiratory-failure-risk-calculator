# -*- coding: utf-8 -*-
"""
make_tableS8.py (v2)
Table S8: 外部队列完整案例 (n=434) vs 排除患者 (n=2,022) 对比
- 从 mimic_wide.csv 取 ALB/BUN/BE/outcome
- 从 mimic_wide_with_demo.csv 取 age/male
- 按 hadm_id merge

输入：
  D:/mimic_wide.csv             （ALB/BUN/BE/outcome）
  D:/mimic_wide_with_demo.csv   （含 age/male）
输出：
  results/TableS8_inclusion_vs_exclusion.csv
  results/TableS8.txt
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# ⚠️ mimic_wide.csv 的 be_mmol_l 是真 BE（非 HCO3-）。
# 勿与 mimic_wide_hco3.csv 混淆（后者 be_mmol_l 存的是 HCO3-）。

MIMIC_CSV      = os.environ.get("MIMIC_WIDE_CSV",      r"D:\mimic_wide.csv")
MIMIC_DEMO_CSV = os.environ.get("MIMIC_WIDE_DEMO_CSV", r"D:\mimic_wide_with_demo.csv")

print("=" * 70)
print("Table S8: 纳入 vs 排除对比")
print("=" * 70)

# ============================================================
# 读两个文件
# ============================================================
d = pd.read_csv(MIMIC_CSV)
print(f"mimic_wide.csv: n = {len(d)}")
print(f"  列: {d.columns.tolist()}")

demo = pd.read_csv(MIMIC_DEMO_CSV)
print(f"mimic_wide_with_demo.csv: n = {len(demo)}")
print(f"  列: {demo.columns.tolist()}")

# ============================================================
# 确定 demo 里的 age / male 列名
# ============================================================
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

col_age  = find_col(demo, ["age", "Age", "AGE"])
col_male = find_col(demo, ["male", "Male", "MALE", "sex", "Sex", "gender"])
col_hadm = find_col(demo, ["hadm_id", "hadm", "HADM_ID"])

print(f"\nDemo 列匹配:")
print(f"  age  -> {col_age}")
print(f"  male -> {col_male}")
print(f"  hadm -> {col_hadm}")

if col_hadm is None:
    raise KeyError("demo 表没有 hadm_id，无法 merge")

# ============================================================
# Merge（只保留我们需要的列）
# ============================================================
keep_cols = [col_hadm]
if col_age:  keep_cols.append(col_age)
if col_male: keep_cols.append(col_male)

demo_sub = demo[keep_cols].copy()
demo_sub = demo_sub.rename(columns={col_hadm: "hadm_id"})
if col_age:  demo_sub = demo_sub.rename(columns={col_age: "age"})
if col_male: demo_sub = demo_sub.rename(columns={col_male: "male"})

# 若 demo 里 hadm_id 重复，去重
demo_sub = demo_sub.drop_duplicates(subset=["hadm_id"], keep="first")

merged = d.merge(demo_sub, on="hadm_id", how="left")
print(f"\nmerge 后: n = {len(merged)}")
print(f"  age 覆盖率:  {merged['age'].notna().mean()*100:.1f}%" if "age" in merged.columns else "  age 不存在")
print(f"  male 覆盖率: {merged['male'].notna().mean()*100:.1f}%" if "male" in merged.columns else "  male 不存在")

# ============================================================
# 纳入 vs 排除
# ============================================================
need = ["alb_g_l", "bun_mmol_l", "be_mmol_l", "outcome"]
missing_cols = [c for c in need if c not in merged.columns]
if missing_cols:
    raise KeyError(f"缺少列: {missing_cols}")

cc_mask = merged[["alb_g_l", "bun_mmol_l", "be_mmol_l"]].notna().all(axis=1)
d_inc = merged[cc_mask].copy()
d_exc = merged[~cc_mask].copy()

print(f"\n纳入 (完整案例): n = {len(d_inc)}, 事件 = {int(d_inc['outcome'].sum())}")
print(f"排除: n = {len(d_exc)}, 事件 = {int(d_exc['outcome'].sum())}")

# ============================================================
# 辅助函数
# ============================================================
def fmt_mean_sd(x, decimals=2):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return "N/A"
    return f"{x.mean():.{decimals}f} ({x.std():.{decimals}f})"

def fmt_n_pct(n, total, decimals=1):
    if total == 0:
        return "N/A"
    return f"{int(n)} ({100 * n / total:.{decimals}f})"

def smd_continuous(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    s_pool = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if s_pool == 0:
        return 0.0
    return abs(a.mean() - b.mean()) / s_pool

def smd_binary(p1, p2):
    if np.isnan(p1) or np.isnan(p2):
        return np.nan
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if denom == 0:
        return 0.0
    return abs(p1 - p2) / denom

def p_continuous(a, b):
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return "N/A"
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return "<0.001" if p < 0.001 else f"{p:.3f}"

def p_categorical(a, b):
    a = pd.Series(a).dropna()
    b = pd.Series(b).dropna()
    if len(a) == 0 or len(b) == 0:
        return "N/A"
    try:
        ct = pd.crosstab(
            pd.Series(["A"] * len(a) + ["B"] * len(b)),
            pd.Series(list(a) + list(b))
        )
        chi2, p, _, _ = stats.chi2_contingency(ct)
        return "<0.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return "N/A"

# ============================================================
# 构建表
# ============================================================
rows = []

# n
rows.append({
    "Variable": "n",
    "Included (n=434)": str(len(d_inc)),
    "Excluded (n=2,022)": str(len(d_exc)),
    "SMD": "",
    "P value": "",
})

# Age
if "age" in merged.columns:
    smd_age = smd_continuous(d_inc["age"], d_exc["age"])
    p_age = p_continuous(d_inc["age"], d_exc["age"])
    rows.append({
        "Variable": "Age, years, mean (SD)",
        "Included (n=434)": fmt_mean_sd(d_inc["age"], 1),
        "Excluded (n=2,022)": fmt_mean_sd(d_exc["age"], 1),
        "SMD": f"{smd_age:.2f}",
        "P value": p_age,
    })

# Male
if "male" in merged.columns:
    # 统一转 0/1
    def to_bin(s):
        s = s.copy()
        if s.dtype == object:
            return s.astype(str).str.upper().isin(["M", "MALE", "1", "TRUE"]).astype(int)
        return (s == 1).astype(int)
    inc_m = to_bin(d_inc["male"])
    exc_m = to_bin(d_exc["male"])
    p1 = inc_m.mean()
    p2 = exc_m.mean()
    smd_male = smd_binary(p1, p2)
    p_str = p_categorical(inc_m, exc_m)
    rows.append({
        "Variable": "Male, n (%)",
        "Included (n=434)": fmt_n_pct(inc_m.sum(), len(inc_m)),
        "Excluded (n=2,022)": fmt_n_pct(exc_m.sum(), len(exc_m)),
        "SMD": f"{smd_male:.2f}",
        "P value": p_str,
    })

# BUN
smd_bun = smd_continuous(d_inc["bun_mmol_l"], d_exc["bun_mmol_l"])
p_bun = p_continuous(d_inc["bun_mmol_l"], d_exc["bun_mmol_l"])
rows.append({
    "Variable": "BUN, mmol/L, mean (SD)",
    "Included (n=434)": fmt_mean_sd(d_inc["bun_mmol_l"], 2),
    "Excluded (n=2,022)": fmt_mean_sd(d_exc["bun_mmol_l"], 2),
    "SMD": f"{smd_bun:.2f}",
    "P value": p_bun,
})

# BE
smd_be = smd_continuous(d_inc["be_mmol_l"], d_exc["be_mmol_l"])
p_be = p_continuous(d_inc["be_mmol_l"], d_exc["be_mmol_l"])
rows.append({
    "Variable": "BE, mmol/L, mean (SD)",
    "Included (n=434)": fmt_mean_sd(d_inc["be_mmol_l"], 2),
    "Excluded (n=2,022)": fmt_mean_sd(d_exc["be_mmol_l"], 2),
    "SMD": f"{smd_be:.2f}",
    "P value": p_be,
})

# HCO3（如存在）
if "hco3_mmol_l" in merged.columns:
    smd_hco3 = smd_continuous(d_inc["hco3_mmol_l"], d_exc["hco3_mmol_l"])
    p_hco3 = p_continuous(d_inc["hco3_mmol_l"], d_exc["hco3_mmol_l"])
    rows.append({
        "Variable": "HCO₃⁻, mmol/L, mean (SD)",
        "Included (n=434)": fmt_mean_sd(d_inc["hco3_mmol_l"], 2),
        "Excluded (n=2,022)": fmt_mean_sd(d_exc["hco3_mmol_l"], 2),
        "SMD": f"{smd_hco3:.2f}",
        "P value": p_hco3,
    })

# 院内死亡
p1_out = d_inc["outcome"].mean()
p2_out = d_exc["outcome"].mean()
smd_out = smd_binary(p1_out, p2_out)
p_str = p_categorical(d_inc["outcome"], d_exc["outcome"])
rows.append({
    "Variable": "In-hospital mortality, n (%)",
    "Included (n=434)": fmt_n_pct(d_inc["outcome"].sum(), len(d_inc)),
    "Excluded (n=2,022)": fmt_n_pct(d_exc["outcome"].sum(), len(d_exc)),
    "SMD": f"{smd_out:.2f}",
    "P value": p_str,
})

# 保存
df = pd.DataFrame(rows)
print("\n" + "=" * 100)
print("Table S8. Baseline comparison — included vs excluded patients")
print("=" * 100)
print(df.to_string(index=False))

out_csv = os.path.join(RESULTS_DIR, "TableS8_inclusion_vs_exclusion.csv")
df.to_csv(out_csv, index=False, encoding="utf-8-sig")

out_txt = os.path.join(RESULTS_DIR, "TableS8.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    f.write("Table S8. Baseline characteristics and in-hospital mortality — "
            "patients included in the primary external validation (complete cases, n = 434) "
            "vs patients excluded due to missing ALB, BUN, or BE (n = 2,022)\n")
    f.write("=" * 110 + "\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")
    f.write("SMD, standardized mean difference; ALB, albumin; BUN, blood urea nitrogen; "
            "BE, base excess; HCO₃⁻, bicarbonate; SD, standard deviation; N/A, not applicable.\n")
    f.write("SMDs are reported as absolute values. For continuous variables, "
            "SMD = |mean₁ − mean₂| / pooled SD. For binary variables, "
            f"SMD = |p₁ − p₂| / √[(p₁(1−p₁) + p₂(1−p₂))/2]. "
            "An SMD above 0.10 is conventionally considered non-negligible.\n")
    f.write("P values: Welch's t-test for continuous; chi-square test for categorical.\n")

print(f"\n✅ 已保存:")
print(f"   {out_csv}")
print(f"   {out_txt}")