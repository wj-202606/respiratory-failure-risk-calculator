# -*- coding: utf-8 -*-
"""
eICU 外部验证队列清洗（与开发队列/MIMIC-IV 对齐）
- 呼吸衰竭：ICD-9 518.81 作为主要诊断（diagnosispriority == 'Primary'）
- 年龄 ≥18，住院 ≥24h，排除 24h 内死亡
- 每患者首次 ICU stay
输出:
  results/eicu_wide_be.csv    (ALB + BUN + BE 完整案例)
  results/eicu_wide_hco3.csv  (ALB + BUN + HCO3- 完整案例)
"""
import os
import pandas as pd
import numpy as np

PATH = r"D:\eicu数据库\EICU 2.0数据"
BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("eICU 外部验证队列构建")
print("=" * 70)

# ============================================================
# 1. patient.csv.gz —— 患者层
# ============================================================
print("\n[1/4] 读 patient.csv.gz ...")
pat = pd.read_csv(os.path.join(PATH, "patient.csv.gz"), usecols=[
    "patientunitstayid", "uniquepid", "patienthealthsystemstayid",
    "gender", "age", "hospitalid",
    "hospitaladmitoffset", "hospitaldischargeoffset",
    "hospitaldischargestatus", "unitvisitnumber", "unitstaytype",
])

print(f"  总 ICU stays: {len(pat)}")
print(f"  唯一 patients: {pat['uniquepid'].nunique()}")

def clean_age(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    if x == "> 89":
        return 90
    try:
        return float(x)
    except ValueError:
        return np.nan

pat["age_num"] = pat["age"].apply(clean_age)

# 去重：每患者首次 ICU stay（按 hospitaladmitoffset 最小）
pat = pat.sort_values(["uniquepid", "hospitaladmitoffset", "patientunitstayid"])
pat_first = pat.drop_duplicates(subset=["uniquepid"], keep="first").copy()
print(f"  去重后（每患者首次 ICU stay）: {len(pat_first)}")

# ============================================================
# 2. diagnosis.csv.gz —— 呼吸衰竭主要诊断
# ============================================================
print("\n[2/4] 读 diagnosis.csv.gz ...")
dx = pd.read_csv(os.path.join(PATH, "diagnosis.csv.gz"),
                 usecols=["patientunitstayid", "diagnosisstring", "icd9code", "diagnosispriority"])

print(f"  总诊断记录: {len(dx)}")
print(f"  diagnosispriority 分布: {dx['diagnosispriority'].value_counts().to_dict()}")

dx["icd9code"] = dx["icd9code"].astype(str)
mask_rf = dx["icd9code"].str.contains("518.81", na=False)
dx_rf = dx[mask_rf].copy()
print(f"  含 518.81 的记录: {len(dx_rf)}, 涉及 stays: {dx_rf['patientunitstayid'].nunique()}")

# 取主要诊断（Primary）
dx_rf_primary = dx_rf[dx_rf["diagnosispriority"].astype(str).str.lower() == "primary"].copy()
print(f"  518.81 作为 Primary: {len(dx_rf_primary)}, stays: {dx_rf_primary['patientunitstayid'].nunique()}")
if len(dx_rf_primary) == 0:
    raise ValueError("无 518.81 Primary 诊断，检查 ICD 代码或 priority 值")

dx_rf_primary = dx_rf_primary.drop_duplicates(subset=["patientunitstayid"], keep="first")
print(f"  dx 层最终 stays: {len(dx_rf_primary)}")

# ============================================================
# 3. 合并患者 + 诊断，应用入组标准
# ============================================================
print("\n[3/4] 合并 + 入组标准 ...")
cohort = pat_first.merge(dx_rf_primary[["patientunitstayid"]], on="patientunitstayid", how="inner")
print(f"  呼吸衰竭 + 患者层合并: {len(cohort)}")

cohort = cohort[cohort["age_num"] >= 18].copy()
print(f"  年龄 ≥18: {len(cohort)}")

cohort = cohort[cohort["hospitaldischargeoffset"] >= 1440].copy()
print(f"  住院 ≥24h: {len(cohort)}")

mask_early_death = (cohort["hospitaldischargestatus"] == "Expired") & (cohort["hospitaldischargeoffset"] < 1440)
cohort = cohort[~mask_early_death].copy()
print(f"  排除 24h 内死亡: {len(cohort)}")

cohort["outcome"] = (cohort["hospitaldischargestatus"] == "Expired").astype(int)
print(f"  院内死亡: {int(cohort['outcome'].sum())} ({cohort['outcome'].mean()*100:.1f}%)")

# ============================================================
# 4. lab.csv.gz —— 实验室指标
# ============================================================
print("\n[4/4] 读 lab.csv.gz（只读目标 labname + 24h 窗口）...")
lab = pd.read_csv(os.path.join(PATH, "lab.csv.gz"),
                  usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])

targets = ["albumin", "BUN", "Base Excess", "bicarbonate"]
lab = lab[lab["labname"].isin(targets)].copy()
print(f"  目标 labname 记录: {len(lab)}")

lab = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= 1440)].copy()
print(f"  24h 窗口内: {len(lab)}")

lab = lab.sort_values(["patientunitstayid", "labname", "labresultoffset"])
lab_first = lab.drop_duplicates(subset=["patientunitstayid", "labname"], keep="first")

wide = lab_first.pivot_table(index="patientunitstayid", columns="labname",
                             values="labresult", aggfunc="first").reset_index()
wide = wide.rename(columns={
    "albumin": "ALB_g_dL",
    "BUN": "BUN_mg_dL",
    "Base Excess": "BE",
    "bicarbonate": "HCO3",
})
print(f"  宽表 stays: {len(wide)}")
print(f"  列: {wide.columns.tolist()}")

# ============================================================
# 5. 合并 + 单位转换 + 异常值过滤
# ============================================================
merged = cohort.merge(wide, on="patientunitstayid", how="left")

merged["ALB"]  = merged["ALB_g_dL"] * 10.0        # g/dL → g/L
merged["BUN"]  = merged["BUN_mg_dL"] / 2.8        # mg/dL → mmol/L
merged["BE"]   = merged["BE"]                     # 已是 mmol/L
merged["HCO3"] = merged["HCO3"]                   # 已是 mmol/L

print("\n  异常值过滤:")
n0 = len(merged)
merged = merged[(merged["ALB"].isna()) | ((merged["ALB"] > 5) & (merged["ALB"] < 100))].copy()
print(f"    ALB 5-100 g/L: {len(merged)} / {n0}")
n1 = len(merged)
merged = merged[(merged["BUN"].isna()) | ((merged["BUN"] > 0.5) & (merged["BUN"] < 71))].copy()
print(f"    BUN 0.5-71 mmol/L: {len(merged)} / {n1}")
n2 = len(merged)
merged = merged[(merged["BE"].isna()) | ((merged["BE"] > -40) & (merged["BE"] < 40))].copy()
print(f"    BE -40 到 40: {len(merged)} / {n2}")
n3 = len(merged)
merged = merged[(merged["HCO3"].isna()) | ((merged["HCO3"] > 5) & (merged["HCO3"] < 60))].copy()
print(f"    HCO3 5-60: {len(merged)} / {n3}")

# ============================================================
# 6. 生成两个宽表
# ============================================================
be = merged.dropna(subset=["ALB", "BUN", "BE"]).copy()
print(f"\n  BE 版完整案例: n={len(be)}, 事件={int(be['outcome'].sum())} ({be['outcome'].mean()*100:.1f}%)")
be_out = be[["patientunitstayid", "ALB", "BUN", "BE", "outcome", "age_num", "gender", "hospitalid"]].copy()
be_out = be_out.rename(columns={"age_num": "age"})
be_out.to_csv(os.path.join(RESULTS_DIR, "eicu_wide_be.csv"), index=False)
print(f"  已保存: results/eicu_wide_be.csv")

hco3 = merged.dropna(subset=["ALB", "BUN", "HCO3"]).copy()
print(f"\n  HCO3 版完整案例: n={len(hco3)}, 事件={int(hco3['outcome'].sum())} ({hco3['outcome'].mean()*100:.1f}%)")
hco3_out = hco3[["patientunitstayid", "ALB", "BUN", "HCO3", "outcome", "age_num", "gender", "hospitalid"]].copy()
hco3_out = hco3_out.rename(columns={"age_num": "age"})
hco3_out.to_csv(os.path.join(RESULTS_DIR, "eicu_wide_hco3.csv"), index=False)
print(f"  已保存: results/eicu_wide_hco3.csv")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)