# -*- coding: utf-8 -*-
"""从 eICU 原始 csv.gz 构建全队列（目标 n=5,099），供 S12/S13 使用。
入组逻辑与 eicu_build_wide.py 完全同源，但不要求 ALB/BUN/BE/HCO3 完整。
"""
import os
import pandas as pd
import numpy as np

PATH = r"D:\eicu数据库\EICU 2.0数据"
BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("eICU 全队列构建（与 eicu_build_wide.py 同源）")
print("=" * 70)

# ============================================================
# 1. patient.csv.gz
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

# 去重：每患者首次 ICU stay（按 hospitaladmitoffset 最小）—— 与 wide 脚本一致
pat = pat.sort_values(["uniquepid", "hospitaladmitoffset", "patientunitstayid"])
pat_first = pat.drop_duplicates(subset=["uniquepid"], keep="first").copy()
print(f"  去重后（每患者首次 ICU stay）: {len(pat_first)}")

# ============================================================
# 2. diagnosis.csv.gz
# ============================================================
print("\n[2/4] 读 diagnosis.csv.gz ...")
dx = pd.read_csv(os.path.join(PATH, "diagnosis.csv.gz"),
                 usecols=["patientunitstayid", "diagnosisstring", "icd9code", "diagnosispriority"])
print(f"  总诊断记录: {len(dx)}")

dx["icd9code"] = dx["icd9code"].astype(str)
mask_rf = dx["icd9code"].str.contains("518.81", na=False)
dx_rf = dx[mask_rf].copy()
print(f"  含 518.81 的记录: {len(dx_rf)}, 涉及 stays: {dx_rf['patientunitstayid'].nunique()}")

dx_rf_primary = dx_rf[dx_rf["diagnosispriority"].astype(str).str.lower() == "primary"].copy()
print(f"  518.81 作为 Primary: {len(dx_rf_primary)}, stays: {dx_rf_primary['patientunitstayid'].nunique()}")
if len(dx_rf_primary) == 0:
    raise ValueError("无 518.81 Primary 诊断，检查 ICD 代码或 priority 值")

dx_rf_primary = dx_rf_primary.drop_duplicates(subset=["patientunitstayid"], keep="first")
print(f"  dx 层最终 stays: {len(dx_rf_primary)}")

# ============================================================
# 3. 合并 + 入组标准
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
# 4. lab.csv.gz —— 允许缺失，不要求完整
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
# 5. 合并 + 单位转换 + 异常值过滤（与 wide 脚本一致）
# ============================================================
merged = cohort.merge(wide, on="patientunitstayid", how="left")

merged["ALB"]  = merged["ALB_g_dL"] * 10.0
merged["BUN"]  = merged["BUN_mg_dL"] / 2.8
merged["BE"]   = merged["BE"]
merged["HCO3"] = merged["HCO3"]

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
# 6. 保存全队列
# ============================================================
full = merged[["patientunitstayid", "uniquepid", "age_num", "gender",
               "hospitaldischargestatus", "outcome",
               "ALB", "BUN", "BE", "HCO3", "hospitalid"]].copy()
full = full.rename(columns={"age_num": "age", "gender": "male"})
full["male"] = full["male"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})
full = full.rename(columns={"ALB": "alb", "BUN": "bun", "BE": "be", "HCO3": "hco3"})

full.to_csv(os.path.join(RESULTS_DIR, "eicu_full_cohort.csv"), index=False)

print("\n" + "=" * 70)
print(f"全队列 n = {len(full)}，死亡 = {int(full['outcome'].sum())}")
print(f"ALB 覆盖率 = {full['alb'].notna().mean()*100:.1f}%")
print(f"BUN 覆盖率 = {full['bun'].notna().mean()*100:.1f}%")
print(f"BE 覆盖率 = {full['be'].notna().mean()*100:.1f}%")
print(f"HCO3 覆盖率 = {full['hco3'].notna().mean()*100:.1f}%")
print("=" * 70)