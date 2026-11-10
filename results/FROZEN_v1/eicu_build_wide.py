# -*- coding: utf-8 -*-
"""
eICU 外部验证队列清洗（方案 B：以 ICU 入科为时间起点）v6.0

相对 v4.x：
  [1] 统一输出列名为大写：ALB / BUN / BE / HCO3
  [2] gender 字符串改为 male（0/1），与 eicu_full_cohort_v2.csv 一致
  [3] eicu_full_cohort_v2.csv 去掉 hospitaldischargestatus，列名统一大写
  [4] 文件末尾加输出契约说明，供下游脚本对照

队列定义（与 v4.x 一致）：
  - 呼吸衰竭：ICD-9 518.81 作为主要诊断（diagnosispriority == 'Primary'）
  - 年龄 ≥18，存活至 ICU 入科后 24h，排除 ICU 24h 内死亡
  - 每患者首次 ICU stay
  - 预测时点 = ICU 入科后 24h；预测变量窗口 = 0 ≤ labresultoffset ≤ 1440
  - 结局 = ICU 24h 后至出院前院内全因死亡

输出：
  results/eicu_wide_be.csv
    列：patientunitstayid, ALB, BUN, BE, outcome, age, male, hospitalid
    单位：ALB g/L；BUN mmol/L；BE mmol/L
  results/eicu_wide_hco3.csv
    列：patientunitstayid, ALB, BUN, HCO3, outcome, age, male, hospitalid
    单位：ALB g/L；BUN mmol/L；HCO3 mmol/L
  results/eicu_full_cohort_v2.csv
    列：patientunitstayid, uniquepid, age, male, outcome,
        ALB, BUN, BE, HCO3, hospitalid
    含预测因子缺失者，供 S11/S12/S13
"""
import os
import pandas as pd
import numpy as np

PATH = os.environ.get("EICU_PATH", r"D:\eicu2.0")
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("PROJECT_BASE", os.path.dirname(_HERE) if os.path.basename(_HERE) == "analysis" else _HERE)
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("eICU 外部验证队列构建（方案 B：ICU 入科为时间起点）v6.2")
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
    "unitdischargeoffset", "unitdischargestatus",
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

# 暂不提前按患者去重；先识别符合呼吸衰竭诊断的stay，再选择首次符合条件的stay。

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

dx_rf_primary = dx_rf[dx_rf["diagnosispriority"].astype(str).str.lower() == "primary"].copy()
print(f"  518.81 作为 Primary: {len(dx_rf_primary)}, stays: {dx_rf_primary['patientunitstayid'].nunique()}")
if len(dx_rf_primary) == 0:
    raise ValueError("无 518.81 Primary 诊断，检查 ICD 代码或 priority 值")

dx_rf_primary = dx_rf_primary.drop_duplicates(subset=["patientunitstayid"], keep="first")
print(f"  dx 层最终 stays: {len(dx_rf_primary)}")

# ============================================================
# 3. 合并患者 + 诊断，应用入组标准（方案 B：ICU 口径）
# ============================================================
print("\n[3/4] 合并 + 入组标准（ICU 口径）...")
cohort = pat.merge(dx_rf_primary[["patientunitstayid"]], on="patientunitstayid", how="inner")
print(f"  呼吸衰竭 + 患者层合并: {len(cohort)}")
cohort = cohort.sort_values(["uniquepid", "hospitaladmitoffset", "patientunitstayid"])
cohort = cohort.drop_duplicates(subset=["uniquepid"], keep="first").copy()
print(f"  每位患者首次符合呼吸衰竭诊断的ICU stay: {len(cohort)}")

cohort = cohort[cohort["age_num"] >= 18].copy()
print(f"  年龄 ≥18: {len(cohort)}")

# ICU 24h landmark：仅纳入ICU停留超过1440分钟者。
# 这也会排除24h内转出ICU的患者，需在Methods明确，不等同于“仅排除早期死亡”。
cohort["unitdischargeoffset"] = pd.to_numeric(cohort["unitdischargeoffset"], errors="coerce")
n_pre_landmark = len(cohort)
cohort = cohort[cohort["unitdischargeoffset"] > 1440].copy()
print(f"  ICU 24h landmark（unitdischargeoffset > 1440）: {len(cohort)}")
print(f"  ICU停留≤24h排除: {n_pre_landmark-len(cohort)}")

if cohort["hospitaldischargestatus"].isna().any():
    raise ValueError("入组队列 hospitaldischargestatus 存在缺失，不能默认编码为存活")
status = cohort["hospitaldischargestatus"].astype(str).str.strip().str.lower()
valid_status = status.isin(["expired", "alive"])
if not valid_status.all():
    raise ValueError(f"hospitaldischargestatus存在非标准取值: {sorted(status[~valid_status].unique())}")
cohort["outcome"] = status.eq("expired").astype(int)
print(f"  院内死亡: {int(cohort['outcome'].sum())} ({cohort['outcome'].mean()*100:.1f}%)")

# ============================================================
# 4. lab.csv.gz —— 实验室指标（ICU 入科后 24h 窗口）
# ============================================================
print("\n[4/4] 读 lab.csv.gz（只读目标 labname + ICU 后 24h 窗口）...")
lab = pd.read_csv(os.path.join(PATH, "lab.csv.gz"),
                  usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])

targets = ["albumin", "BUN", "Base Excess", "bicarbonate"]
lab = lab[lab["labname"].isin(targets)].copy()
print(f"  目标 labname 记录: {len(lab)}")

lab = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= 1440)].copy()
print(f"  ICU 后 24h 窗口内: {len(lab)}")

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
# 5. 合并 + 单位转换
# ============================================================
merged = cohort.merge(wide, on="patientunitstayid", how="left")

# 单位转换：ALB g/dL -> g/L；BUN mg/dL -> mmol/L；BE / HCO3 已是 mmol/L
merged["ALB"]  = merged["ALB_g_dL"] * 10.0
merged["BUN"]  = merged["BUN_mg_dL"] / 2.8
merged["BE"]   = merged["BE"]
merged["HCO3"] = merged["HCO3"]

# 性别统一为 male（0/1）
merged["male"] = (merged["gender"].astype(str).str.upper() == "MALE").astype(float)

# ============================================================
# 5b. 保存全队列（过滤前，含预测因子缺失者）
# ============================================================
_full = merged.copy()
_full = _full.rename(columns={"age_num": "age"})
_full = _full[[
    "patientunitstayid", "uniquepid", "age", "male", "outcome",
    "ALB", "BUN", "BE", "HCO3", "hospitalid"
]].copy()
_full["age"] = pd.to_numeric(_full["age"], errors="coerce")

_full.to_csv(os.path.join(RESULTS_DIR, "eicu_full_cohort_v2.csv"), index=False)
print(f"  已保存中间表: results/eicu_full_cohort_v2.csv (n={len(_full)})")

# ============================================================
# 6. 异常值过滤
# ============================================================
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
# 7. 生成两个宽表（统一大写列名）
# ============================================================
OUT_COLS_BASE = ["patientunitstayid", "ALB", "BUN", "outcome", "age", "male", "hospitalid"]

be = merged.dropna(subset=["ALB", "BUN", "BE"]).copy()
print(f"\n  BE 版完整案例: n={len(be)}, 事件={int(be['outcome'].sum())} "
      f"({be['outcome'].mean()*100:.1f}%)")
be_out = be[["patientunitstayid", "ALB", "BUN", "BE", "outcome",
             "age_num", "male", "hospitalid"]].copy()
be_out = be_out.rename(columns={"age_num": "age"})
be_out = be_out[["patientunitstayid", "ALB", "BUN", "BE", "outcome",
                 "age", "male", "hospitalid"]]
be_out.to_csv(os.path.join(RESULTS_DIR, "eicu_wide_be.csv"), index=False)
print(f"  已保存: results/eicu_wide_be.csv")

hco3 = merged.dropna(subset=["ALB", "BUN", "HCO3"]).copy()
print(f"\n  HCO3 版完整案例: n={len(hco3)}, 事件={int(hco3['outcome'].sum())} "
      f"({hco3['outcome'].mean()*100:.1f}%)")
hco3_out = hco3[["patientunitstayid", "ALB", "BUN", "HCO3", "outcome",
                 "age_num", "male", "hospitalid"]].copy()
hco3_out = hco3_out.rename(columns={"age_num": "age"})
hco3_out = hco3_out[["patientunitstayid", "ALB", "BUN", "HCO3", "outcome",
                     "age", "male", "hospitalid"]]
hco3_out.to_csv(os.path.join(RESULTS_DIR, "eicu_wide_hco3.csv"), index=False)
print(f"  已保存: results/eicu_wide_hco3.csv")

# ============================================================
# 8. 输出契约（供下游脚本对照）
# ============================================================
print("\n" + "=" * 70)
print("输出契约")
print("=" * 70)
print("eicu_wide_be.csv:")
print("  patientunitstayid, ALB, BUN, BE, outcome, age, male, hospitalid")
print("  单位: ALB g/L; BUN mmol/L; BE mmol/L")
print("eicu_wide_hco3.csv:")
print("  patientunitstayid, ALB, BUN, HCO3, outcome, age, male, hospitalid")
print("  单位: ALB g/L; BUN mmol/L; HCO3 mmol/L")
print("eicu_full_cohort_v2.csv:")
print("  patientunitstayid, uniquepid, age, male, outcome, ALB, BUN, BE, HCO3, hospitalid")
print("  含预测因子缺失者，供 S11/S12/S13")
print("\n下游脚本可直接使用大写列名，无需 rename 兜底。")
print("=" * 70)
print("完成！")
print("=" * 70)