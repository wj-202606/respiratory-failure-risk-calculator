# verify_table1.py
# 放在项目根目录，本地运行。数据不离开你的机器。
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
MIMIC_WIDE = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide_with_hco3.csv")
MIMIC_DEMO = r"D:\mimic_wide_with_demo.csv"
EXT_PRED   = os.path.join(BASE, "results", "external_predictions.csv")
T1_CSV     = os.path.join(BASE, "results", "Table1_v6_2_10.csv")

pred = pd.read_csv(EXT_PRED)
keep = set(pred["hadm_id"].astype(int).tolist())

wide = pd.read_csv(MIMIC_WIDE)
wide = wide[wide["hadm_id"].astype(int).isin(keep)]

demo = pd.read_csv(MIMIC_DEMO)[["hadm_id", "age", "male"]]
ext = wide.merge(demo, on="hadm_id", how="left", validate="one_to_one")
print(f"ext n = {len(ext)} (expect 434)")

age = pd.to_numeric(ext["age"], errors="coerce")
male = pd.to_numeric(ext["male"], errors="coerce")
print(f"external age mean (SD) = {age.mean():.1f} ({age.std(ddof=1):.1f})  (expect 64.1 (15.4))")
print(f"external male n (%)    = {int(male.sum())} ({100*male.mean():.1f})  (expect 246 (56.7))")

t1 = pd.read_csv(T1_CSV)
print()
print(t1.to_string(index=False))