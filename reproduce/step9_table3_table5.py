# -*- coding: utf-8 -*-
"""
step9_table3_table5.py  Table 3 / Table 5 重建与逐格比对（方案 A + 行名统一 HCO₃⁻）
"""
import os, numpy as np, pandas as pd
PROJ = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
F    = os.path.join(PROJ, "results", "FROZEN_v1")
OUT  = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT, exist_ok=True)
TOL  = 1e-3
def rd(fn): return pd.read_csv(os.path.join(F, fn))

# ---- 统一版本标签：BE 保持；HCO3 / HCO3- / HCO₃⁻ 一律显示 HCO₃⁻ ----
def norm_ver(v):
    s = str(v).upper().replace("₃","3").replace("-","")
    if s.startswith("HCO"): return "HCO₃⁻"
    return "BE"

summ    = rd("summary_for_manuscript.csv").set_index("Metric")["Value"]
int50   = rd("internal_validation_50imp.csv").set_index("Metric")
rec_dual= rd("recal_summary_dual.csv")
rec_v2  = rd("recal_bootstrap_summary_v2.csv")
eicu_be = rd("eicu_be_summary.csv").iloc[0]
eicu_h  = rd("eicu_hco3_summary.csv").iloc[0]
eicu_oob= rd("recal_bootstrap_eicu_summary.csv")

# ---------- Table 3 ----------
def row(name, n, ev, auc, brier, slope, inter):
    return dict(Row=name, n=n, events=ev, AUC=auc, Brier=brier, slope=slope, intercept=inter)
t3 = []
t3.append(row("Training", 1420, 154, float(summ["Train_AUC"]), float(summ["Train_Brier"]), None, None))
t3.append(row("Internal validation (50-imp mean)", 608, 72,
              float(int50.loc["AUC","Mean"]), float(int50.loc["Brier","Mean"]),
              float(int50.loc["Cal_slope","Mean"]), float(int50.loc["Cal_intercept","Mean"])))
for _, r in rec_dual.iterrows():
    t3.append(row(f"MIMIC-IV {norm_ver(r.Version)} {r.Stage}", int(r.n), int(r.events),
                  float(r.AUC), float(r.Brier), float(r.slope), float(r.intercept)))
for tag, r in [("BE", eicu_be), ("HCO₃⁻", eicu_h)]:
    t3.append(row(f"eICU-CRD {tag} Original", int(r.n), int(r.events),
                  float(r.AUC), float(r.Brier), float(r.cal_slope), float(r.cal_intercept)))
    t3.append(row(f"eICU-CRD {tag} Recalibrated", int(r.n), int(r.events),
                  float(r.AUC), float(r.Brier_recal), float(r.slope_recal), float(r.intercept_recal)))
T3 = pd.DataFrame(t3)
T3.to_csv(os.path.join(OUT, "Table3_rebuilt.csv"), index=False, encoding="utf-8-sig")
print("="*90); print("Table 3 重建（行名统一 HCO₃⁻）"); print("="*90); print(T3.to_string(index=False))

EXP3 = {
 "Training":                           (0.771, 0.087, None, None),
 "Internal validation (50-imp mean)":  (0.724, 0.099, 0.781, -0.380),
 "MIMIC-IV BE Original":               (0.622, 0.166, 0.420, -0.617),
 "MIMIC-IV BE Recalibrated":           (0.622, 0.158, 0.999, -0.002),
 "MIMIC-IV HCO₃⁻ Original":            (0.642, 0.137, 0.486, -0.809),
 "MIMIC-IV HCO₃⁻ Recalibrated":        (0.642, 0.131, 1.002, 0.004),
 "eICU-CRD BE Original":               (0.675, 0.176, 0.617, -0.484),
 "eICU-CRD BE Recalibrated":           (0.675, 0.171, 0.999, 0.000),
 "eICU-CRD HCO₃⁻ Original":            (0.676, 0.165, 0.642, -0.613),
 "eICU-CRD HCO₃⁻ Recalibrated":        (0.676, 0.158, 1.000, 0.000),
}
pass3=fail3=0; fails=[]
for _, r in T3.iterrows():
    e = EXP3.get(r.Row)
    if e is None: continue
    ok=True
    for col, ev in zip(["AUC","Brier","slope","intercept"], e):
        if ev is None: continue
        g=r[col]
        if g is None or (isinstance(g,float) and np.isnan(g)): ok=False; break
        if abs(float(g)-ev)>TOL: ok=False; break
    if ok: pass3+=1
    else: fail3+=1; fails.append(r.Row)
print(f"\nTable 3 比对：PASS {pass3} / FAIL {fail3}")
if fails: print("失败行：", fails)

# ---------- Table 5 ----------
t5=[]
def add5(ver, cohort, stage, n, ev, auc, brier, slope, inter):
    t5.append(dict(Version=f"{norm_ver(ver)} ({cohort})", Stage=stage, n=n, events=ev,
                   AUC=auc, Brier=brier, slope=slope, intercept=inter))
for _, r in rec_dual.iterrows():
    add5(r.Version,"MIMIC-IV", r.Stage, int(r.n), int(r.events),
         float(r.AUC), float(r.Brier), float(r.slope), float(r.intercept))
for _, r in rec_v2.iterrows():
    add5(r.Version,"MIMIC-IV","Recalibrated (bootstrap OOB)","—","—","—",
         float(r.Brier_oob_median), float(r.slope_oob_median), float(r.intercept_oob_median))
for tag, r in [("BE", eicu_be), ("HCO₃⁻", eicu_h)]:
    add5(tag,"eICU-CRD","Original", int(r.n), int(r.events), float(r.AUC), float(r.Brier), float(r.cal_slope), float(r.cal_intercept))
    add5(tag,"eICU-CRD","Recalibrated (apparent)", int(r.n), int(r.events), float(r.AUC), float(r.Brier_recal), float(r.slope_recal), float(r.intercept_recal))
for _, r in eicu_oob.iterrows():
    add5(r.Version,"eICU-CRD","Recalibrated (bootstrap OOB)", int(r.n), int(r.events), "—",
         float(r.Brier_OOB_median), float(r.slope_OOB_median), float(r.intercept_OOB_median))
T5=pd.DataFrame(t5); T5.to_csv(os.path.join(OUT,"Table5_rebuilt.csv"), index=False, encoding="utf-8-sig")
print("\n"+"="*90); print("Table 5 重建（12 行，行名统一 HCO₃⁻）"); print("="*90); print(T5.to_string(index=False))
print("\n[说明] 方案 A：Internal validation 行用 50 套均值；单次 7:3 主结果 0.712 见正文，冻结文件未含。")
print("Table 3/5 已保存到 output/（UTF-8-BOM，Excel 打开中文不乱码）。")