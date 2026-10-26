# -*- coding: utf-8 -*-
"""
step5_figure3_S8_DCA.py  DCA Figure 3(MIMIC)/S8(eICU) + Table 6a/6b 自动比对
数据：dca_results_dual.csv / eicu_dca.csv
     列：Version( BE / HCO3 ), threshold, NB_original, NB_recalibrated, NB_treat_all, NB_treat_none
期望：Table 6a (MIMIC) / 6b (eICU)，16 组 × 3 列，容差 1e-3
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJ = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
F    = os.path.join(PROJ, "results", "FROZEN_v1")
OUT  = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT, exist_ok=True)

# ---------- 期望值（Table 6a / 6b） ----------
# key: (cohort, version, threshold_pct) -> (NB_original, NB_recalibrated, NB_treat_all)
EXPECT = {
 # MIMIC-IV (Table 6a)
 ("MIMIC","BE",10):   (0.0958, 0.1144, 0.1142),
 ("MIMIC","BE",15):   (0.0606, 0.0767, 0.0621),
 ("MIMIC","BE",20):   (0.0369, 0.0380, 0.0035),
 ("MIMIC","BE",30):   (0.0049, 0.0043,-0.1389),
 ("MIMIC","HCO3",10): (0.0684, 0.0713, 0.0660),
 ("MIMIC","HCO3",15): (0.0367, 0.0359, 0.0110),
 ("MIMIC","HCO3",20): (0.0175, 0.0161,-0.0508),
 ("MIMIC","HCO3",30): (-0.0042,-0.0054,-0.2009),
 # eICU-CRD (Table 6b)
 ("eICU","BE",10):    (0.1548, 0.1587, 0.1606),
 ("eICU","BE",15):    (0.1200, 0.1148, 0.1112),
 ("eICU","BE",20):    (0.0875, 0.0884, 0.0556),
 ("eICU","BE",30):    (0.0414, 0.0479,-0.0793),
 ("eICU","HCO3",10):  (0.1294, 0.1299, 0.1313),
 ("eICU","HCO3",15):  (0.0955, 0.0957, 0.0802),
 ("eICU","HCO3",20):  (0.0648, 0.0675, 0.0227),
 ("eICU","HCO3",30):  (0.0235, 0.0276,-0.1169),
}
TOL = 1e-3

# ---------- 读文件并统一 Version 标签 ----------
def load(fn):
    d = pd.read_csv(os.path.join(F, fn))
    # Version 可能是 BE / HCO3 / HCO₃⁻，统一成 BE / HCO3
    v = d["Version"].astype(str).str.upper().str.replace("₃","3",regex=False)
    v = v.str.replace("HCO3-","HCO3",regex=False).str.replace("HCO","HCO3",regex=False).str.replace("HCO33","HCO3",regex=False)
    d["Version"] = v
    # threshold 统一成整数百分比
    d["thr_pct"] = (d["threshold"].astype(float) * 100).round().astype(int)
    return d

files = {"MIMIC":"dca_results_dual.csv", "eICU":"eicu_dca.csv"}
data  = {c: load(fn) for c, fn in files.items()}

# ---------- 自动比对 ----------
print("="*86)
print("Table 6a / 6b 自动比对（容差 %.0e）" % TOL)
print("="*86)
pass_n = fail_n = 0
fail_rows = []
for (coh, ver, thr), (eo, er, et) in EXPECT.items():
    d = data[coh]
    sub = d[(d.Version == ver) & (d.thr_pct == thr)]
    if sub.empty:
        print(f"[MISS] {coh:<5} {ver:<5} {thr:>2}%  —— 文件中无此阈值行")
        fail_n += 1; fail_rows.append((coh,ver,thr,"MISSING")); continue
    r = sub.iloc[0]
    go, gr, gt = float(r.NB_original), float(r.NB_recalibrated), float(r.NB_treat_all)
    ok = abs(go-eo)<=TOL and abs(gr-er)<=TOL and abs(gt-et)<=TOL
    tag = "PASS" if ok else "FAIL"
    if ok: pass_n += 1
    else:  fail_n += 1; fail_rows.append((coh,ver,thr,(go,gr,gt),(eo,er,et)))
    print(f"[{tag}] {coh:<5} {ver:<5} {thr:>2}%  "
          f"orig {go:+.4f} (期望 {eo:+.4f})  "
          f"recal {gr:+.4f} (期望 {er:+.4f})  "
          f"treat-all {gt:+.4f} (期望 {et:+.4f})")

print("-"*86)
print(f"汇总：PASS {pass_n} / FAIL {fail_n} / 共 {len(EXPECT)} 组（每组 3 列）")
if fail_rows:
    print("\n失败明细：")
    for row in fail_rows: print("  ", row)
else:
    print("全部 16 组 × 3 列与 Table 6a/6b 一致。")

# ---------- 边界提示（论文正文明确写到的三个 10% 例外） ----------
print("\n关键边界核对（应与论文正文一致）：")
def show(coh, ver, thr):
    d = data[coh]; r = d[(d.Version==ver)&(d.thr_pct==thr)].iloc[0]
    print(f"  {coh} {ver} {thr}%: recal {r.NB_recalibrated:+.4f} vs treat-all {r.NB_treat_all:+.4f} "
          f"→ {'recal 低' if r.NB_recalibrated < r.NB_treat_all else 'recal 不低于 treat-all'}")
show("MIMIC","BE",10)    # 0.1144 vs 0.1142 差异可忽略
show("eICU","BE",10)     # 0.1587 < 0.1606
show("eICU","HCO3",10)   # 0.1299 < 0.1313

# ---------- 画图（修正 HCO3 命中） ----------
for coh in ["MIMIC","eICU"]:
    d = data[coh]
    fig, ax = plt.subplots(figsize=(8,6))
    for ver, col, lab in [("BE","C0","BE"), ("HCO3","C1","HCO3")]:
        sub = d[d.Version==ver].sort_values("threshold")
        if sub.empty:
            print(f"[警告] {coh} {ver} 无数据可画"); continue
        ax.plot(sub.threshold, sub.NB_original,   col+"-",  lw=1.5, label=f"{lab} original")
        ax.plot(sub.threshold, sub.NB_recalibrated, col+"--", lw=1.5, label=f"{lab} recal")
    ta = d[d.Version=="BE"].sort_values("threshold")
    if not ta.empty:
        ax.plot(ta.threshold, ta.NB_treat_all, ":", c="grey", lw=1.2, label="treat-all")
    ax.axvspan(0.10, 0.30, alpha=0.10, color="grey")
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.2, 0.25)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"{coh} decision curve analysis")
    ax.legend(fontsize=8)
    pth = os.path.join(OUT, f"Figure{'3' if coh=='MIMIC' else 'S8'}_DCA_reproduced.png")
    plt.tight_layout(); plt.savefig(pth, dpi=300); plt.savefig(pth.replace(".png",".pdf")); plt.close()
    print("图已保存：", pth)