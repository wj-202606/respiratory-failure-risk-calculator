# -*- coding: utf-8 -*-
"""step5_figure3_S8_DCA.py  DCA Figure 3(MIMIC)/S8(eICU) + Table 6a/6b 自动比对
数据：dca_results_dual.csv (F1) / eicu_dca.csv (F2)
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJ = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
F1 = os.path.join(PROJ, "results", "FROZEN_v1")  # MIMIC
F2 = os.path.join(PROJ, "results")               # eICU
OUT = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT, exist_ok=True)

# ---------- 期望值（Table 6a / 6b 方案 B） ----------
EXPECT = {
 # MIMIC-IV (Table 6a) - 不变
 ("MIMIC","BE",10):   (0.0958, 0.1144, 0.1142),
 ("MIMIC","BE",15):   (0.0606, 0.0767, 0.0621),
 ("MIMIC","BE",20):   (0.0369, 0.0380, 0.0035),
 ("MIMIC","BE",30):   (0.0049, 0.0043,-0.1389),
 ("MIMIC","HCO3",10): (0.0684, 0.0713, 0.0660),
 ("MIMIC","HCO3",15): (0.0367, 0.0359, 0.0110),
 ("MIMIC","HCO3",20): (0.0175, 0.0161,-0.0508),
 ("MIMIC","HCO3",30): (-0.0042,-0.0054,-0.2009),
 # eICU-CRD (Table 6b) - 方案 B 新值
 ("eICU","BE",10):    (0.1583, 0.1630, 0.1655),
 ("eICU","BE",15):    (0.1235, 0.1163, 0.1164),
 ("eICU","BE",20):    (0.0895, 0.0900, 0.0612),
 ("eICU","BE",30):    (0.0425, 0.0492,-0.0729),
 ("eICU","HCO3",10):  (0.1392, 0.1395, 0.1420),
 ("eICU","HCO3",15):  (0.1031, 0.0986, 0.0915),
 ("eICU","HCO3",20):  (0.0707, 0.0719, 0.0348),
 ("eICU","HCO3",30):  (0.0271, 0.0277,-0.1031),
}
TOL = 1e-3

def load(path):
    d = pd.read_csv(path)
    v = d["Version"].astype(str).str.upper().str.replace("₃","3",regex=False)
    v = v.str.replace("HCO3-","HCO3",regex=False).str.replace("HCO","HCO3",regex=False).str.replace("HCO33","HCO3",regex=False)
    d["Version"] = v
    d["thr_pct"] = (d["threshold"].astype(float) * 100).round().astype(int)
    return d

data = {"MIMIC": load(os.path.join(F1, "dca_results_dual.csv")),
        "eICU": load(os.path.join(F2, "eicu_dca.csv"))}

# ---------- 自动比对 ----------
print("="*86); print("Table 6a / 6b 自动比对（容差 %.0e）" % TOL); print("="*86)
pass_n = fail_n = 0
fail_rows = []
for (coh, ver, thr), (eo, er, et) in EXPECT.items():
    d = data[coh]
    sub = d[(d.Version == ver) & (d.thr_pct == thr)]
    if sub.empty:
        print(f"[MISS] {coh:<5} {ver:<5} {thr:>2}% —— 文件中无此阈值行")
        fail_n += 1; continue
    r = sub.iloc[0]
    go, gr, gt = float(r.NB_original), float(r.NB_recalibrated), float(r.NB_treat_all)
    ok = abs(go-eo)<=TOL and abs(gr-er)<=TOL and abs(gt-et)<=TOL
    if ok: pass_n += 1
    else:
        fail_n += 1
        fail_rows.append((coh,ver,thr,(go,gr,gt),(eo,er,et)))
    print(f"[{'PASS' if ok else 'FAIL'}] {coh:<5} {ver:<5} {thr:>2}%  "
          f"orig {go:+.4f} (期望 {eo:+.4f}) recal {gr:+.4f} (期望 {er:+.4f}) treat {gt:+.4f} (期望 {et:+.4f})")

print("-"*86)
print(f"汇总：PASS {pass_n} / FAIL {fail_n} / 共 {len(EXPECT)} 组")
if fail_rows:
    print("\n失败明细：")
    for row in fail_rows: print("  ", row)

# ---------- 画图 ----------
for coh, f_suf in [("MIMIC","3"), ("eICU","S8")]:
    d = data[coh]
    fig, ax = plt.subplots(figsize=(8,6))
    for ver, col, lab in [("BE","C0","BE"), ("HCO3","C1","HCO3")]:
        sub = d[d.Version==ver].sort_values("threshold")
        if sub.empty: continue
        ax.plot(sub.threshold, sub.NB_original,    col+"-",  lw=1.5, label=f"{lab} original")
        ax.plot(sub.threshold, sub.NB_recalibrated, col+"--", lw=1.5, label=f"{lab} recal")
    ta = d[d.Version=="BE"].sort_values("threshold")
    if not ta.empty: ax.plot(ta.threshold, ta.NB_treat_all, ":", c="grey", lw=1.2, label="treat-all")
    ax.axvspan(0.10, 0.30, alpha=0.10, color="grey")
    ax.set_xlim(0, 0.5); ax.set_ylim(-0.2, 0.25)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"{coh} decision curve analysis")
    ax.legend(fontsize=8)
    pth = os.path.join(OUT, f"Figure{f_suf}_DCA_reproduced.png")
    plt.tight_layout(); plt.savefig(pth, dpi=300); plt.savefig(pth.replace(".png",".pdf")); plt.close()
    print("图已保存：", pth)