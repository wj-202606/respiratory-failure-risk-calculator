# -*- coding: utf-8 -*-
"""diag_find_calib.py  精确搜校准口径（含 results/FROZEN_v1）"""
import os, io
PROJ   = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
FROZEN = os.path.join(PROJ, "results", "FROZEN_v1")
KEYS = ["val.prob","calibrate","cal_slope","cal_intercept","Cal_slope","Cal_intercept",
        "LogisticRegression","C=1e6","C = 1e6","lp - mean","lp-mean","center","scale(lp)",
        "calib_slope","brier","Brier"]
EXTS = (".R",".r",".py",".md",".txt")

roots = [PROJ, FROZEN, os.path.join(PROJ,"results")]
targets = []
for root in roots:
    if not os.path.isdir(root): continue
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in (".git","__pycache__","node_modules")]
        for f in files:
            if f.lower().endswith(EXTS):
                targets.append(os.path.join(r,f))
targets = sorted(set(targets))
print("将搜索 %d 个文件（含 FROZEN_v1）" % len(targets))
for t in targets: print("  ", os.path.relpath(t, PROJ))

for path in targets:
    try:
        lines = io.open(path, encoding="utf-8", errors="ignore").read().splitlines()
    except Exception as e:
        print("\n[读失败]", path, e); continue
    hits = [i for i,l in enumerate(lines) if any(k.lower() in l.lower() for k in KEYS)]
    if not hits: continue
    print("\n" + "="*80)
    print("文件：", os.path.relpath(path, PROJ), " 命中", len(hits), "行")
    print("="*80)
    shown=set()
    for i in hits:
        lo,hi = max(0,i-2), min(len(lines),i+3)
        if any(j in shown for j in range(lo,hi)): continue
        for j in range(lo,hi):
            shown.add(j); mark = ">>" if j==i else "  "
            print(f"{mark} {j+1:5d}: {lines[j]}")
        print("-"*60)