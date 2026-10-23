# -*- coding: utf-8 -*-
"""
make_boot_summary_v2.py
把 MIMIC-IV 的 1,000 行逐次 bootstrap 结果压成与 eICU 同结构的宽表：
每个 Version 一行，18 列。
输入：results/recal_bootstrap_BE.csv, results/recal_bootstrap_HCO3.csv
输出：results/recal_bootstrap_summary_v2.csv
"""
import os
import numpy as np
import pandas as pd

RESULTS_DIR = r"results"
INPUTS = [
    ("BE",   os.path.join(RESULTS_DIR, "recal_bootstrap_BE.csv")),
    ("HCO3", os.path.join(RESULTS_DIR, "recal_bootstrap_HCO3.csv")),
]
OUT = os.path.join(RESULTS_DIR, "recal_bootstrap_summary_v2.csv")

def med_q(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))

rows = []
for version, f in INPUTS:
    d = pd.read_csv(f)
    bm, bl, bu = med_q(d["Brier_oob"])
    sm, sl, su = med_q(d["slope_oob"])
    im, il, iu = med_q(d["intercept_oob"])
    rows.append(dict(
        Version=version, n_iter=int(d.shape[0]),
        Brier_oob_median=bm, Brier_oob_q25=bl, Brier_oob_q75=bu,
        slope_oob_median=sm, slope_oob_q25=sl, slope_oob_q75=su,
        intercept_oob_median=im, intercept_oob_q25=il, intercept_oob_q75=iu,
    ))
    print(f"[{version}] n_iter={len(d)}  Brier={bm:.4f} [{bl:.4f}-{bu:.4f}]  "
          f"slope={sm:.4f} [{sl:.4f}-{su:.4f}]  intercept={im:.4f} [{il:.4f}-{iu:.4f}]")

pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.4f")
print("Saved:", OUT)