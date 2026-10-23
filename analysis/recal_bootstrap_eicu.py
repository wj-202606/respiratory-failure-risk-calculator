# -*- coding: utf-8 -*-
"""
recal_bootstrap_eicu.py
eICU-CRD (BE 与 HCO3) 的 1000 次 bootstrap 重校准 + OOB 评估
口径与 recal_bootstrap.py (MIMIC-IV) 完全一致：
  - logit(p) 未中心化（减均值与否不影响斜率，但影响截距参考点；此处与 MIMIC-IV 一致：不中心化）
  - 校准斜率/截距用 LogisticRegression(C=1e6) 等价于无惩罚最大似然
  - bootstrap 整数索引有放回；OOB = 未抽中的样本
输出：
  results/recal_bootstrap_eicu_BE.csv   (1,000 行逐次结果)
  results/recal_bootstrap_eicu_HCO3.csv
  results/recal_bootstrap_eicu_summary.csv  (median + q25 + q75)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------- 配置 ----------------
RESULTS_DIR = r"results"
FROZEN_DIR  = r"results/FROZEN_v1"

INPUTS = [
    # version, 文件路径, 预测概率列, 结局列
    ("BE",   os.path.join(FROZEN_DIR, "eicu_predictions_be.csv"),   "pred_prob", "outcome"),
    ("HCO3", os.path.join(FROZEN_DIR, "eicu_predictions_hco3.csv"), "pred_prob", "outcome"),
]
OUT_ITER = {
    "BE":   os.path.join(RESULTS_DIR, "recal_bootstrap_eicu_BE.csv"),
    "HCO3": os.path.join(RESULTS_DIR, "recal_bootstrap_eicu_HCO3.csv"),
}
OUT_SUMMARY = os.path.join(RESULTS_DIR, "recal_bootstrap_eicu_summary.csv")

B        = 1000
SEED     = 20261020
EPS      = 1e-6
MIN_OOB  = 10

# ---------------- 工具函数 ----------------
def calib_slope_intercept(y, p, eps=EPS):
    """未中心化 logit(p)：logit(Y) = a*logit(p) + b。C=1e6 等价无惩罚 MLE。"""
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    if y.min() == y.max():  # 全 0 或全 1，无法拟合
        return np.nan, np.nan
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    m.fit(lp, y)
    a = float(m.coef_[0, 0])
    b = float(m.intercept_[0])
    return a, b

def brier(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))

def apply_recal(p, a, b, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    lp = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * lp + b)))

def med_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)), float(np.percentile(x, 25)), float(np.percentile(x, 75))

# ---------------- 主循环 ----------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary_rows = []

    for version, fpath, pcol, ycol in INPUTS:
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        d = pd.read_csv(fpath)
        n = len(d)
        y = d[ycol].astype(int).values
        p = d[pcol].astype(float).values

        # 1) 原始 Brier
        brier_orig = brier(y, p)

        # 2) 表观重校准（全队列拟合 a,b）
        a_app, b_app = calib_slope_intercept(y, p)
        p_app = apply_recal(p, a_app, b_app)
        brier_app = brier(y, p_app)

        print(f"\n[{version}] n={n}  events={int(y.sum())}")
        print(f"  Brier_original = {brier_orig:.6f}")
        print(f"  apparent: a={a_app:.6f}  b={b_app:.6f}  Brier_apparent={brier_app:.6f}")

        # 3) bootstrap OOB
        iters = []
        for it in range(B):
            idx = rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            if len(oob) < MIN_OOB:
                continue
            a_b, b_b = calib_slope_intercept(y[idx], p[idx])
            if np.isnan(a_b) or np.isnan(b_b):
                continue
            p_oob = apply_recal(p[oob], a_b, b_b)
            br = brier(y[oob], p_oob)
            a_o, b_o = calib_slope_intercept(y[oob], p_oob)
            iters.append(dict(iter=it, n_oob=len(oob),
                              a=a_b, b=b_b,
                              Brier_oob=br,
                              slope_oob=a_o,
                              intercept_oob=b_o))

        it_df = pd.DataFrame(iters)
        it_df.to_csv(OUT_ITER[version], index=False, float_format="%.6f")

        br_m, br_l, br_u = med_iqr(it_df["Brier_oob"])
        sl_m, sl_l, sl_u = med_iqr(it_df["slope_oob"])
        ic_m, ic_l, ic_u = med_iqr(it_df["intercept_oob"])

        print(f"  OOB (n_iter={len(it_df)}):")
        print(f"    Brier     median={br_m:.6f}  IQR=[{br_l:.6f}, {br_u:.6f}]")
        print(f"    slope     median={sl_m:.6f}  IQR=[{sl_l:.6f}, {sl_u:.6f}]")
        print(f"    intercept median={ic_m:.6f}  IQR=[{ic_l:.6f}, {ic_u:.6f}]")

        summary_rows.append(dict(
            Version=version, n=n, events=int(y.sum()),
            Brier_original=brier_orig,
            Brier_apparent=brier_app,
            slope_apparent=a_app, intercept_apparent=b_app,
            Brier_OOB_median=br_m,  Brier_OOB_q25=br_l,  Brier_OOB_q75=br_u,
            slope_OOB_median=sl_m,  slope_OOB_q25=sl_l,  slope_OOB_q75=sl_u,
            intercept_OOB_median=ic_m, intercept_OOB_q25=ic_l, intercept_OOB_q75=ic_u,
        ))

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False, float_format="%.6f")
    print("\nSaved:")
    for v in OUT_ITER:
        print("  ", OUT_ITER[v])
    print("  ", OUT_SUMMARY)

if __name__ == "__main__":
    main()