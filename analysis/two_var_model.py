# -*- coding: utf-8 -*-
"""
④ 两变量模型敏感性分析
- 用三变量冻结系数，ALB 固定为开发队列均值（34.44 g/L）
- 在 MIMIC-IV 全队列中 BUN+BE 完整的患者上算 AUC/Brier/校准
- 与三变量模型（434 例，ALB+BUN+BE 完整）对比
输出:
  results/two_var_summary.csv
  results/two_var_predictions.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MIMIC_CSV = r"D:\mimic_wide_with_demo.csv"
print(f"读 {MIMIC_CSV} ...")
d = pd.read_csv(MIMIC_CSV)
print(f"总行数: {len(d)}")

# 列名
col_alb = "alb_g_l"
col_bun = "bun_mmol_l"
col_be  = "be_mmol_l"
col_out = "outcome"

# 三变量冻结系数
B0, B_ALB, B_BUN, B_BE = 1.7825, -0.1227, 0.0343, -0.0585
ALB_MEAN = 34.44  # 开发队列 ALB 均值

def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def calib_slope_intercept(y, logit_pred):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(logit_pred.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

# ---------- 两变量模型：ALB 固定为均值，BUN+BE 完整 ----------
print("\n=== 两变量模型（ALB 固定为开发队列均值 34.44 g/L）===")
d2 = d.dropna(subset=[col_bun, col_be, col_out]).copy()
print(f"BUN+BE+outcome 完整: n={len(d2)}, 事件={int(d2[col_out].sum())}")

y2 = d2[col_out].values.astype(int)
# LP 用 BUN + BE，ALB 固定为均值
lp2 = (B0
       + B_ALB * ALB_MEAN
       + B_BUN * d2[col_bun].values
       + B_BE  * d2[col_be].values)
p2 = 1.0 / (1.0 + np.exp(-lp2))

auc2 = roc_auc_score(y2, p2)
brier2 = brier_score_loss(y2, p2)
sl2, ic2 = calib_slope_intercept(y2, lp2)
print(f"AUC = {auc2:.4f}")
print(f"Brier = {brier2:.4f}")
print(f"Calibration slope = {sl2:.4f}, intercept = {ic2:.4f}")

# ---------- 三变量模型：ALB+BUN+BE 完整（434 例）----------
print("\n=== 三变量模型（ALB+BUN+BE 完整，434 例）===")
d3 = d.dropna(subset=[col_alb, col_bun, col_be, col_out]).copy()
print(f"三变量完整: n={len(d3)}, 事件={int(d3[col_out].sum())}")

y3 = d3[col_out].values.astype(int)
lp3 = (B0
       + B_ALB * d3[col_alb].values
       + B_BUN * d3[col_bun].values
       + B_BE  * d3[col_be].values)
p3 = 1.0 / (1.0 + np.exp(-lp3))

auc3 = roc_auc_score(y3, p3)
brier3 = brier_score_loss(y3, p3)
sl3, ic3 = calib_slope_intercept(y3, lp3)
print(f"AUC = {auc3:.4f}")
print(f"Brier = {brier3:.4f}")
print(f"Calibration slope = {sl3:.4f}, intercept = {ic3:.4f}")

# ---------- 保存 ----------
summary = pd.DataFrame([
    {"Model": "Two-variable (ALB fixed at 34.44 g/L, BUN + BE)",
     "n": len(y2), "events": int(y2.sum()),
     "AUC": auc2, "Brier": brier2, "slope": sl2, "intercept": ic2},
    {"Model": "Three-variable (ALB + BUN + BE)",
     "n": len(y3), "events": int(y3.sum()),
     "AUC": auc3, "Brier": brier3, "slope": sl3, "intercept": ic3},
])
summary.to_csv(os.path.join(RESULTS_DIR, "two_var_summary.csv"), index=False, float_format="%.6f")
print(f"\n✅ 已保存: results/two_var_summary.csv")

# 逐例预测
pred = pd.DataFrame({
    "hadm_id": d2["hadm_id"].values,
    "outcome": y2,
    "pred_two_var": p2,
    "pred_three_var_if_ALB_available": np.nan,
})
pred.to_csv(os.path.join(RESULTS_DIR, "two_var_predictions.csv"), index=False, float_format="%.6f")
print(f"✅ 已保存: results/two_var_predictions.csv")