# -*- coding: utf-8 -*-
"""
Figure S2: HCO3 sensitivity analysis ROC + calibration (CHEST style)
- 修改版：直接读取 CSV 权威值，Legend 移出图外
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style，使用默认: {_e}")

# ============================================================
# 1. 读数据与预测
# ============================================================
tr = pd.read_csv(os.path.join(RESULTS_DIR, "traindata.csv"))
mimic = pd.read_csv(os.environ.get("MIMIC_WIDE_HCO3_CSV", r"D:\mimic_wide_hco3.csv")).dropna(
    subset=["alb_g_l", "bun_mmol_l", "be_mmol_l", "outcome"])

# 重命名（防坑）
mimic["ALB"]   = mimic["alb_g_l"]
mimic["BUN"]   = mimic["bun_mmol_l"]
mimic["HCO3_"] = mimic["be_mmol_l"]

X_mi = mimic[["ALB", "BUN", "HCO3_"]].values
y_mi = mimic["outcome"].values.astype(int)

# ★ 用 R 侧权威系数（不重拟合）
B0, B_ALB, B_BUN, B_HCO3 = 3.1936, -0.1209, 0.0374, -0.0604
lp_mi = B0 + B_ALB * X_mi[:, 0] + B_BUN * X_mi[:, 1] + B_HCO3 * X_mi[:, 2]
p_mi = 1.0 / (1.0 + np.exp(-lp_mi))

# ============================================================
# 2. 读 CSV 权威值（完全对齐 Table 3）
# ============================================================
_hco3 = pd.read_csv(os.path.join(RESULTS_DIR, "external_hco3_sensitivity_v2.csv"))
_h = dict(zip(_hco3["metric"], _hco3["value"]))
AUC_MAIN       = float(_h["AUC"])
AUC_LO         = float(_h["AUC_lo"])
AUC_HI         = float(_h["AUC_hi"])
SLOPE_MAIN     = float(_h["cal_slope"])
INTERCEPT_MAIN = float(_h["cal_intercept"])

print(f"权威值: AUC={AUC_MAIN:.3f} ({AUC_LO:.3f}-{AUC_HI:.3f}), slope={SLOPE_MAIN:.3f}, intercept={INTERCEPT_MAIN:.3f}")

# ============================================================
# 3. 画图
# ============================================================
COLOR_HCO3 = "#B2182B" 
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

# ---------- A: ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y_mi, p_mi)
ax.plot(fpr, tpr, color=COLOR_HCO3, lw=1.25,
        label=f"Logistic (HCO$_3^-$)\nAUC {AUC_MAIN:.3f} (95% CI {AUC_LO:.3f}-{AUC_HI:.3f})")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Reference")

ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
# ★ legend 图外下方
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=8.5)
ax.text(-0.16, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

# ---------- B: Calibration ----------
ax = axes[1]
frac, mean_p = calibration_curve(y_mi, p_mi, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color=COLOR_HCO3, lw=1.25, markersize=4.5, label="Logistic (HCO$_3^-$)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")

ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)
# ★ legend 图外下方
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=8.5,
          title=f"Slope {SLOPE_MAIN:.3f}; Intercept {INTERCEPT_MAIN:.3f}", title_fontsize=8.5)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

# ★ 给图外 legend 留空间
plt.subplots_adjust(bottom=0.28, wspace=0.35)
out_pdf = os.path.join(FIG_DIR_CHEST, "FigureS2_hco3_ROC_calibration.pdf")

# 注意：不要用 bbox_inches="tight"，否则会把图外的 legend 裁掉
plt.savefig(out_pdf, dpi=1200)
plt.close()

print(f"✅ 已保存: {out_pdf}")