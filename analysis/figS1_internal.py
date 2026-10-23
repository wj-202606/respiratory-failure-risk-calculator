# -*- coding: utf-8 -*-
"""
Figure S1: Internal validation ROC + calibration curves (CHEST style)
- n = 608, events = 72
- 3-variable logistic model (ALB, BUN, BE)
- 输出: 论文图/FigureS1_internal_ROC_calibration.pdf
- v3: 彻底统一统计口径：A 面板为预设单次分析，B 面板图例明确标注为 50 套均值的稳定性分析
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

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
# 1. 读数据 + 用冻结系数算预测
# ============================================================
va = pd.read_csv(os.path.join(RESULTS_DIR, "valdata.csv"))
F = ["ALB", "BUN", "BE"]
X_va = va[F].values
y_va = va["Outcome"].values.astype(int)

# 用 R 侧 50 套 Rubin's rules 合并系数
B0, B_ALB, B_BUN, B_BE = 1.7825, -0.1227, 0.0343, -0.0585
lp_va = B0 + B_ALB * X_va[:, 0] + B_BUN * X_va[:, 1] + B_BE * X_va[:, 2]
p_va = 1.0 / (1.0 + np.exp(-lp_va))

# ============================================================
# 2. 动态计算真实统计值（确保与图完全一致）
# ============================================================
# A 面板：AUC
AUC_MAIN = roc_auc_score(y_va, p_va)
AUC_LO = 0.653 # 用预设值
AUC_HI = 0.776

# B 面板：动态计算当前的单次表观斜率/截距
def get_slope_intercept(y_true, p_pred):
    logit_p = np.log(np.clip(p_pred, 1e-6, 1-1e-6) / (1 - np.clip(p_pred, 1e-6, 1-1e-6)))
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(logit_p.reshape(-1, 1), y_true)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

slope_apparent, intercept_apparent = get_slope_intercept(y_va, p_va)

# 权威均值（来自 R 侧 50 套插补均值）
SLOPE_MEAN = 0.781
INTERCEPT_MEAN = -0.380

print(f"内部验证: AUC={AUC_MAIN:.3f}, 表观 slope={slope_apparent:.3f}, 表观 intercept={intercept_apparent:.3f}")

# ============================================================
# 3. 画图
# ============================================================
COLOR_BE = "#1F77B4"

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

# ---------- A: ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y_va, p_va)
ax.plot(fpr, tpr, color=COLOR_BE, lw=1.25,
        label=f"Logistic (BE)\nAUC {AUC_MAIN:.3f} (95% CI {AUC_LO:.3f}-{AUC_HI:.3f})")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Reference")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=1, frameon=False, fontsize=8.5)
ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

# ---------- B: Calibration ----------
ax = axes[1]
frac, mean_p = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color=COLOR_BE, lw=1.25, markersize=4.5,
        label="Logistic (BE)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)

# ★ 关键改动：在图例标题中明确写出这是 50 套插补的均值
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=1, frameon=False, fontsize=8.5,
          title=f"Mean calibration slope/intercept across 50 imputations:\n{SLOPE_MEAN:.3f} / {INTERCEPT_MEAN:.3f}",
          title_fontsize=8)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.subplots_adjust(bottom=0.28, wspace=0.35)
out_pdf = os.path.join(FIG_DIR_CHEST, "FigureS1_internal_ROC_calibration.pdf")
plt.savefig(out_pdf, dpi=1200)
plt.close()

print(f"✅ 已保存: {out_pdf}")