# -*- coding: utf-8 -*-
"""
figS2_hco3.py v2.0 (v6.2.9)
Figure S2: HCO3- 外部验证 ROC + calibration。
数据来源：results/external_predictions_hco3_v2.csv（冻结 684 例）。
CI 来源：results/external_auc_stats.csv（与 Table 3 / Figure 2 同一份）。
本脚本不自算 bootstrap，不重拟合，不重算预测概率。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})
def file_sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from shared_calib import to_logit, calib_slope_intercept

BASE          = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(BASE, "results")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

PRED  = os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv")
STATS = os.path.join(RESULTS_DIR, "external_auc_stats.csv")

sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style，使用默认: {_e}")

if not os.path.exists(PRED):
    raise FileNotFoundError(f"缺少 HCO3 冻结预测文件: {PRED}")
if not os.path.exists(STATS):
    raise FileNotFoundError(
        f"缺少 {STATS}；请先运行 make_external_stats.py（v6.2.9）")

df    = pd.read_csv(PRED)
stats = pd.read_csv(STATS).set_index("Version").loc["HCO3"]

y = df["outcome"].astype(int).values
p = df["pred_prob"].astype(float).values
auc = roc_auc_score(y, p)
if abs(auc - float(stats["AUC"])) > 1e-6:
    raise RuntimeError(f"FigureS2 AUC {auc} != stats AUC {stats['AUC']}")

expected_sha = str(stats["pred_sha256"])
actual_sha   = file_sha256(PRED)
if expected_sha != actual_sha:
    raise RuntimeError(
        f"FigureS2 prediction SHA256 {actual_sha} != "
        f"stats pred_sha256 {expected_sha}")

lp = to_logit(p)
slope, intercept = calib_slope_intercept(y, lp)
auc_lo, auc_hi = float(stats["AUC_lo"]), float(stats["AUC_hi"])

COLOR_HCO3 = "#B2182B"
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

# ---------- A: ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y, p)
ax.plot(fpr, tpr, color=COLOR_HCO3, lw=1.25,
        label=f"Logistic (HCO$_3^-$)\nAUC {auc:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Reference")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=1, frameon=False, fontsize=8.5)
ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

# ---------- B: Calibration ----------
ax = axes[1]
frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color=COLOR_HCO3, lw=1.25, markersize=4.5,
        label="Logistic (HCO$_3^-$)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=1, frameon=False, fontsize=8.5,
          title=f"Slope {slope:.3f}; Intercept {intercept:.3f}",
          title_fontsize=8.5)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.subplots_adjust(bottom=0.28, wspace=0.35)
out_pdf = os.path.join(FIG_DIR_CHEST, "FigureS2_hco3_ROC_calibration.pdf")
plt.savefig(out_pdf, dpi=1200)   # 不用 bbox_inches="tight"，避免裁掉图外 legend
plt.close()

print(f"[Figure S2] n={len(y)} events={int(y.sum())} "
      f"AUC={auc:.3f} CI=({auc_lo:.3f},{auc_hi:.3f}) "
      f"slope={slope:.3f} intercept={intercept:.3f}")
print(f"[Figure S2] saved: {out_pdf}")