# -*- coding: utf-8 -*-
"""
fig2_be.py v1.0 (v6.2.10)
Figure 2: BE model ROC + calibration.
输入：
  results/external_predictions.csv    （冻结 BE 预测）
  results/external_auc_stats.csv      （BE 的 AUC / AUC_lo / AUC_hi / pred_sha256）
本脚本不重拟合、不重算 bootstrap；CI 从 stats 读。
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from shared_calib import to_logit, calib_slope_intercept

BASE          = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(BASE, "results")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

PRED  = os.path.join(RESULTS_DIR, "external_predictions.csv")
STATS = os.path.join(RESULTS_DIR, "external_auc_stats.csv")

def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] chest_style not found, using default: {_e}")

if not os.path.exists(PRED):
    raise FileNotFoundError(f"Missing frozen BE prediction file: {PRED}")
if not os.path.exists(STATS):
    raise FileNotFoundError(
        f"Missing {STATS}; run make_external_stats.py first (v6.2.10)")

df    = pd.read_csv(PRED)
stats = pd.read_csv(STATS).set_index("Version").loc["BE"]

y = df["outcome"].astype(int).values
p = df["pred_prob"].astype(float).values

auc = roc_auc_score(y, p)
if abs(auc - float(stats["AUC"])) > 1e-6:
    raise RuntimeError(f"Figure2 AUC {auc} != stats AUC {stats['AUC']}")

expected_sha = str(stats["pred_sha256"])
actual_sha   = file_sha256(PRED)
if expected_sha != actual_sha:
    raise RuntimeError(
        f"Figure2 prediction SHA256 {actual_sha} != "
        f"stats pred_sha256 {expected_sha}")

lp = to_logit(p)
slope, intercept = calib_slope_intercept(y, lp)
auc_lo, auc_hi = float(stats["AUC_lo"]), float(stats["AUC_hi"])

COLOR_BE = "#1F77B4"
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

ax = axes[0]
fpr, tpr, _ = roc_curve(y, p)
ax.plot(fpr, tpr, color=COLOR_BE, lw=1.25, label="Logistic (BE)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Reference")
ax.text(0.03, 0.97,
        f"AUC {auc:.3f}\n(95% CI {auc_lo:.3f}-{auc_hi:.3f})",
        transform=ax.transAxes, fontsize=8, va="top", ha="left")
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.tick_params(axis='both', labelsize=10)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, frameon=False, fontsize=8.5)
ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

ax = axes[1]
frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
ax.plot(mean_pred, frac_pos, "o-", color=COLOR_BE,
        lw=1.25, markersize=4.5, label="Logistic (BE)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
ax.text(0.03, 0.97,
        f"Slope {slope:.3f}; Intercept {intercept:.3f}",
        transform=ax.transAxes, fontsize=8, va="top", ha="left")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.tick_params(axis='both', labelsize=10)
ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, frameon=False, fontsize=8.5)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.subplots_adjust(bottom=0.28, wspace=0.35)
out_pdf = os.path.join(FIG_DIR_CHEST, "Figure2_external_ROC_calibration.pdf")
plt.savefig(out_pdf, dpi=1200)
plt.close()

print(f"[Figure 2] {PRED}  n={len(y)}  events={int(y.sum())}  "
      f"AUC={auc:.4f}  CI=({auc_lo:.3f}-{auc_hi:.3f})  "
      f"slope={slope:.3f}  intercept={intercept:.3f}")
print(f"[Figure 2] saved: {out_pdf}")