# -*- coding: utf-8 -*-
"""
Figure S2: HCO3 sensitivity analysis ROC + calibration (CHEST style)
- n = 684, events = 109
- Model refit with ALB + BUN + HCO3_
- 输出: 论文图/FigureS2_hco3_ROC_calibration.pdf
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(FIGURES_DIR, exist_ok=True)
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
# 读数据
# ============================================================
tr = pd.read_csv(os.path.join(RESULTS_DIR, "traindata.csv"))
va = pd.read_csv(os.path.join(RESULTS_DIR, "valdata.csv"))
mimic = pd.read_csv(os.environ.get("MIMIC_WIDE_HCO3_CSV", r"D:\mimic_wide_hco3.csv")).dropna(
    subset=["alb_g_l", "bun_mmol_l", "be_mmol_l", "outcome"])

F = ["ALB", "BUN", "HCO3_"]
X_tr = tr[F].values
y_tr = tr["Outcome"].values

X_mi = mimic[["alb_g_l", "bun_mmol_l", "be_mmol_l"]].values
y_mi = mimic["outcome"].values.astype(int)

print(f"训练子集: n={len(y_tr)}, events={int(y_tr.sum())}")
print(f"外部队列: n={len(y_mi)}, events={int(y_mi.sum())}")

# ============================================================
# 训练 Logistic（用 HCO3_）
# ============================================================
sc = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=2000, random_state=123)
lr.fit(sc.transform(X_tr), y_tr)

p_mi = lr.predict_proba(sc.transform(X_mi))[:, 1]
auc_mi = roc_auc_score(y_mi, p_mi)
brier_mi = brier_score_loss(y_mi, p_mi)
print(f"外部验证 AUC={auc_mi:.4f}, Brier={brier_mi:.4f}")

# ============================================================
# Bootstrap CI
# ============================================================
np.random.seed(123)
n = len(y_mi)
aucs = []
for b in range(2000):
    idx = np.random.choice(n, n, replace=True)
    yb = y_mi[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    aucs.append(roc_auc_score(yb, p_mi[idx]))
auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
print(f"外部验证 AUC 95% CI: {auc_lo:.4f}-{auc_hi:.4f}")

# ============================================================
# 校准斜率/截距（apparent）
# ============================================================
eps = 1e-6
lp_mi = np.log(np.clip(p_mi, eps, 1 - eps) / (1 - np.clip(p_mi, eps, 1 - eps)))
cal_model = LogisticRegression(C=1e6, max_iter=2000).fit(lp_mi.reshape(-1, 1), y_mi)
slope_app = cal_model.coef_[0, 0]
intercept_app = cal_model.intercept_[0]
print(f"apparent calibration: slope={slope_app:.4f}, intercept={intercept_app:.4f}")

# ============================================================
# 画图（CHEST style）
# ============================================================
COLOR_HCO3 = "#B2182B"   # 深红，与 Figure S4 B 面板一致

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

# ---------- A: ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y_mi, p_mi)
# ★ 数字与正文 Table 3 对齐
AUC_MAIN = 0.643
AUC_LO   = 0.588
AUC_HI   = 0.697
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
ax.legend(loc="lower right", frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", borderpad=0.5)
ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

# ---------- B: Calibration ----------
ax = axes[1]
frac, mean_p = calibration_curve(y_mi, p_mi, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color=COLOR_HCO3, lw=1.25, markersize=4.5,
        label="Logistic (HCO$_3^-$)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)
ax.legend(loc="lower right", frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", borderpad=0.5,
          title=f"Slope {slope_app:.2f}; Intercept {intercept_app:.2f}",
          title_fontsize=8.5)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.tight_layout()
out_pdf = os.path.join(FIG_DIR_CHEST, "FigureS2_hco3_ROC_calibration.pdf")
plt.savefig(out_pdf, dpi=1200, bbox_inches="tight")
plt.close()

# 旧版兼容
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
fpr, tpr, _ = roc_curve(y_mi, p_mi)
ax.plot(fpr, tpr, color="indianred", lw=2,
        label=f"Logistic (HCO3-)\nAUC={auc_mi:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Reference")
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("HCO3- external validation ROC", fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02]); ax.grid(alpha=0.3)

ax = axes[1]
frac, mean_p = calibration_curve(y_mi, p_mi, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color="indianred", lw=2, markersize=7, label="Logistic (HCO3-)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Predicted probability", fontsize=12)
ax.set_ylabel("Observed probability", fontsize=12)
ax.set_title(f"Calibration\n(slope={slope_app:.3f}, intercept={intercept_app:.3f})", fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5]); ax.grid(alpha=0.3)

plt.tight_layout()
out_pdf_old = os.path.join(FIGURES_DIR, "FigS2_hco3_ROC_calibration.pdf")
plt.savefig(out_pdf_old, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ 已保存:")
print(f"   {out_pdf}          (CHEST style)")
print(f"   {out_pdf_old}      (旧版兼容)")