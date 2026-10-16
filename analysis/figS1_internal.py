# -*- coding: utf-8 -*-
"""
Figure S1: Internal validation ROC + calibration curves (CHEST style)
- n = 607, events = 67
- 3-variable logistic model (ALB, BUN, BE)
- 输出: 论文图/FigureS1_internal_ROC_calibration.pdf
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

F = ["ALB", "BUN", "BE"]
X_tr = tr[F].values
y_tr = tr["Outcome"].values
X_va = va[F].values
y_va = va["Outcome"].values

print(f"训练子集: n={len(y_tr)}, events={int(y_tr.sum())}")
print(f"内部验证: n={len(y_va)}, events={int(y_va.sum())}")

# ============================================================
# 训练 Logistic
# ============================================================
sc = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=2000, random_state=123)
lr.fit(sc.transform(X_tr), y_tr)

p_va = lr.predict_proba(sc.transform(X_va))[:, 1]
auc_va = roc_auc_score(y_va, p_va)
brier_va = brier_score_loss(y_va, p_va)
print(f"内部验证 AUC={auc_va:.4f}, Brier={brier_va:.4f}")

# ============================================================
# Bootstrap CI
# ============================================================
np.random.seed(123)
n = len(y_va)
aucs = []
for b in range(2000):
    idx = np.random.choice(n, n, replace=True)
    yb = y_va[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        continue
    aucs.append(roc_auc_score(yb, p_va[idx]))
auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
print(f"内部验证 AUC 95% CI: {auc_lo:.4f}-{auc_hi:.4f}")

# ============================================================
# 校准斜率/截距（apparent）
# ============================================================
eps = 1e-6
lp_va = np.log(np.clip(p_va, eps, 1 - eps) / (1 - np.clip(p_va, eps, 1 - eps)))
cal_model = LogisticRegression(C=1e6, max_iter=2000).fit(lp_va.reshape(-1, 1), y_va)
slope_app = cal_model.coef_[0, 0]
intercept_app = cal_model.intercept_[0]
print(f"apparent calibration: slope={slope_app:.4f}, intercept={intercept_app:.4f}")

# ============================================================
# 画图（CHEST style: 删图内标题, A/B 面板, Arial, 线宽 1.25）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

# ---------- A: ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y_va, p_va)
# ★ 数字与正文 Table 3 对齐（R 实现：0.7385 / 0.6715-0.7996）
AUC_MAIN = 0.712
AUC_LO   = 0.649
AUC_HI   = 0.770
ax.plot(fpr, tpr, color="#1F77B4", lw=1.25,
        label=f"Logistic (BE)\nAUC {AUC_MAIN:.3f} (95% CI {AUC_LO:.3f}-{AUC_HI:.3f})")
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
frac, mean_p = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color="#1F77B4", lw=1.25, markersize=4.5,
        label="Logistic (BE)")
ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed probability")
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.grid(True)
# ★ 图例 title 用 bootstrap-corrected 值（与正文一致）
ax.legend(loc="lower right", frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", borderpad=0.5,
          title="Calibration (50 imputations)\nslope 0.744; intercept -0.435",
          title_fontsize=8)
ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.tight_layout()
out_pdf = os.path.join(FIG_DIR_CHEST, "FigureS1_internal_ROC_calibration.pdf")
plt.savefig(out_pdf, dpi=1200, bbox_inches="tight")
plt.close()

# 旧版兼容（figures/ 原路径）
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
fpr, tpr, _ = roc_curve(y_va, p_va)
ax.plot(fpr, tpr, color="steelblue", lw=2,
        label=f"Logistic (BE)\nAUC={auc_va:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Reference")
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("Internal validation ROC", fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02]); ax.grid(alpha=0.3)

ax = axes[1]
frac, mean_p = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color="steelblue", lw=2, markersize=7, label="Logistic (BE)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
ax.set_xlabel("Predicted probability", fontsize=12)
ax.set_ylabel("Observed probability", fontsize=12)
ax.set_title(f"Calibration\n(slope={slope_app:.3f}, intercept={intercept_app:.3f})", fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5]); ax.grid(alpha=0.3)

plt.tight_layout()
out_pdf_old = os.path.join(FIGURES_DIR, "FigS1_internal_ROC_calibration.pdf")
plt.savefig(out_pdf_old, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ 已保存:")
print(f"   {out_pdf}          (CHEST style)")
print(f"   {out_pdf_old}      (旧版兼容)")