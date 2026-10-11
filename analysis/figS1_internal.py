# -*- coding: utf-8 -*-
"""
Figure S1: Internal validation ROC + calibration curves
- n = 607, events = 67
- 3-variable logistic model (ALB, BUN, BE)
"""
import os
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

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

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
print(f"内部验证 AUC 95% CI: {auc_lo:.4f}–{auc_hi:.4f}")

# ============================================================
# 画图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ---------- 左：ROC ----------
ax = axes[0]
fpr, tpr, _ = roc_curve(y_va, p_va)
ax.plot(fpr, tpr, color="steelblue", lw=2,
        label=f"Logistic (BE)\nAUC={auc_va:.3f} (95% CI {auc_lo:.3f}–{auc_hi:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Reference")
ax.set_xlabel("1 - Specificity", fontsize=12)
ax.set_ylabel("Sensitivity", fontsize=12)
ax.set_title("Internal validation ROC", fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(alpha=0.3)

# ---------- 右：校准曲线 ----------
ax = axes[1]
frac, mean_p = calibration_curve(y_va, p_va, n_bins=10, strategy="quantile")
ax.plot(mean_p, frac, "o-", color="steelblue", lw=2, markersize=7,
        label="Logistic (BE)")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

# 校准斜率/截距
eps = 1e-6
lp = np.log(np.clip(p_va, eps, 1 - eps) / (1 - np.clip(p_va, eps, 1 - eps)))
cm = LogisticRegression(C=1e6, max_iter=2000).fit(lp.reshape(-1, 1), y_va)
slope = cm.coef_[0, 0]
intercept = cm.intercept_[0]

ax.set_xlabel("Predicted probability", fontsize=12)
ax.set_ylabel("Observed probability", fontsize=12)
ax.set_title(f"Calibration\n(slope={slope:.3f}, intercept={intercept:.3f})",
             fontsize=13)
ax.legend(loc="lower right", fontsize=10, frameon=True)
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])
ax.grid(alpha=0.3)

plt.tight_layout()
out_pdf = os.path.join(FIGURES_DIR, "FigS1_internal_ROC_calibration.pdf")
out_png = os.path.join(FIGURES_DIR, "FigS1_internal_ROC_calibration.png")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ 已保存:")
print(f"   {out_pdf}")
print(f"   {out_png}")