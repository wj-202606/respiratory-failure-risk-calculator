# -*- coding: utf-8 -*-
"""
step1_figure2.py
复现 Figure 2：MIMIC-IV 外部验证 ROC + 校准（BE 版）
期望：AUC 0.622 (0.553-0.682), Brier 0.166 (0.141-0.194),
      calibration slope 0.420 (0.151-0.661), intercept -0.617 (-1.149 to -0.142)
数据：results/FROZEN_v1/external_predictions.csv (n=434, 88 events)
"""
import os, csv, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ   = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
FROZEN = os.path.join(PROJ, "results", "FROZEN_v1")
REPO   = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator"
OUT    = os.path.join(REPO, "reproduce", "output")
os.makedirs(OUT, exist_ok=True)

# ---------- 读数据 ----------
df = pd.read_csv(os.path.join(FROZEN, "external_predictions.csv"))
y  = df["outcome"].astype(int).values
p  = df["pred_prob"].astype(float).values
print(f"n={len(y)}, events={int(y.sum())}, mortality={y.mean():.3f}")

# ---------- AUC / Brier ----------
auc   = roc_auc_score(y, p)
brier = brier_score_loss(y, p)
print(f"\n[AUC]   复现 {auc:.4f}   期望 0.622")
print(f"[Brier] 复现 {brier:.4f}   期望 0.166")

# ---------- 校准斜率/截距（未中心化，C=1e6） ----------
def calib_slope_intercept(y, p):
    eps = 1e-6
    pc  = np.clip(p, eps, 1-eps)
    lp  = np.log(pc / (1 - pc))
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)
    clf.fit(lp.reshape(-1, 1), y)
    return clf.intercept_[0], clf.coef_[0][0]   # a, b
a, b = calib_slope_intercept(y, p)
print(f"[Calib] 复现 slope={b:.4f} (期望 0.420)  intercept={a:.4f} (期望 -0.617)")

# ---------- Bootstrap 95% CI ----------
rng, B = np.random.default_rng(123), 2000
aucs, briers, slopes, ints = [], [], [], []
for _ in range(B):
    idx = rng.integers(0, len(y), len(y))
    yy, pp = y[idx], p[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        continue
    aucs.append(roc_auc_score(yy, pp))
    briers.append(brier_score_loss(yy, pp))
    aa, bb = calib_slope_intercept(yy, pp)
    slopes.append(bb); ints.append(aa)
def ci(v): return np.percentile(v, 2.5), np.percentile(v, 97.5)

print(f"\nBootstrap 95% CI (B={B}, 有效 {len(aucs)}):")
print(f"  AUC    {auc:.4f} ({ci(aucs)[0]:.4f}-{ci(aucs)[1]:.4f})   期望 0.622 (0.553-0.682)")
print(f"  Brier  {brier:.4f} ({ci(briers)[0]:.4f}-{ci(briers)[1]:.4f})   期望 0.166 (0.141-0.194)")
print(f"  slope  {b:.4f} ({ci(slopes)[0]:.4f}-{ci(slopes)[1]:.4f})   期望 0.420 (0.151-0.661)")
print(f"  int    {a:.4f} ({ci(ints)[0]:.4f}-{ci(ints)[1]:.4f})   期望 -0.617 (-1.149--0.142)")

# ---------- 图：ROC + 校准 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fpr, tpr, _ = roc_curve(y, p)
axes[0].plot(fpr, tpr, lw=2,
             label=f"AUC = {auc:.3f} (95% CI {ci(aucs)[0]:.3f}-{ci(aucs)[1]:.3f})")
axes[0].plot([0, 1], [0, 1], "--", c="grey", lw=1)
axes[0].set_xlabel("1 - Specificity"); axes[0].set_ylabel("Sensitivity")
axes[0].set_title("MIMIC-IV external validation (BE)\nROC"); axes[0].legend(loc="lower right")

qs = pd.qcut(p, 10, labels=False, duplicates="drop")
obs = [y[qs == k].mean() for k in sorted(np.unique(qs))]
prd = [p[qs == k].mean() for k in sorted(np.unique(qs))]
axes[1].plot(prd, obs, "o-", lw=2, label="Observed vs predicted")
axes[1].plot([0, 1], [0, 1], "--", c="grey", lw=1)
axes[1].set_xlabel("Predicted probability"); axes[1].set_ylabel("Observed mortality")
axes[1].set_title(f"Calibration (10 deciles)\nslope={b:.3f}, intercept={a:.3f}")
axes[1].legend(loc="upper left")

plt.tight_layout()
out_png = os.path.join(OUT, "Figure2_reproduced.png")
out_pdf = os.path.join(OUT, "Figure2_reproduced.pdf")
plt.savefig(out_png, dpi=300); plt.savefig(out_pdf)
print(f"\n图已保存：{out_png}\n图已保存：{out_pdf}")

# ---------- 对照表 ----------
print("\n" + "="*60)
print("Figure 2 复现对照")
print("="*60)
for name, got, exp in [("AUC", auc, "0.622"), ("Brier", brier, "0.166"),
                       ("slope", b, "0.420"), ("intercept", a, "-0.617")]:
    print(f"{name:<10} 复现 {got:.4f} | 期望 {exp}")