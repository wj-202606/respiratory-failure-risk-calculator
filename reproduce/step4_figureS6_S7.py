# -*- coding: utf-8 -*-
"""step4_figureS6_S7.py  eICU 外部验证 Figure S6/S7 (方案 B：ICU 24h 口径)
数据：eicu_predictions_be.csv / eicu_predictions_hco3.csv（outcome, pred_prob, pred_prob_recal）
期望：BE AUC 0.670(0.638-0.703) Brier 0.179 slope 0.599 int -0.486
      HCO3 AUC 0.670(0.643-0.696) Brier 0.171 slope 0.619 int -0.591
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PROJ = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
F1 = os.path.join(PROJ, "results", "FROZEN_v1")  # MIMIC 旧数据
F2 = os.path.join(PROJ, "results")               # eICU 最新数据
OUT = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT, exist_ok=True)

def cal(y, p):
    e = 1e-6; pc = np.clip(p, e, 1-e); lp = np.log(pc/(1-pc))
    c = LogisticRegression(C=1e6, max_iter=10000).fit(lp.reshape(-1,1), y)
    return c.intercept_[0], c.coef_[0][0]

def run(tag, fn, exp_auc, exp_b, exp_s, exp_i):
    df = pd.read_csv(os.path.join(F2, fn))
    y = df.outcome.astype(int).values; p = df.pred_prob.astype(float).values
    auc = roc_auc_score(y, p); bri = brier_score_loss(y, p); a, b = cal(y, p)
    print(f"\n[{tag}] n={len(y)} events={y.sum()} mortality={y.mean():.3f}")
    print(f"  AUC {auc:.4f} 期望{exp_auc} | Brier {bri:.4f} 期望{exp_b} | slope {b:.4f} 期望{exp_s} | int {a:.4f} 期望{exp_i}")
    
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set_xlabel("1-Specificity"); ax[0].set_ylabel("Sensitivity")
    ax[0].set_title(f"eICU {tag}\nROC"); ax[0].legend()
    
    qs = pd.qcut(p, 10, labels=False, duplicates="drop")
    obs = [y[qs==k].mean() for k in range(10)]; prd = [p[qs==k].mean() for k in range(10)]
    ax[1].plot(prd, obs, "o-", lw=2); ax[1].plot([0, 1], [0, 1], "--", c="grey")
    ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Observed")
    ax[1].set_title(f"Calibration\nslope={b:.3f} int={a:.3f}")
    
    plt.tight_layout()
    pth = os.path.join(OUT, f"FigureS{6 if tag=='BE' else 7}_reproduced.png")
    plt.savefig(pth, dpi=300); plt.savefig(pth.replace(".png", ".pdf")); plt.close()
    print("  图已保存：", pth)

run("BE", "eicu_predictions_be.csv", "0.670", "0.179", "0.599", "-0.486")
run("HCO3", "eicu_predictions_hco3.csv", "0.670", "0.171", "0.619", "-0.591")