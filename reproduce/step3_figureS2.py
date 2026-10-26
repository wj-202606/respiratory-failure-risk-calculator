# -*- coding: utf-8 -*-
"""step3_figureS2.py  MIMIC HCO3 敏感性分析 Figure S2
数据：external_predictions_hco3_v2.csv（列 outcome, p, p_rec）
期望：AUC 0.642(0.588-0.697) Brier 0.137(0.119-0.156) slope 0.486(0.294-0.709) intercept -0.809
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"; F=os.path.join(PROJ,"results","FROZEN_v1")
OUT=r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv(os.path.join(F,"external_predictions_hco3_v2.csv")); y=df.outcome.astype(int).values; p=df.p.astype(float).values
def cal(y,p):
    e=1e-6; pc=np.clip(p,e,1-e); lp=np.log(pc/(1-pc)); c=LogisticRegression(C=1e6,max_iter=10000).fit(lp.reshape(-1,1),y); return c.intercept_[0],c.coef_[0][0]
auc=roc_auc_score(y,p); bri=brier_score_loss(y,p); a,b=cal(y,p)
print(f"n={len(y)} events={y.sum()} mortality={y.mean():.3f}")
print(f"[AUC] {auc:.4f} 期望0.642 | [Brier] {bri:.4f} 期望0.137 | slope {b:.4f} 期望0.486 | int {a:.4f} 期望-0.809")
fig,ax=plt.subplots(1,2,figsize=(11,5))
fpr,tpr,_=roc_curve(y,p); ax[0].plot(fpr,tpr,lw=2,label=f"AUC={auc:.3f}"); ax[0].plot([0,1],[0,1],"--",c="grey"); ax[0].set_xlabel("1-Specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title("MIMIC-IV HCO3 (sensitivity)\nROC"); ax[0].legend()
qs=pd.qcut(p,10,labels=False,duplicates="drop"); obs=[y[qs==k].mean() for k in range(10)]; prd=[p[qs==k].mean() for k in range(10)]
ax[1].plot(prd,obs,"o-",lw=2); ax[1].plot([0,1],[0,1],"--",c="grey"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Observed"); ax[1].set_title(f"Calibration\nslope={b:.3f} int={a:.3f}")
plt.tight_layout(); pth=os.path.join(OUT,"FigureS2_reproduced.png"); plt.savefig(pth,dpi=300); plt.savefig(pth.replace(".png",".pdf")); print("图已保存：",pth)