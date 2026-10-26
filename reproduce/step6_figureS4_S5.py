# -*- coding: utf-8 -*-
"""step6_figureS4_S5.py  Figure S4 重校准前后 + S5 AUC 100 次划分
数据：external_predictions_hco3_v2.csv(p,p_rec)、eicu_predictions_*.csv、auc_100splits.csv
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"; F=os.path.join(PROJ,"results","FROZEN_v1")
OUT=r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"; os.makedirs(OUT,exist_ok=True)
# S5
a=pd.read_csv(os.path.join(F,"auc_100splits.csv")); med=a.AUC.median(); q=a.AUC.quantile([.25,.75]).values
print(f"[S5] n={len(a)} 中位 AUC {med:.4f} IQR {q[0]:.4f}-{q[1]:.4f} 期望 0.766 (0.745-0.785)")
plt.figure(figsize=(7,5)); plt.hist(a.AUC,bins=20,color="C0",alpha=.8); plt.axvline(med,ls="--",c="k"); plt.xlabel("AUC"); plt.ylabel("Count"); plt.title(f"100 random 7:3 splits\nmedian={med:.3f}"); pth=os.path.join(OUT,"FigureS5_reproduced.png"); plt.savefig(pth,dpi=300); plt.savefig(pth.replace(".png",".pdf")); plt.close(); print(" 图已保存：",pth)
# S4 重校准曲线（MIMIC BE 用 external_predictions.csv 无 p_rec，用公式重构；HCO3 有 p_rec）
def curve(y,p,tag):
    qs=pd.qcut(p,10,labels=False,duplicates="drop"); return [p[qs==k].mean() for k in range(10)],[y[qs==k].mean() for k in range(10)]
df=pd.read_csv(os.path.join(F,"external_predictions.csv")); y=df.outcome.astype(int).values; p=df.pred_prob.astype(float).values
lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); p_rec=1/(1+np.exp(-(0.4196*lp-0.6172)))
fig,ax=plt.subplots(figsize=(7,6)); pr,ob=curve(y,p,"BE"); ax.plot(pr,ob,"o",mfc="none",label="BE original"); pr2,ob2=curve(y,p_rec,"BE"); ax.plot(pr2,ob2,"s",label="BE recal (formula 0.4196/-0.6172)")
df2=pd.read_csv(os.path.join(F,"external_predictions_hco3_v2.csv")); y2=df2.outcome.astype(int).values
pr3,ob3=curve(y2,df2.p.values,"H"); ax.plot(pr3,ob3,"^",mfc="none",label="HCO3 original"); pr4,ob4=curve(y2,df2.p_rec.values,"H"); ax.plot(pr4,ob4,"D",label="HCO3 recal (file p_rec)")
ax.plot([0,1],[0,1],"--",c="grey"); ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title("Recalibration before/after (MIMIC-IV)"); ax.legend(fontsize=8)
pth=os.path.join(OUT,"FigureS4_reproduced.png"); plt.savefig(pth,dpi=300); plt.savefig(pth.replace(".png",".pdf")); plt.close(); print(" 图已保存：",pth)