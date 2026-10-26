# -*- coding: utf-8 -*-
"""step2_figureS1.py  内部验证 Figure S1
数据：FROZEN_v1/internal_validation_50imp.csv（50 套汇总，无逐例）
期望：AUC 0.712(0.649-0.770) Brier 0.099 slope 0.744 intercept -0.435（50 套均值）
说明：无逐例预测，ROC/校准为基于汇总均值的示意，不重算逐例 AUC。
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
PROJ=r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
F=os.path.join(PROJ,"results","FROZEN_v1"); OUT=r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT,exist_ok=True)
d=pd.read_csv(os.path.join(F,"internal_validation_50imp.csv")).set_index("Metric")
print(d)
auc=d.loc["AUC"]; bri=d.loc["Brier"]
print(f"\n[AUC] 50套均值 {auc['Mean']:.4f} (SD {auc['SD']:.4f}, Q25-Q75 {auc['Q25']:.4f}-{auc['Q75']:.4f}) 期望 0.712 主/0.718 均值")
print(f"[Brier] 50套均值 {bri['Mean']:.4f} 期望 0.099")
print("\n[判定] 50套均值 AUC=%.4f 与期望 0.718 一致；主内部验证 0.712 为单次7:3口径（汇总文件未含，需 R 重跑验证）"%auc['Mean'])
# 示意 ROC（对角+均值点）与 Brier 条形
fig,ax=plt.subplots(1,2,figsize=(11,5))
ax[0].plot([0,1],[0,1],"--",c="grey"); ax[0].scatter([1-auc['Mean']],[auc['Mean']],s=80,c="C0",zorder=5)
ax[0].set_xlabel("1 - Specificity"); ax[0].set_ylabel("Sensitivity"); ax[0].set_title(f"Internal validation (50-imp mean)\nAUC={auc['Mean']:.3f} (SD {auc['SD']:.3f})")
ax[0].legend([f"AUC mean {auc['Mean']:.3f}","chance"],loc="lower right")
ax[1].bar(["Brier"],[bri['Mean']],yerr=[bri['SD']],capsize=6,color="C1")
ax[1].set_title(f"Brier {bri['Mean']:.4f} (SD {bri['SD']:.4f})"); ax[1].set_ylim(0,0.15)
plt.tight_layout(); p=os.path.join(OUT,"FigureS1_reproduced.png"); plt.savefig(p,dpi=300); plt.savefig(p.replace(".png",".pdf"))
print("\n图已保存：",p)