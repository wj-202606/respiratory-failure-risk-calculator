# -*- coding: utf-8 -*-
"""
step8_figureS3_SHAP.py  Figure S3 / S3b：XGBoost SHAP
数据：results/traindata.csv（n=1,420, 154 events, ALB/BUN/BE 零缺失）
口径：与 run_model.py 一致（1000 树, lr 0.03, max_depth 3, 早停）
验收：排序 ALB > BE > BUN；方向 ALB 低→正、BUN 高→正、BE 低→正
"""
import os, numpy as np, pandas as pd
import xgboost as xgb
import shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

PROJ = r"C:\Users\24339\Desktop\6.30呼吸衰竭\6.30呼衰\1.R数据预处理+变量筛选9.21"
OUT  = r"C:\Users\24339\Desktop\respiratory-failure-risk-calculator\reproduce\output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(PROJ, "results", "traindata.csv"))
X  = df[["ALB","BUN","BE"]].astype(float).values
y  = df["Outcome"].astype(int).values
feat = ["ALB","BUN","BE"]
print(f"n={len(y)}, events={y.sum()}, mortality={y.mean():.3f}")

# 早停用 8:2 内部划分（只用于确定 n_estimators，最终模型在全集上重训）
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat)
dva = xgb.DMatrix(Xva, label=yva, feature_names=feat)
params = dict(objective="binary:logistic", eval_metric="auc", eta=0.03,
              max_depth=3, lambda_=1.0, alpha=0.1, tree_method="hist", seed=123)
bst_cv = xgb.train(params, dtr, num_boost_round=1000, evals=[(dva,"va")],
                   early_stopping_rounds=50, verbose_eval=False)
best = bst_cv.best_iteration + 1
print("早停最佳轮数：", best)
dfull = xgb.DMatrix(X, label=y, feature_names=feat)
bst = xgb.train(params, dfull, num_boost_round=best, verbose_eval=False)
print("训练集 AUC（重训后）：%.4f" % roc_auc_score(y, bst.predict(dfull)))

# SHAP
expl = shap.TreeExplainer(bst)
sv = expl.shap_values(dfull)          # (n, 3) log-odds 空间
sv = np.asarray(sv)
mean_abs = np.abs(sv).mean(axis=0)
order = np.argsort(-mean_abs)
print("\n平均 |SHAP|：", {feat[i]: round(float(mean_abs[i]),4) for i in order})
print("排序：", [feat[i] for i in order], " 期望 ALB > BE > BUN")
assert [feat[i] for i in order] == ["ALB","BE","BUN"], "SHAP 排序与期望不符"

# 方向断言：低值→正 SHAP 等
def direction_ok(x, s, expect_low_pos=True):
    lo = x <= np.percentile(x, 25); hi = x >= np.percentile(x, 75)
    d = s[lo].mean() - s[hi].mean()
    return (d > 0) if expect_low_pos else (d < 0)
checks = {
 "ALB 低→正": direction_ok(X[:,0], sv[:,0], True),
 "BUN 高→正": direction_ok(X[:,1], sv[:,1], False),
 "BE  低→正": direction_ok(X[:,2], sv[:,2], True),
}
print("\n方向核对：")
for k,v in checks.items(): print(f"  {k}: {'OK' if v else 'FAIL'}")
assert all(checks.values()), "SHAP 方向与期望不符"

# 图 S3：beeswarm
plt.figure(figsize=(8,5))
shap.summary_plot(sv, X, feature_names=feat, show=False, plot_size=None)
plt.title("SHAP summary (XGBoost, training subset)")
p1 = os.path.join(OUT, "FigureS3_SHAP_summary_reproduced.png")
plt.tight_layout(); plt.savefig(p1, dpi=300); plt.savefig(p1.replace(".png",".pdf")); plt.close()
# 图 S3b：mean |SHAP| bar
plt.figure(figsize=(6,4))
idx = order
plt.barh([feat[i] for i in idx][::-1], [mean_abs[i] for i in idx][::-1], color="C0")
plt.xlabel("mean |SHAP|"); plt.title("Mean absolute SHAP (XGBoost)")
p2 = os.path.join(OUT, "FigureS3b_SHAP_bar_reproduced.png")
plt.tight_layout(); plt.savefig(p2, dpi=300); plt.savefig(p2.replace(".png",".pdf")); plt.close()
print("\n图已保存：", p1); print("图已保存：", p2)
print("\n[验收] 排序 ALB>BE>BUN ✅；方向三项 ✅；训练 AUC 期望≈0.681（run_model.py 口径）")