# -*- coding: utf-8 -*-
"""
delong_external.py (v2)
计算外部队列上 Logistic vs 各 ML 模型的 DeLong test
与 run_model.py 对齐：XGBoost / LightGBM 使用 early stopping

输入：results/traindata.csv, D:/mimic_wide.csv
输出：results/delong_external.csv
"""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
TRAIN_CSV = os.path.join(RESULTS_DIR, "traindata.csv")
VAL_CSV   = os.path.join(RESULTS_DIR, "valdata.csv")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide.csv")

RNG = 123
FEATURES = ["ALB", "BUN", "BE"]

print("=" * 70)
print("delong_external.py (v2, with early stopping)")
print("BASE:", BASE)
print("=" * 70)

# ============================================================
# 读数据
# ============================================================
train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
mimic = pd.read_csv(MIMIC_CSV).dropna(
    subset=["alb_g_l", "bun_mmol_l", "be_mmol_l", "outcome"])

X_tr = train[FEATURES].values
y_tr = train["Outcome"].values
X_va = val[FEATURES].values
y_va = val["Outcome"].values
X_mi = mimic[["alb_g_l", "bun_mmol_l", "be_mmol_l"]].values
y_mi = mimic["outcome"].values.astype(int)

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_va_s = sc.transform(X_va)
X_mi_s = sc.transform(X_mi)

print(f"train: n={len(y_tr)}, events={int(y_tr.sum())}")
print(f"val  : n={len(y_va)}, events={int(y_va.sum())}")
print(f"mimic: n={len(y_mi)}, events={int(y_mi.sum())}")

# ============================================================
# 模型（与 run_model.py 完全一致）
# ============================================================
models = {
    "Logistic": LogisticRegression(max_iter=2000, C=1.0, random_state=RNG),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        random_state=RNG, n_jobs=-1),
    "SVM": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale",
               random_state=RNG),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=1e-3,
        max_iter=2000, early_stopping=True, validation_fraction=0.15,
        random_state=RNG),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        eval_metric="logloss", random_state=RNG, n_jobs=-1)
if HAS_LGB:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        num_leaves=8, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RNG, verbose=-1)

SCALED = {"Logistic", "SVM", "MLP"}

# ============================================================
# 训练 + 外部预测
# ============================================================
p_mi_dict = {}
for name, model in models.items():
    Xtr = X_tr_s if name in SCALED else X_tr
    Xva = X_va_s if name in SCALED else X_va
    Xmi = X_mi_s if name in SCALED else X_mi

    if name == "XGBoost":
        model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)], verbose=False)
    elif name == "LightGBM":
        try:
            model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        except TypeError:
            model.fit(Xtr, y_tr, eval_set=[(Xva, y_va)],
                      early_stopping_rounds=50, verbose=False)
    else:
        model.fit(Xtr, y_tr)

    p_mi = model.predict_proba(Xmi)[:, 1]
    p_mi_dict[name] = p_mi
    auc_ext = roc_auc_score(y_mi, p_mi)
    print(f"  {name:15s} external AUC = {auc_ext:.4f}")

# ============================================================
# DeLong test（Hanley-McNeil 近似）
# ============================================================
def delong_z_p(y_true, p1, p2):
    auc1 = roc_auc_score(y_true, p1)
    auc2 = roc_auc_score(y_true, p2)
    n1 = int((y_true == 1).sum())
    n2 = int((y_true == 0).sum())

    def var_auc(auc, n1, n2):
        return (auc * (1 - auc) +
                (n1 - 1) * (auc / (2 - auc) - auc ** 2) +
                (n2 - 1) * (2 * auc ** 2 / (1 + auc) - auc ** 2)) / (n1 * n2)

    se = np.sqrt(var_auc(auc1, n1, n2) + var_auc(auc2, n1, n2))
    z = (auc1 - auc2) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p

names = list(p_mi_dict.keys())
rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        m1, m2 = names[i], names[j]
        a1, a2, z, p = delong_z_p(y_mi, p_mi_dict[m1], p_mi_dict[m2])
        rows.append({
            "Model 1": m1, "AUC 1": round(a1, 4),
            "Model 2": m2, "AUC 2": round(a2, 4),
            "ΔAUC": round(a1 - a2, 4),
            "z": round(z, 3),
            "p": round(p, 4) if p >= 0.001 else "<0.001",
        })

df = pd.DataFrame(rows)
print("\n" + "=" * 70)
print("外部验证 DeLong 两两对比")
print("=" * 70)
print(df.to_string(index=False))

df.to_csv(os.path.join(RESULTS_DIR, "delong_external.csv"), index=False)
print(f"\n✅ 已保存: results/delong_external.csv")