# -*- coding: utf-8 -*-
"""
Table S5: DeLong test for pairwise AUC comparisons
6 models: Logistic, RF, LightGBM, XGBoost, SVM, MLP
- 内部验证 (n=607) 和外部验证 (n=434) 各做一遍
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

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide.csv")

RNG = 123
FEATURES = ["ALB", "BUN", "BE"]

# ============================================================
# 读数据
# ============================================================
train = pd.read_csv(os.path.join(RESULTS_DIR, "traindata.csv"))
val   = pd.read_csv(os.path.join(RESULTS_DIR, "valdata.csv"))

X_tr = train[FEATURES].values
y_tr = train["Outcome"].values
X_va = val[FEATURES].values
y_va = val["Outcome"].values

sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr)
X_va_s = sc.transform(X_va)

# 外部数据
mimic = pd.read_csv(MIMIC_CSV).dropna(
    subset=["alb_g_l", "bun_mmol_l", "be_mmol_l", "outcome"])
X_mi = mimic[["alb_g_l", "bun_mmol_l", "be_mmol_l"]].values
y_mi = mimic["outcome"].values.astype(int)
X_mi_s = sc.transform(X_mi)

# ============================================================
# 定义模型（和 run_model.py 一致）
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
# 训练 + 预测
# ============================================================
fitted = {}
p_val_dict = {}
p_mi_dict = {}

for name, model in models.items():
    Xtr = X_tr_s if name in SCALED else X_tr
    Xva = X_va_s if name in SCALED else X_va
    Xmi = X_mi_s if name in SCALED else X_mi

    try:
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

        fitted[name] = model
        p_val_dict[name] = model.predict_proba(Xva)[:, 1]
        p_mi_dict[name] = model.predict_proba(Xmi)[:, 1]
        print(f"  {name:15s} val AUC={roc_auc_score(y_va, p_val_dict[name]):.4f}  "
              f"ext AUC={roc_auc_score(y_mi, p_mi_dict[name]):.4f}")
    except Exception as e:
        print(f"  {name:15s} FAILED: {e}")

# ============================================================
# DeLong 检验（Hanley-McNeil 近似）
# ============================================================
def delong_z_p(y_true, p1, p2):
    """Hanley-McNeil 近似 DeLong test"""
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

# ============================================================
# 内部验证：所有两两对比
# ============================================================
model_names = list(p_val_dict.keys())
print("\n" + "=" * 70)
print("内部验证 DeLong（n=607）")
print("=" * 70)

rows_val = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        a1, a2, z, p = delong_z_p(y_va, p_val_dict[m1], p_val_dict[m2])
        rows_val.append({
            "Model 1": m1, "AUC 1": round(a1, 4),
            "Model 2": m2, "AUC 2": round(a2, 4),
            "ΔAUC": round(a1 - a2, 4),
            "z": round(z, 3),
            "p": round(p, 4) if p >= 0.001 else "<0.001",
        })

df_val = pd.DataFrame(rows_val)
print(df_val.to_string(index=False))

# ============================================================
# 外部验证：所有两两对比
# ============================================================
print("\n" + "=" * 70)
print("外部验证 DeLong（n=434）")
print("=" * 70)

rows_mi = []
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        a1, a2, z, p = delong_z_p(y_mi, p_mi_dict[m1], p_mi_dict[m2])
        rows_mi.append({
            "Model 1": m1, "AUC 1": round(a1, 4),
            "Model 2": m2, "AUC 2": round(a2, 4),
            "ΔAUC": round(a1 - a2, 4),
            "z": round(z, 3),
            "p": round(p, 4) if p >= 0.001 else "<0.001",
        })

df_mi = pd.DataFrame(rows_mi)
print(df_mi.to_string(index=False))

# ============================================================
# 保存
# ============================================================
df_val.to_csv(os.path.join(RESULTS_DIR, "TableS5_Delong_internal.csv"),
              index=False)
df_mi.to_csv(os.path.join(RESULTS_DIR, "TableS5_Delong_external.csv"),
             index=False)

print(f"\n✅ 已保存:")
print(f"   results/TableS5_Delong_internal.csv")
print(f"   results/TableS5_Delong_external.csv")

# ============================================================
# 汇总 AUC
# ============================================================
print("\n" + "=" * 70)
print("AUC 汇总")
print("=" * 70)
summary = []
for name in model_names:
    summary.append({
        "Model": name,
        "Internal AUC": round(roc_auc_score(y_va, p_val_dict[name]), 4),
        "External AUC": round(roc_auc_score(y_mi, p_mi_dict[name]), 4),
    })
print(pd.DataFrame(summary).sort_values("External AUC", ascending=False).to_string(index=False))