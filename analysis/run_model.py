# -*- coding: utf-8 -*-
"""
呼吸衰竭预后预测模型 — Python ML 对比 + SHAP 可解释性
v4.3：Figure 2 数据源改为 FROZEN_v1/external_predictions.csv，确保与 Table 3/5、正文 AUC 0.622 严格闭环
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from scipy import stats

# ============================================================
# 0. 路径设置
# ============================================================
BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

TRAIN_CSV = os.path.join(RESULTS_DIR, "traindata.csv")
VAL_CSV   = os.path.join(RESULTS_DIR, "valdata.csv")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide.csv")

RNG = 123
FEATURES = ["ALB", "BUN", "BE"]
EXT_AUC_CI = (0.553, 0.682)

print("=" * 70)
print("Python:", sys.executable)
print("BASE:", BASE)
print("=" * 70)

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style，使用默认: {_e}")

plt.rcParams['axes.edgecolor']  = 'black'
plt.rcParams['axes.linewidth']  = 1.0
plt.rcParams['xtick.color']     = 'black'
plt.rcParams['ytick.color']     = 'black'
plt.rcParams['text.color']      = 'black'
plt.rcParams['axes.labelcolor'] = 'black'

# ============================================================
# 1. 检查可选包
# ============================================================
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[警告] 未安装 xgboost")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[警告] 未安装 lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[警告] 未安装 shap")

# ============================================================
# 2. 读数据
# ============================================================
print("\n[1/6] 读入数据 ...")
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"找不到训练集: {TRAIN_CSV}")
if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(f"找不到验证集: {VAL_CSV}")

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)

X_train = train[FEATURES].values
y_train = train["Outcome"].values
X_val   = val[FEATURES].values
y_val   = val["Outcome"].values

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

# ============================================================
# 3. 定义模型
# ============================================================
print("\n[2/6] 训练模型 ...")

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

fitted = {}
results = []
for name, model in models.items():
    print(f"  -> {name} ...", end=" ")
    Xtr = X_train_s if name in SCALED else X_train
    Xva = X_val_s   if name in SCALED else X_val

    try:
        if name == "XGBoost":
            model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
        elif name == "LightGBM":
            try:
                model.fit(Xtr, y_train, eval_set=[(Xva, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            except TypeError:
                model.fit(Xtr, y_train, eval_set=[(Xva, y_val)],
                          early_stopping_rounds=50, verbose=False)
        else:
            model.fit(Xtr, y_train)

        p_val = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        brier = brier_score_loss(y_val, p_val)
        fitted[name] = model
        results.append({"Model": name,
                        "AUC_val": round(auc, 4),
                        "Brier_val": round(brier, 4)})
        print(f"AUC={auc:.4f}")
    except Exception as e:
        print(f"失败: {e}")

res_df = pd.DataFrame(results).sort_values("AUC_val", ascending=False)
res_df.to_csv(os.path.join(RESULTS_DIR, "python_model_comparison.csv"),
              index=False)

# ============================================================
# 4. 外部验证（仍保留，用于写出 external_predictions.csv 与 ML 外部对比）
# ============================================================
print("\n[3/6] 外部验证（MIMIC-IV）...")
ext_df = None
mimic_cc = None
p_ext_logistic = None
y_mimic = None

if not os.path.exists(MIMIC_CSV):
    print(f"  [跳过] 找不到 {MIMIC_CSV}")
else:
    mimic = pd.read_csv(MIMIC_CSV)
    col_alb = "alb_g_l"  if "alb_g_l"  in mimic.columns else "ALB"
    col_bun = "bun_mmol_l" if "bun_mmol_l" in mimic.columns else "BUN"
    col_be  = "be_mmol_l" if "be_mmol_l" in mimic.columns else "BE"
    col_out = "outcome" if "outcome" in mimic.columns else "Outcome"

    need = [col_alb, col_bun, col_be, col_out]
    if not all(c in mimic.columns for c in need):
        print(f"  [跳过] MIMIC 缺少列: {[c for c in need if c not in mimic.columns]}")
    else:
        mimic_cc = mimic.dropna(subset=need).copy()
        print(f"  完整案例: {len(mimic_cc)}, 事件: {int(mimic_cc[col_out].sum())}")

        X_mimic = mimic_cc[[col_alb, col_bun, col_be]].values
        y_mimic = mimic_cc[col_out].values.astype(int)
        X_mimic_s = scaler.transform(X_mimic)

        ext_rows = []
        for name, model in fitted.items():
            if name == "Logistic":
                B0, B_ALB, B_BUN, B_BE = 1.7825, -0.1227, 0.0343, -0.0585
                lp_ext = (B0
                          + B_ALB * X_mimic[:, 0]
                          + B_BUN * X_mimic[:, 1]
                          + B_BE  * X_mimic[:, 2])
                p_ext = 1.0 / (1.0 + np.exp(-lp_ext))
                p_ext_logistic = p_ext
            else:
                Xm = X_mimic_s if name in SCALED else X_mimic
                p_ext = model.predict_proba(Xm)[:, 1]
            ext_rows.append({
                "Model": name,
                "AUC_ext": round(roc_auc_score(y_mimic, p_ext), 4),
                "Brier_ext": round(brier_score_loss(y_mimic, p_ext), 4),
            })

        ext_df = pd.DataFrame(ext_rows).sort_values("AUC_ext", ascending=False)
        ext_df.to_csv(os.path.join(RESULTS_DIR, "python_external_validation.csv"),
                      index=False)

        if p_ext_logistic is not None:
            pred_df = pd.DataFrame({
                "outcome": y_mimic,
                "pred_prob": p_ext_logistic,
            })
            if "hadm_id" in mimic_cc.columns:
                pred_df.insert(0, "hadm_id", mimic_cc["hadm_id"].values)
            pred_df.to_csv(os.path.join(RESULTS_DIR, "external_predictions.csv"),
                           index=False)

# ============================================================
# 5. DeLong
# ============================================================
def delong_test(y_true, p1, p2):
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

if "Logistic" in fitted and len(res_df) > 1:
    best_ml = [m for m in res_df["Model"] if m != "Logistic"][0]
    p_log = fitted["Logistic"].predict_proba(X_val_s)[:, 1]
    X_ml = X_val_s if best_ml in SCALED else X_val
    p_ml = fitted[best_ml].predict_proba(X_ml)[:, 1]
    a1, a2, z, p = delong_test(y_val, p_log, p_ml)

# ============================================================
# 6. SHAP — 蓝-白-红
# ============================================================
from matplotlib.colors import LinearSegmentedColormap
CMAP_SHAP = LinearSegmentedColormap.from_list(
    "blue_red",
    ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"],
    N=256
)

if HAS_SHAP:
    tree_candidates = []
    if "XGBoost" in fitted:    tree_candidates.append("XGBoost")
    if "LightGBM" in fitted:   tree_candidates.append("LightGBM")
    if "RandomForest" in fitted: tree_candidates.append("RandomForest")

    if tree_candidates:
        best_tree = tree_candidates[0]
        explainer = shap.TreeExplainer(fitted[best_tree])
        shap_values = explainer.shap_values(X_train)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        # Summary plot
        plt.figure(figsize=(8, 5.0))
        shap.summary_plot(shap_values, X_train,
                          feature_names=FEATURES, plot_type="dot",
                          cmap=CMAP_SHAP, show=False, plot_size=None)
        plt.xlabel("SHAP value (log-odds)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR_CHEST, "FigureS3_SHAP_summary.pdf"),
                    dpi=1200, bbox_inches="tight")
        plt.close()

        # Bar plot
        plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_values, X_train,
                          feature_names=FEATURES, plot_type="bar",
                          color="#1F77B4", show=False, plot_size=None)
        plt.xlabel("mean(|SHAP value|)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR_CHEST, "FigureS3b_SHAP_bar.pdf"),
                    dpi=1200, bbox_inches="tight")
        plt.close()

# ============================================================
# 7. Figure 2: ROC + Calibration
# ============================================================
# 数据源：results/FROZEN_v1/external_predictions.csv（与 Table 3 / Table 5 / 正文一致）
_ext_path = os.path.join(RESULTS_DIR, "FROZEN_v1", "external_predictions.csv")
if not os.path.exists(_ext_path):
    _ext_path = os.path.join(RESULTS_DIR, "external_predictions.csv")

if os.path.exists(_ext_path):
    _ext = pd.read_csv(_ext_path)
    y_mimic = _ext["outcome"].astype(int).values
    p_ext_logistic = _ext["pred_prob"].astype(float).values

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # ================= A. ROC =================
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_mimic, p_ext_logistic)
    auc_ext = roc_auc_score(y_mimic, p_ext_logistic)
    auc_lo, auc_hi = EXT_AUC_CI

    ax.plot(fpr, tpr, color="#1F77B4", lw=1.25, label="Logistic (BE)")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--",
            label="Reference")

    ax.text(0.03, 0.97,
            f"AUC {auc_ext:.3f}\n(95% CI {auc_lo:.3f}–{auc_hi:.3f})",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left")

    ax.set_xlabel("1 − Specificity", fontsize=11)
    ax.set_ylabel("Sensitivity", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=8.5)
    ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

    # ================= B. Calibration =================
    ax = axes[1]
    frac_pos, mean_pred = calibration_curve(y_mimic, p_ext_logistic,
                                            n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color="#1F77B4",
            lw=1.25, markersize=4.5, label="Logistic (BE)")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--",
            label="Ideal")

    slope = 0.4196
    intercept = -0.6172
    ax.text(0.03, 0.97,
            f"Slope {slope:.3f}; Intercept {intercept:.3f}",
            transform=ax.transAxes, fontsize=8,
            va="top", ha="left")

    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Observed probability", fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=8.5)
    ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

    plt.subplots_adjust(bottom=0.28, wspace=0.35)
    out_fig2 = os.path.join(FIG_DIR_CHEST,
                            "Figure2_external_ROC_calibration.pdf")
    plt.savefig(out_fig2, dpi=1200)
    plt.close()
    print(f"[Figure 2] 数据源：{_ext_path}  n={len(y_mimic)}  events={int(y_mimic.sum())}  AUC={auc_ext:.4f}")
else:
    print("[Figure 2] 跳过：未找到 external_predictions.csv")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)