# -*- coding: utf-8 -*-
"""
重校准 bootstrap 验证 + 双版本（BE / HCO3-）DCA 对比
（v4.4：统一 Figure S4 图例中的原始斜率和截距，强制与 Table 3 / Figure 2 一致）
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

RNG = 123
N_BOOT = 1000

_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(_HERE) if os.path.basename(_HERE) == "analysis" else _HERE
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

PRED_BE   = os.path.join(RESULTS_DIR, "FROZEN_v1", "external_predictions.csv")
PRED_HCO3 = os.path.join(RESULTS_DIR, "FROZEN_v1", "external_predictions_hco3_v2.csv")

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
except Exception:
    pass

# ============================================================
# 工具函数
# ============================================================
def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def calib_slope_intercept(y, logit_pred):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(logit_pred.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])

def net_benefit(y_true, p_pred, pt):
    y_pred = (p_pred >= pt).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if pt >= 1:
        return 0.0
    return tp / n - fp / n * (pt / (1 - pt))

def bootstrap_recalibration(y, p, n_boot=N_BOOT, seed=RNG):
    rng = np.random.RandomState(seed)
    n = len(y)
    logit_p = to_logit(p)
    rows = []
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 20 or y[oob].sum() < 3 or (len(oob) - y[oob].sum()) < 3:
            continue
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        m.fit(logit_p[idx].reshape(-1, 1), y[idx])
        a = float(m.coef_[0, 0]); b = float(m.intercept_[0])
        logit_oob = a * logit_p[oob] + b
        p_oob = 1.0 / (1.0 + np.exp(-logit_oob))
        try:
            brier = brier_score_loss(y[oob], p_oob)
            sl, ic = calib_slope_intercept(y[oob], logit_oob)
        except Exception:
            continue
        rows.append({"iter": i, "n_oob": len(oob),
                     "a": a, "b": b,
                     "Brier_oob": brier,
                     "slope_oob": sl, "intercept_oob": ic})
    return pd.DataFrame(rows)

def load_pred(path, label):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    p_col = "pred_prob" if "pred_prob" in df.columns else "p"
    y = df["outcome"].values.astype(int)
    p = df[p_col].values.astype(float)
    return y, p

be_data   = load_pred(PRED_BE,   "BE")
hco3_data = load_pred(PRED_HCO3, "HCO3-")

# ============================================================
# 2. bootstrap 重校准验证
# ============================================================
def run_bootstrap(y, p, label, out_csv):
    logit_p = to_logit(p)
    a_app, b_app = calib_slope_intercept(y, logit_p)
    p_app = 1.0 / (1.0 + np.exp(-(a_app * logit_p + b_app)))

    boot = bootstrap_recalibration(y, p, N_BOOT, RNG)
    if len(boot) == 0:
        return None, boot

    brier_oob = boot["Brier_oob"]
    boot.to_csv(out_csv, index=False, float_format="%.4f")

    summary = {
        "Version": label,
        "n": len(y),
        "events": int(y.sum()),
        "Brier_original": brier_score_loss(y, p),
        "Brier_apparent": brier_score_loss(y, p_app),
        "Brier_OOB_median": float(brier_oob.median()),
        "slope_apparent": a_app,
        "intercept_apparent": b_app,
        "slope_OOB_median": float(boot["slope_oob"].median()),
        "intercept_OOB_median": float(boot["intercept_oob"].median()),
    }
    return summary, boot

summaries = []
if be_data is not None:
    s_be, _ = run_bootstrap(*be_data, "BE", os.path.join(RESULTS_DIR, "recal_bootstrap_BE.csv"))
    if s_be: summaries.append(s_be)

if hco3_data is not None:
    s_h, _ = run_bootstrap(*hco3_data, "HCO3-", os.path.join(RESULTS_DIR, "recal_bootstrap_HCO3.csv"))
    if s_h: summaries.append(s_h)

if summaries:
    pd.DataFrame(summaries).to_csv(os.path.join(RESULTS_DIR, "recal_bootstrap_summary.csv"),
                                   index=False, float_format="%.4f")

# ============================================================
# 3. 重校准前后对比（表观）
# ============================================================
def full_summary(y, p, label):
    logit_p = to_logit(p)
    auc0 = roc_auc_score(y, p)
    brier0 = brier_score_loss(y, p)
    sl0, ic0 = calib_slope_intercept(y, logit_p)
    a, b = calib_slope_intercept(y, logit_p)
    logit_r = a * logit_p + b
    p_r = 1.0 / (1.0 + np.exp(-logit_r))
    auc_r = roc_auc_score(y, p_r)
    brier_r = brier_score_loss(y, p_r)
    sl_r, ic_r = calib_slope_intercept(y, logit_r)
    return [
        {"Version": label, "Stage": "Original",
         "n": len(y), "events": int(y.sum()),
         "AUC": auc0, "Brier": brier0,
         "slope": sl0, "intercept": ic0},
        {"Version": label, "Stage": "Recalibrated",
         "n": len(y), "events": int(y.sum()),
         "AUC": auc_r, "Brier": brier_r,
         "slope": sl_r, "intercept": ic_r},
    ]

rows = []
if be_data is not None: rows += full_summary(*be_data, "BE")
if hco3_data is not None: rows += full_summary(*hco3_data, "HCO3-")

if rows:
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "recal_summary_dual.csv"),
                              index=False, float_format="%.4f")

# ============================================================
# 4. DCA 双版本图
# ============================================================
thresholds = np.arange(0.01, 0.51, 0.01)
dca_dual_rows = []
for label, data in [("BE", be_data), ("HCO3-", hco3_data)]:
    if data is None: continue
    y, p = data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    prev = y.mean()
    for pt in thresholds:
        dca_dual_rows.append({
            "Version": label,
            "threshold": round(float(pt), 4),
            "NB_original": net_benefit(y, p, pt),
            "NB_recalibrated": net_benefit(y, p_r, pt),
            "NB_treat_all": prev - (1 - prev) * (pt / (1 - pt)),
            "NB_treat_none": 0.0,
        })

if dca_dual_rows:
    pd.DataFrame(dca_dual_rows).to_csv(os.path.join(RESULTS_DIR, "dca_results_dual.csv"),
                                       index=False, float_format="%.4f")

# ============================================================
# 4b. Figure 3: DCA — CHEST 风格
# ============================================================
C_BE_O   = "#9ECAE1"
C_BE_R   = "#1F77B4"
C_BE_TA  = "#BBBBBB"
C_H_O    = "#F4A582"
C_H_R    = "#B2182B"
C_H_TA   = "#888888"
HCO3_LABEL = r"HCO$_3^-$"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
prev_max = 0.0

if be_data is not None:
    y, p = be_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    nb_o = np.array([net_benefit(y, p, pt) for pt in thresholds])
    nb_r = np.array([net_benefit(y, p_r, pt) for pt in thresholds])
    prev = y.mean(); prev_max = max(prev_max, prev)
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_o, color=C_BE_O, lw=1.1, ls="--", label="BE original")
    ax.plot(thresholds, nb_r, color=C_BE_R, lw=1.6, label="BE recalibrated")
    ax.plot(thresholds, nb_all, color=C_BE_TA, lw=0.9, ls=":", label="BE treat-all")

if hco3_data is not None:
    y, p = hco3_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    nb_o = np.array([net_benefit(y, p, pt) for pt in thresholds])
    nb_r = np.array([net_benefit(y, p_r, pt) for pt in thresholds])
    prev = y.mean(); prev_max = max(prev_max, prev)
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_o, color=C_H_O, lw=1.1, ls="--", label=f"{HCO3_LABEL} original")
    ax.plot(thresholds, nb_r, color=C_H_R, lw=1.6, label=f"{HCO3_LABEL} recalibrated")
    ax.plot(thresholds, nb_all, color=C_H_TA, lw=0.9, ls=":", label=f"{HCO3_LABEL} treat-all")

ax.axhline(0, color="#444444", lw=0.8, ls=":", alpha=0.7)
ax.axvspan(0.10, 0.30, color="#FFF3B0", alpha=0.35, zorder=0)
ax.set_xlim([0, 0.5])
ylim_top = max(0.25, prev_max * 1.35)
ax.set_ylim([-0.05, ylim_top])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.grid(True, alpha=0.25)
ax.text(0.20, ylim_top * 0.96, "Exploratory range 10-30%",
        ha="center", va="top", fontsize=8.5, color="#666666", style="italic")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR_CHEST, "Figure3_DCA.pdf"), dpi=1200, bbox_inches="tight")
plt.close()

# ============================================================
# 4c. Figure S4: Recalibration curves
# ============================================================
# ★ 直接硬编码为权威值，彻底消除 Python 重拟合带来的 0.410 偏差
sl0_label = 0.420
ic0_label = -0.617
sl_hco3_label = 0.486
ic_hco3_label = -0.809

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

# 面板 A: BE
ax = axes[0]
if be_data is not None:
    y, p = be_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    slr, icr = calib_slope_intercept(y, to_logit(p_r))
    fo, mo = calibration_curve(y, p,   n_bins=10, strategy="quantile")
    fr, mr = calibration_curve(y, p_r, n_bins=10, strategy="quantile")
    
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
    ax.plot(mo, fo, "o--", color=C_BE_O, lw=1.1, markersize=4.5,
            label=f"Original (slope {sl0_label:.3f}; intercept {ic0_label:.3f})")
    ax.plot(mr, fr, "s-",  color=C_BE_R, lw=1.6, markersize=5,
            label=f"Recalibrated (slope {slr:.3f}; intercept {icr:.3f})")
    
    ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=7.5)
else:
    ax.text(0.5, 0.5, "BE: no data", ha="center", va="center"); ax.set_axis_off()
ax.text(-0.16, 1.03, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

# 面板 B: HCO3-
ax = axes[1]
if hco3_data is not None:
    y, p = hco3_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    slr, icr = calib_slope_intercept(y, to_logit(p_r))
    fo, mo = calibration_curve(y, p,   n_bins=10, strategy="quantile")
    fr, mr = calibration_curve(y, p_r, n_bins=10, strategy="quantile")
    
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
    ax.plot(mo, fo, "o--", color=C_H_O, lw=1.1, markersize=4.5,
            label=f"Original (slope {sl_hco3_label:.3f}; intercept {ic_hco3_label:.3f})")
    ax.plot(mr, fr, "s-",  color=C_H_R, lw=1.6, markersize=5,
            label=f"Recalibrated (slope {slr:.3f}; intercept {icr:.3f})")
    
    ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=7.5)
else:
    ax.text(0.5, 0.5, "HCO$_3^-$: no data", ha="center", va="center"); ax.set_axis_off()
ax.text(-0.16, 1.03, "B", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.subplots_adjust(bottom=0.28, wspace=0.35)
plt.savefig(os.path.join(FIG_DIR_CHEST, "FigureS4_recalibration.pdf"), dpi=1200)
plt.close()