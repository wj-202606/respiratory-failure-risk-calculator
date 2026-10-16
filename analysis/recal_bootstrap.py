# -*- coding: utf-8 -*-
"""
重校准 bootstrap 验证 + 双版本（BE / HCO3-）DCA 对比
（v2：DCA 存盘放在第 [4/4] 节最开头，确保一定执行并打印）
（v3：CHEST 风格 Figure 3 / Figure S4，输出到 论文图/，Arial，无图内标题，加 A/B）
（v3.1：HCO3- 版改读 external_predictions_hco3_v2.csv（v3.2 权威 50 套合并系数）；
       load_pred 兼容 pred_prob / p 两种列名；
       legend 里 HCO₃⁻ 改用 mathtext HCO$_3^-$ 避免 Arial 缺字形；
       临床区间标注 10-30% 改 hyphen）

输入：
  results/external_predictions.csv            (BE 版，n=434)
  results/external_predictions_hco3_v2.csv    (HCO3- 版 v3.2，n=684)

输出（保持不变）：
  results/recal_bootstrap_BE.csv
  results/recal_bootstrap_HCO3.csv
  results/recal_bootstrap_summary.csv
  results/recal_summary_dual.csv
  results/dca_results_dual.csv
  figures/Fig3_DCA_dual.pdf
  figures/FigS3_recalibration_dual.pdf

输出（v3 CHEST 风格）：
  论文图/Figure3_DCA.pdf
  论文图/FigureS4_recalibration.pdf

⚠️ BE 版：D:/mimic_wide.csv 的 be_mmol_l 是真 BE
   HCO3- 版：D:/mimic_wide_hco3.csv 的 be_mmol_l 实际是 HCO3-
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

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

PRED_BE   = os.path.join(RESULTS_DIR, "external_predictions.csv")
PRED_HCO3 = os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv")  # ★ v3.1 改动

print("=" * 70)
print("Python:", sys.executable)
print("脚本文件:", os.path.abspath(__file__))
print("BASE:", BASE)
print("RESULTS_DIR:", RESULTS_DIR)
print("FIG_DIR_CHEST:", FIG_DIR_CHEST)
print("PRED_BE   :", PRED_BE)
print("PRED_HCO3 :", PRED_HCO3)
print("=" * 70)

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style，使用默认: {_e}")

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

# ============================================================
# 1. 读两个版本
# ============================================================
print("\n[1/4] 读入预测文件")

def load_pred(path, label):
    """★ v3.1：兼容 pred_prob / p 两种列名"""
    if not os.path.exists(path):
        print(f"  [{label}] 跳过：找不到 {path}")
        return None
    df = pd.read_csv(path)
    if "outcome" not in df.columns:
        raise KeyError(f"{path} 缺少 outcome 列")
    if "pred_prob" in df.columns:
        p_col = "pred_prob"
    elif "p" in df.columns:
        p_col = "p"
    else:
        raise KeyError(f"{path} 缺少 pred_prob / p 列")
    y = df["outcome"].values.astype(int)
    p = df[p_col].values.astype(float)
    print(f"  [{label}] n={len(df)}, 事件={int(y.sum())} ({y.mean()*100:.1f}%), "
          f"pred 范围 [{p.min():.4f},{p.max():.4f}]  (col={p_col})")
    return y, p

be_data   = load_pred(PRED_BE,   "BE")
hco3_data = load_pred(PRED_HCO3, "HCO3-")

if be_data is None and hco3_data is None:
    raise FileNotFoundError("两个预测文件都不存在，无法继续。")

# ============================================================
# 2. bootstrap 重校准验证
# ============================================================
print(f"\n[2/4] Bootstrap 重校准验证（n_boot={N_BOOT}）")

def run_bootstrap(y, p, label, out_csv):
    logit_p = to_logit(p)
    a_app, b_app = calib_slope_intercept(y, logit_p)
    p_app = 1.0 / (1.0 + np.exp(-(a_app * logit_p + b_app)))

    print(f"\n  --- {label} ---")
    print(f"  表观重校准: a={a_app:.4f}, b={b_app:.4f}")
    print(f"  原始 Brier = {brier_score_loss(y, p):.4f}")
    print(f"  表观重校准 Brier = {brier_score_loss(y, p_app):.4f}")

    boot = bootstrap_recalibration(y, p, N_BOOT, RNG)
    if len(boot) == 0:
        print("  [警告] bootstrap 有效迭代为 0，跳过")
        return None, boot

    brier_oob = boot["Brier_oob"]
    print(f"  Bootstrap 有效迭代: {len(boot)} / {N_BOOT}")
    print(f"  OOB Brier  : 中位数 {brier_oob.median():.4f} "
          f"(IQR {brier_oob.quantile(0.25):.4f}-{brier_oob.quantile(0.75):.4f})")
    print(f"  OOB slope  : 中位数 {boot['slope_oob'].median():.4f} "
          f"(IQR {boot['slope_oob'].quantile(0.25):.4f}-{boot['slope_oob'].quantile(0.75):.4f})")
    print(f"  OOB interc.: 中位数 {boot['intercept_oob'].median():.4f} "
          f"(IQR {boot['intercept_oob'].quantile(0.25):.4f}-{boot['intercept_oob'].quantile(0.75):.4f})")
    print(f"  a 分布     : 中位数 {boot['a'].median():.4f} "
          f"(IQR {boot['a'].quantile(0.25):.4f}-{boot['a'].quantile(0.75):.4f})")

    boot.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"  已保存: {out_csv}")

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
    s_be, boot_be = run_bootstrap(*be_data, "BE",
                                  os.path.join(RESULTS_DIR, "recal_bootstrap_BE.csv"))
    if s_be: summaries.append(s_be)

if hco3_data is not None:
    s_h, boot_h = run_bootstrap(*hco3_data, "HCO3-",
                                os.path.join(RESULTS_DIR, "recal_bootstrap_HCO3.csv"))
    if s_h: summaries.append(s_h)

if summaries:
    sum_df = pd.DataFrame(summaries)
    sum_df.to_csv(os.path.join(RESULTS_DIR, "recal_bootstrap_summary.csv"),
                  index=False, float_format="%.4f")
    print(f"\n  已保存: results/recal_bootstrap_summary.csv")

# ============================================================
# 3. 重校准前后对比（表观）
# ============================================================
print("\n[3/4] 重校准前后对比（表观）")

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
if be_data is not None:
    rows += full_summary(*be_data, "BE")
if hco3_data is not None:
    rows += full_summary(*hco3_data, "HCO3-")

if rows:
    dual_df = pd.DataFrame(rows)
    dual_df.to_csv(os.path.join(RESULTS_DIR, "recal_summary_dual.csv"),
                   index=False, float_format="%.4f")
    print(dual_df.to_string(index=False))
    print(f"\n  已保存: results/recal_summary_dual.csv")

# ============================================================
# 4. DCA 双版本图 + 重校准曲线
# ============================================================
print("\n[4/4] DCA + 画图")

thresholds = np.arange(0.01, 0.51, 0.01)

# -------- 4a. 先算并保存 DCA 数值 --------
dca_dual_rows = []
for label, data in [("BE", be_data), ("HCO3-", hco3_data)]:
    if data is None:
        continue
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
    dca_dual_df = pd.DataFrame(dca_dual_rows)
    out_dca = os.path.join(RESULTS_DIR, "dca_results_dual.csv")
    dca_dual_df.to_csv(out_dca, index=False, float_format="%.4f")
    print(f"\n  已保存: {out_dca} "
          f"({len(dca_dual_df)} 行, {dca_dual_df['Version'].nunique()} 个版本)")
    print(f"\n  {'版本':<8}{'阈值':>7}{'原始':>12}{'重校准':>12}{'TreatAll':>12}")
    for label in dca_dual_df["Version"].unique():
        sub = dca_dual_df[dca_dual_df["Version"] == label]
        for pt in [0.10, 0.15, 0.20, 0.30]:
            row = sub.iloc[(sub["threshold"] - pt).abs().argmin()]
            print(f"  {label:<8}{row['threshold']:>7.2f}"
                  f"{row['NB_original']:>12.4f}"
                  f"{row['NB_recalibrated']:>12.4f}"
                  f"{row['NB_treat_all']:>12.4f}")
else:
    print("  [警告] 无 DCA 数据可保存")

# ============================================================
# 4b. Figure 3: DCA — CHEST 风格
# ============================================================
print("\n  画 Figure 3 (DCA, CHEST style) ...")

# 配色
C_BE_O   = "#9ECAE1"
C_BE_R   = "#1F77B4"
C_BE_TA  = "#BBBBBB"
C_H_O    = "#F4A582"
C_H_R    = "#B2182B"
C_H_TA   = "#888888"

# ★ v3.1：HCO3- 的图例统一用 mathtext，避免 Arial 缺字形
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
    ax.plot(thresholds, nb_o, color=C_BE_O, lw=1.1, ls="--",
            label="BE original")
    ax.plot(thresholds, nb_r, color=C_BE_R, lw=1.6,
            label="BE recalibrated")
    ax.plot(thresholds, nb_all, color=C_BE_TA, lw=0.9, ls=":",
            label="BE treat-all")

if hco3_data is not None:
    y, p = hco3_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    nb_o = np.array([net_benefit(y, p, pt) for pt in thresholds])
    nb_r = np.array([net_benefit(y, p_r, pt) for pt in thresholds])
    prev = y.mean(); prev_max = max(prev_max, prev)
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_o, color=C_H_O, lw=1.1, ls="--",
            label=f"{HCO3_LABEL} original")
    ax.plot(thresholds, nb_r, color=C_H_R, lw=1.6,
            label=f"{HCO3_LABEL} recalibrated")
    ax.plot(thresholds, nb_all, color=C_H_TA, lw=0.9, ls=":",
            label=f"{HCO3_LABEL} treat-all")

ax.axhline(0, color="#444444", lw=0.8, ls=":", alpha=0.7)
ax.axvspan(0.10, 0.30, color="#FFF3B0", alpha=0.35, zorder=0)

ax.set_xlim([0, 0.5])
ylim_top = max(0.25, prev_max * 1.35)
ax.set_ylim([-0.05, ylim_top])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")

ax.grid(True, alpha=0.25)

# ★ v3.1：en dash → hyphen
ax.text(0.20, ylim_top * 0.96, "Clinical range 10-30%",
        ha="center", va="top", fontsize=8.5,
        color="#666666", style="italic")

ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.95,
          edgecolor="#CCCCCC", fontsize=8.5,
          ncol=1, borderpad=0.6)

plt.tight_layout()
out_fig3_chest = os.path.join(FIG_DIR_CHEST, "Figure3_DCA.pdf")
plt.savefig(out_fig3_chest, dpi=1200, bbox_inches="tight")
plt.close()
print(f"  已保存 (CHEST style): {out_fig3_chest}")

# 旧版兼容（figures/Fig3_DCA_dual.pdf）
fig, ax = plt.subplots(figsize=(9, 6))
if be_data is not None:
    y, p = be_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    nb_o = np.array([net_benefit(y, p, pt) for pt in thresholds])
    nb_r = np.array([net_benefit(y, p_r, pt) for pt in thresholds])
    prev = y.mean()
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_o, color="steelblue", lw=1.6, ls="--", label="BE original")
    ax.plot(thresholds, nb_r, color="steelblue", lw=2.2, label="BE recalibrated")
    ax.plot(thresholds, nb_all, color="steelblue", lw=1.0, ls=":", alpha=0.7, label="BE treat-all")
if hco3_data is not None:
    y, p = hco3_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    nb_o = np.array([net_benefit(y, p, pt) for pt in thresholds])
    nb_r = np.array([net_benefit(y, p_r, pt) for pt in thresholds])
    prev = y.mean()
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_o, color="firebrick", lw=1.6, ls="--", label="HCO3- original")
    ax.plot(thresholds, nb_r, color="firebrick", lw=2.2, label="HCO3- recalibrated")
    ax.plot(thresholds, nb_all, color="firebrick", lw=1.0, ls=":", alpha=0.7, label="HCO3- treat-all")
ax.axhline(0, color="black", lw=1.0, ls=":", alpha=0.6)
ax.axvspan(0.10, 0.30, color="lightyellow", alpha=0.45, zorder=0, label="Clinical range 10-30%")
ax.set_xlim([0, 0.5])
ax.set_ylim([-0.05, max(0.3, prev_max * 1.4)])
ax.set_xlabel("Threshold probability", fontsize=12)
ax.set_ylabel("Net benefit", fontsize=12)
ax.set_title("Decision Curve Analysis: BE vs HCO3- (MIMIC-IV)", fontsize=13)
ax.legend(loc="upper right", fontsize=9, ncol=2, frameon=True)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "Fig3_DCA_dual.pdf"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"  已保存 (旧版兼容): figures/Fig3_DCA_dual.pdf")

# ============================================================
# 4c. Figure S4: Recalibration curves — CHEST 风格
# ============================================================
print("\n  画 Figure S4 (Recalibration curves, CHEST style) ...")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

# 面板 A: BE
ax = axes[0]
if be_data is not None:
    y, p = be_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    sl0, ic0 = calib_slope_intercept(y, logit_p)
    slr, icr = calib_slope_intercept(y, to_logit(p_r))
    fo, mo = calibration_curve(y, p,   n_bins=10, strategy="quantile")
    fr, mr = calibration_curve(y, p_r, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
    ax.plot(mo, fo, "o--", color=C_BE_O, lw=1.1, markersize=4.5,
            label=f"Original (slope {sl0:.2f}; intercept {ic0:.2f})")
    ax.plot(mr, fr, "s-",  color=C_BE_R, lw=1.6, markersize=5,
            label=f"Recalibrated (slope {slr:.2f}; intercept {icr:.2f})")
    ax.set_xlim([0, 0.4]); ax.set_ylim([0, 0.4])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", fontsize=7.5, borderpad=0.5)
else:
    ax.text(0.5, 0.5, "BE: no data", ha="center", va="center")
    ax.set_axis_off()

ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

# 面板 B: HCO3-
ax = axes[1]
if hco3_data is not None:
    y, p = hco3_data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    sl0, ic0 = calib_slope_intercept(y, logit_p)
    slr, icr = calib_slope_intercept(y, to_logit(p_r))
    fo, mo = calibration_curve(y, p,   n_bins=10, strategy="quantile")
    fr, mr = calibration_curve(y, p_r, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
    ax.plot(mo, fo, "o--", color=C_H_O, lw=1.1, markersize=4.5,
            label=f"Original (slope {sl0:.2f}; intercept {ic0:.2f})")
    ax.plot(mr, fr, "s-",  color=C_H_R, lw=1.6, markersize=5,
            label=f"Recalibrated (slope {slr:.2f}; intercept {icr:.2f})")
    ax.set_xlim([0, 0.45]); ax.set_ylim([0, 0.45])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC", fontsize=7.5, borderpad=0.5)
else:
    ax.text(0.5, 0.5, "HCO$_3^-$: no data", ha="center", va="center")
    ax.set_axis_off()

ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="bottom", ha="left")

plt.tight_layout()
out_figs4_chest = os.path.join(FIG_DIR_CHEST, "FigureS4_recalibration.pdf")
plt.savefig(out_figs4_chest, dpi=1200, bbox_inches="tight")
plt.close()
print(f"  已保存 (CHEST style): {out_figs4_chest}")

# 旧版兼容（figures/FigS3_recalibration_dual.pdf）
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, data, label, color in [
    (axes[0], be_data,   "BE (n=434)",    "steelblue"),
    (axes[1], hco3_data, "HCO3- (n=684)", "firebrick"),
]:
    if data is None:
        ax.text(0.5, 0.5, f"{label}: no data", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        continue
    y, p = data
    logit_p = to_logit(p)
    a, b = calib_slope_intercept(y, logit_p)
    p_r = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
    sl0, ic0 = calib_slope_intercept(y, logit_p)
    slr, icr = calib_slope_intercept(y, to_logit(p_r))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    fo, mo = calibration_curve(y, p,   n_bins=10, strategy="quantile")
    fr, mr = calibration_curve(y, p_r, n_bins=10, strategy="quantile")
    ax.plot(mo, fo, "o--", color=color, lw=1.6, markersize=6, alpha=0.7,
            label=f"Original (slope={sl0:.2f}, int={ic0:.2f})")
    ax.plot(mr, fr, "s-",  color=color, lw=2.2, markersize=7,
            label=f"Recalibrated (slope={slr:.2f}, int={icr:.2f})")
    ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
    ax.set_xlabel("Predicted probability", fontsize=12)
    ax.set_ylabel("Observed probability", fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "FigS3_recalibration_dual.pdf"),
            dpi=300, bbox_inches="tight")
plt.close()
print(f"  已保存 (旧版兼容): figures/FigS3_recalibration_dual.pdf")

print("\n" + "=" * 70)
print("完成！输出文件：")
print("  results/recal_bootstrap_BE.csv")
print("  results/recal_bootstrap_HCO3.csv")
print("  results/recal_bootstrap_summary.csv")
print("  results/recal_summary_dual.csv")
print("  results/dca_results_dual.csv")
print("  figures/Fig3_DCA_dual.pdf              (旧版兼容)")
print("  figures/FigS3_recalibration_dual.pdf   (旧版兼容)")
print("  论文图/Figure3_DCA.pdf                 (CHEST 风格)")
print("  论文图/FigureS4_recalibration.pdf      (CHEST 风格)")
print("=" * 70)