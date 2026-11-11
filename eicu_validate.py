# -*- coding: utf-8 -*-
"""
eICU 外部验证：BE 版 + HCO3- 版
v6.2：保持与MIMIC预测文件独立的输入输出链。
v6.0：
  [1] 本地校准/OOB/bootstrap 实现抽到 shared_calib.py，本脚本只 import
      （与 recal_bootstrap.py 共用同一份实现）
  [2] OOB 汇总用 shared_calib.summarize_oob()

v6.0（保留）：
  [1] 部署系数统一从 results/external_validation_coef*.csv 读取
  [2] 校准回归改为精确无惩罚 logistic（penalty=None）
  [3] 其余逻辑（AUC/Brier CI、OOB、DCA、Figure S6/S7/S8）不变

时间口径（方案 B）：ICU 入科为时间起点，预测时点 = ICU 入科后 24h。
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

from shared_calib import (
    to_logit,
    calib_slope_intercept,
    net_benefit,
    bootstrap_ci_auc_brier,
    bootstrap_oob_recal,
    summarize_oob,
    DEFAULT_N_BOOT,
    DEFAULT_N_CI,
    DEFAULT_SEED,
)

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style: {_e}")

RNG = DEFAULT_SEED
N_BOOT = DEFAULT_N_BOOT
N_CI  = DEFAULT_N_CI

DEPLOY_COEF_BE   = os.path.join(RESULTS_DIR, "external_validation_coef.csv")
DEPLOY_COEF_HCO3 = os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv")

# ============================================================
# 0. 读取部署系数
# ============================================================
def load_deploy_coef(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到 {label} 部署系数: {path}")
    df = pd.read_csv(path)
    term_col = "Term" if "Term" in df.columns else df.columns[0]
    beta_col = "Beta" if "Beta" in df.columns else (
        "estimate" if "estimate" in df.columns else df.columns[1])
    coef = dict(zip(df[term_col].astype(str), df[beta_col].astype(float)))
    if "(Intercept)" in coef and "Intercept" not in coef:
        coef["Intercept"] = coef.pop("(Intercept)")
    expected = ["Intercept", "ALB", "BUN", ("BE" if label == "BE" else "HCO3")]
    missing = [k for k in expected if k not in coef]
    if missing:
        raise KeyError(f"{label} 系数文件缺少必要项: {missing}")
    extra = [k for k in coef if k not in expected]
    if extra:
        raise ValueError(f"{label} 系数文件含模型定义之外的项: {extra}")
    if not all(np.isfinite(v) for v in coef.values()):
        raise ValueError(f"{label} 系数文件含非有限值")
    return coef

DEPLOY_BE = load_deploy_coef(DEPLOY_COEF_BE, "BE")
print(f"[coef] BE: {DEPLOY_BE}")

DEPLOY_HCO3 = load_deploy_coef(DEPLOY_COEF_HCO3, "HCO3")
print(f"[coef] HCO3: {DEPLOY_HCO3}")

def predict_frozen(df, coef, x_col):
    lp = (coef["Intercept"]
          + coef["ALB"] * df["ALB"].values
          + coef["BUN"] * df["BUN"].values
          + coef[x_col] * df[x_col].values)
    return 1.0 / (1.0 + np.exp(-lp))

# ============================================================
# 主分析函数
# ============================================================
def run_version(version, coef, csv_path, out_summary, out_pred, out_fig):
    x_col = "BE" if version == "BE" else "HCO3"
    label = "BE" if version == "BE" else r"HCO$_3^-$"
    color = "#1F77B4" if version == "BE" else "#B2182B"

    print("\n" + "=" * 60)
    print(f"eICU 外部验证 — {version} 版（方案 B：ICU 24h 口径）")
    print("=" * 60)

    if coef is None:
        print(f"  [跳过] 缺少 {version} 部署系数")
        return None, None, None, None

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    required = ["ALB", "BUN", x_col, "outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{version} 队列缺少列: {missing}")
    if df[required].isna().any().any():
        raise ValueError(f"{version} 完整案例数据仍有缺失")
    y = pd.to_numeric(df["outcome"], errors="raise").to_numpy(dtype=int)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{version} outcome必须同时包含0和1")
    p = predict_frozen(df, coef, x_col)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{version} 预测概率非法")
    print(f"  n={len(y)}, events={int(y.sum())} ({y.mean()*100:.1f}%)")

    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    lp = to_logit(p)
    slope, intercept = calib_slope_intercept(y, lp)
    ci_res = bootstrap_ci_auc_brier(y, p)
auc_lo, auc_hi = ci_res["auc_lo"], ci_res["auc_hi"]
br_lo, br_hi = ci_res["brier_lo"], ci_res["brier_hi"]

    lp_r = slope * lp + intercept
    p_r = 1.0 / (1.0 + np.exp(-lp_r))
    brier_r = brier_score_loss(y, p_r)
    sl_r, ic_r = calib_slope_intercept(y, lp_r)

    oob = bootstrap_oob_recal(y, p)
    oob_summary = summarize_oob(oob)

    print(f"  AUC={auc:.4f} (95% CI {auc_lo:.4f}-{auc_hi:.4f})")
    print(f"  Brier={brier:.4f} (95% CI {br_lo:.4f}-{br_hi:.4f})")
    print(f"  Calibration slope={slope:.4f}, intercept={intercept:.4f}")
    print(f"  Recalibrated: Brier={brier_r:.4f} "
          f"(Δ={100*(brier_r-brier)/brier:+.1f}%), "
          f"slope={sl_r:.4f}, intercept={ic_r:.4f}")
    print(f"  OOB: Brier={oob_summary['Brier_OOB_median']:.4f} "
          f"[{oob_summary['Brier_OOB_q25']:.4f}-{oob_summary['Brier_OOB_q75']:.4f}], "
          f"slope={oob_summary['slope_OOB_median']:.4f} "
          f"[{oob_summary['slope_OOB_q25']:.4f}-{oob_summary['slope_OOB_q75']:.4f}], "
          f"intercept={oob_summary['intercept_OOB_median']:.4f} "
          f"[{oob_summary['intercept_OOB_q25']:.4f}-{oob_summary['intercept_OOB_q75']:.4f}]")

    summary = pd.DataFrame([{
        "Version": version, "Stage": "Original",
        "n": len(y), "events": int(y.sum()),
        "AUC": auc, "AUC_lo": auc_lo, "AUC_hi": auc_hi,
        "Brier": brier, "Brier_lo": br_lo, "Brier_hi": br_hi,
        "cal_slope": slope, "cal_intercept": intercept,
        "recal_a": slope, "recal_b": intercept,
        "Brier_recal": brier_r, "slope_recal": sl_r, "intercept_recal": ic_r,
        "OOB_Brier_median": oob_summary["Brier_OOB_median"],
        "OOB_Brier_q25":    oob_summary["Brier_OOB_q25"],
        "OOB_Brier_q75":    oob_summary["Brier_OOB_q75"],
        "OOB_slope_median": oob_summary["slope_OOB_median"],
        "OOB_slope_q25":    oob_summary["slope_OOB_q25"],
        "OOB_slope_q75":    oob_summary["slope_OOB_q75"],
        "OOB_intercept_median": oob_summary["intercept_OOB_median"],
        "OOB_intercept_q25":    oob_summary["intercept_OOB_q25"],
        "OOB_intercept_q75":    oob_summary["intercept_OOB_q75"],
    }])
    summary.to_csv(out_summary, index=False, float_format="%.6f")
    print(f"  已保存: {out_summary}")

    pred_df = pd.DataFrame({"outcome": y, "pred_prob": p,
                            "pred_prob_recal": p_r})
    if "patientunitstayid" in df.columns:
        pred_df.insert(0, "patientunitstayid", df["patientunitstayid"].values)
    pred_df.to_csv(out_pred, index=False, float_format="%.6f")
    print(f"  已保存: {out_pred}")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    ax = axes[0]
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=color, lw=1.25,
            label=f"3-variable logistic ({label})")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Reference")
    ax.text(0.03, 0.97,
            f"AUC {auc:.3f}\n(95% CI {auc_lo:.3f}–{auc_hi:.3f})",
            transform=ax.transAxes, fontsize=8, va="top", ha="left")
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=8.5)
    ax.text(-0.16, 1.03, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

    ax = axes[1]
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ax.plot(mean_p, frac, "o-", color=color, lw=1.25, markersize=4.5,
            label=f"3-variable logistic ({label})")
    ax.plot([0, 1], [0, 1], color="#777777", lw=0.9, ls="--", label="Ideal")
    ax.text(0.03, 0.97,
            f"Slope {slope:.3f}; Intercept {intercept:.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="left")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_xlim([0, 0.5]); ax.set_ylim([0, 0.5])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, frameon=False, fontsize=8.5)
    ax.text(-0.16, 1.03, "B", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

    plt.subplots_adjust(bottom=0.28, wspace=0.35)
    plt.savefig(out_fig, dpi=1200)
    plt.close()
    print(f"  已保存: {out_fig}")

    return df, y, p, summary

# ============================================================
# 跑 BE
# ============================================================
df_be, y_be, p_be, sum_be = run_version(
    "BE", DEPLOY_BE,
    os.path.join(RESULTS_DIR, "eicu_wide_be.csv"),
    os.path.join(RESULTS_DIR, "eicu_be_summary.csv"),
    os.path.join(RESULTS_DIR, "eicu_predictions_be.csv"),
    os.path.join(FIG_DIR_CHEST, "FigureS6_eicu_ROC_calibration.pdf"),
)

# ============================================================
# 跑 HCO3
# ============================================================
df_hco3, y_hco3, p_hco3, sum_hco3 = run_version(
    "HCO3", DEPLOY_HCO3,
    os.path.join(RESULTS_DIR, "eicu_wide_hco3.csv"),
    os.path.join(RESULTS_DIR, "eicu_hco3_summary.csv"),
    os.path.join(RESULTS_DIR, "eicu_predictions_hco3.csv"),
    os.path.join(FIG_DIR_CHEST, "FigureS7_eicu_hco3_ROC_calibration.pdf"),
)

# ============================================================
# DCA 双版本
# ============================================================
print("\n" + "=" * 60)
print("eICU DCA 双版本（方案 B：ICU 24h 口径）")
print("=" * 60)

thresholds = np.arange(0.01, 0.51, 0.01)
dca_rows = []

for version, df, y, p in [("BE", df_be, y_be, p_be),
                          ("HCO3", df_hco3, y_hco3, p_hco3)]:
    if y is None:
        continue
    lp = to_logit(p)
    a, b = calib_slope_intercept(y, lp)
    p_r = 1.0 / (1.0 + np.exp(-(a*lp + b)))
    prev = y.mean()
    for pt in thresholds:
        dca_rows.append({
            "Version": version,
            "threshold": round(float(pt), 4),
            "NB_original": net_benefit(y, p, pt),
            "NB_recalibrated": net_benefit(y, p_r, pt),
            "NB_treat_all": prev - (1 - prev) * (pt / (1 - pt)),
            "NB_treat_none": 0.0,
        })

dca_df = pd.DataFrame(dca_rows)
dca_df.to_csv(os.path.join(RESULTS_DIR, "eicu_dca.csv"),
              index=False, float_format="%.4f")
print(f"  已保存: results/eicu_dca.csv ({len(dca_df)} 行)")

print("\n  代表阈值净获益：")
print(f"  {'版本':<8}{'阈值':>7}{'原始':>12}{'重校准':>12}{'TreatAll':>12}")
for version in ["BE", "HCO3"]:
    sub = dca_df[dca_df["Version"] == version]
    if len(sub) == 0:
        continue
    for pt in [0.10, 0.15, 0.20, 0.30]:
        row = sub.iloc[(sub["threshold"] - pt).abs().argmin()]
        print(f"  {version:<8}{row['threshold']:>7.2f}"
              f"{row['NB_original']:>12.4f}{row['NB_recalibrated']:>12.4f}"
              f"{row['NB_treat_all']:>12.4f}")

# ============================================================
# Figure S8: DCA
# ============================================================
C_BE_O  = "#9ECAE1"; C_BE_R  = "#1F77B4"; C_BE_TA = "#BBBBBB"
C_H_O   = "#F4A582"; C_H_R   = "#B2182B"; C_H_TA  = "#888888"
HCO3_LABEL = r"HCO$_3^-$"

fig, ax = plt.subplots(figsize=(7.2, 4.6))
prev_max = 0.0

for version, label_o, label_r, label_ta, c_o, c_r, c_ta in [
    ("BE",    "BE original",              "BE recalibrated",              "BE treat-all",
     C_BE_O, C_BE_R, C_BE_TA),
    ("HCO3",  f"{HCO3_LABEL} original",   f"{HCO3_LABEL} recalibrated",   f"{HCO3_LABEL} treat-all",
     C_H_O, C_H_R, C_H_TA),
]:
    sub = dca_df[dca_df["Version"] == version]
    if len(sub) == 0:
        continue
    ax.plot(sub["threshold"], sub["NB_original"],    color=c_o, lw=1.1, ls="--", label=label_o)
    ax.plot(sub["threshold"], sub["NB_recalibrated"], color=c_r, lw=1.6,          label=label_r)
    ax.plot(sub["threshold"], sub["NB_treat_all"],    color=c_ta, lw=0.9, ls=":",  label=label_ta)
    prev_max = max(prev_max, sub["NB_treat_all"].iloc[0] / (1 - 0.01))

ax.axhline(0, color="#444444", lw=0.8, ls=":", alpha=0.7)
ax.axvspan(0.10, 0.30, color="#FFF3B0", alpha=0.35, zorder=0)
ax.set_xlim([0, 0.5])
ax.set_ylim([-0.05, max(0.25, prev_max * 1.35)])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.grid(True, alpha=0.25)
ax.text(0.20, max(0.25, prev_max * 1.35) * 0.96,
        "Exploratory threshold range (10%–30%)",
        ha="center", va="top", fontsize=8.5, color="#666666", style="italic")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
          frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
          fontsize=8.5, ncol=1, borderpad=0.6)

plt.tight_layout()
out_s8 = os.path.join(FIG_DIR_CHEST, "FigureS8_eicu_DCA.pdf")
plt.savefig(out_s8, dpi=1200, bbox_inches="tight")
plt.close()
print(f"  已保存: {out_s8}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)