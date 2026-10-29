# -*- coding: utf-8 -*-
"""
eICU 外部验证：BE 版 + HCO3- 版 (v4.4，方案 B 口径，含 OOB IQR)
- 时间起点：ICU 入科；预测时点 = ICU 入科后 24h；结局 = ICU 24h 后至出院前院内死亡
- 该口径与开发队列/MIMIC-IV（住院入院为起点）不同，结果不与 MIMIC 合并、不直接比较
- 统一 Figure S6/S7 的图例位置与 Figure 2/S1/S2 完全一致
- 统一斜率/截距显示方式（图内左上角标注）
- AUC/Brier 95% CI：B=2000 百分位法
- 重校准：logit(p_new) = a*logit(p) + b，a/b 在 eICU 上估计
- Bootstrap OOB：N=1000，seed=123；输出 median 与 q25/q75
- DCA：10%-30% 探索性区间
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

# ---- CHEST style ----
sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] 未找到 chest_style: {_e}")

RNG = 123
N_BOOT = 1000
N_CI  = 2000

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

def bootstrap_ci_auc_brier(y, p, n_boot=N_CI, seed=RNG):
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs, briers = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        aucs.append(roc_auc_score(yb, p[idx]))
        briers.append(brier_score_loss(yb, p[idx]))
    return (np.percentile(aucs, [2.5, 97.5]),
            np.percentile(briers, [2.5, 97.5]))

def bootstrap_oob_recal(y, p, n_boot=N_BOOT, seed=RNG):
    rng = np.random.RandomState(seed)
    n = len(y)
    logit_p = to_logit(p)
    rows = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 20 or y[oob].sum() < 3 or (len(oob) - y[oob].sum()) < 3:
            continue
        m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        m.fit(logit_p[idx].reshape(-1, 1), y[idx])
        a = float(m.coef_[0, 0]); b = float(m.intercept_[0])
        lp_oob = a * logit_p[oob] + b
        p_oob = 1.0 / (1.0 + np.exp(-lp_oob))
        try:
            br = brier_score_loss(y[oob], p_oob)
            sl, ic = calib_slope_intercept(y[oob], lp_oob)
        except Exception:
            continue
        rows.append({"Brier_oob": br, "slope_oob": sl, "intercept_oob": ic})
    df = pd.DataFrame(rows)
    return df

# ============================================================
# 冻结系数
# ============================================================
COEF = {
    "BE":    {"b0": 1.7825, "alb": -0.1227, "bun": 0.0343, "x": -0.0585},
    "HCO3":  {"b0": 3.1936, "alb": -0.1209, "bun": 0.0374, "x": -0.0604},
}

def predict_frozen(df, version):
    c = COEF[version]
    x_col = "BE" if version == "BE" else "HCO3"
    lp = c["b0"] + c["alb"]*df["ALB"].values + c["bun"]*df["BUN"].values + c["x"]*df[x_col].values
    return 1.0 / (1.0 + np.exp(-lp))

# ============================================================
# 主分析函数
# ============================================================
def run_version(version, csv_path, out_summary, out_pred, out_fig):
    x_col = "BE" if version == "BE" else "HCO3"
    label = "BE" if version == "BE" else r"HCO$_3^-$"
    color = "#1F77B4" if version == "BE" else "#B2182B"

    print("\n" + "=" * 60)
    print(f"eICU 外部验证 — {version} 版（方案 B：ICU 24h 口径）")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    y = df["outcome"].values.astype(int)
    p = predict_frozen(df, version)
    print(f"  n={len(y)}, events={int(y.sum())} ({y.mean()*100:.1f}%)")

    # 主指标
    auc = roc_auc_score(y, p)
    brier = brier_score_loss(y, p)
    lp = to_logit(p)
    slope, intercept = calib_slope_intercept(y, lp)
    (auc_lo, auc_hi), (br_lo, br_hi) = bootstrap_ci_auc_brier(y, p)

    # 重校准
    lp_r = slope * lp + intercept
    p_r = 1.0 / (1.0 + np.exp(-lp_r))
    brier_r = brier_score_loss(y, p_r)
    sl_r, ic_r = calib_slope_intercept(y, lp_r)

    # OOB（median + IQR）
    oob = bootstrap_oob_recal(y, p)
    if len(oob):
        oob_brier   = oob["Brier_oob"].median()
        oob_slope   = oob["slope_oob"].median()
        oob_ic      = oob["intercept_oob"].median()
        oob_brier_q25 = oob["Brier_oob"].quantile(0.25)
        oob_brier_q75 = oob["Brier_oob"].quantile(0.75)
        oob_slope_q25 = oob["slope_oob"].quantile(0.25)
        oob_slope_q75 = oob["slope_oob"].quantile(0.75)
        oob_ic_q25    = oob["intercept_oob"].quantile(0.25)
        oob_ic_q75    = oob["intercept_oob"].quantile(0.75)
    else:
        oob_brier = oob_slope = oob_ic = np.nan
        oob_brier_q25 = oob_brier_q75 = np.nan
        oob_slope_q25 = oob_slope_q75 = np.nan
        oob_ic_q25 = oob_ic_q75 = np.nan

    print(f"  AUC={auc:.4f} (95% CI {auc_lo:.4f}-{auc_hi:.4f})")
    print(f"  Brier={brier:.4f} (95% CI {br_lo:.4f}-{br_hi:.4f})")
    print(f"  Calibration slope={slope:.4f}, intercept={intercept:.4f}")
    print(f"  Recalibrated: Brier={brier_r:.4f} (Δ={100*(brier_r-brier)/brier:+.1f}%), slope={sl_r:.4f}, intercept={ic_r:.4f}")
    print(f"  OOB: Brier={oob_brier:.4f} [{oob_brier_q25:.4f}-{oob_brier_q75:.4f}], "
          f"slope={oob_slope:.4f} [{oob_slope_q25:.4f}-{oob_slope_q75:.4f}], "
          f"intercept={oob_ic:.4f} [{oob_ic_q25:.4f}-{oob_ic_q75:.4f}]")

    # summary CSV
    summary = pd.DataFrame([
        {"Version": version, "Stage": "Original", "n": len(y), "events": int(y.sum()),
         "AUC": auc, "AUC_lo": auc_lo, "AUC_hi": auc_hi,
         "Brier": brier, "Brier_lo": br_lo, "Brier_hi": br_hi,
         "cal_slope": slope, "cal_intercept": intercept,
         "recal_a": slope, "recal_b": intercept,
         "Brier_recal": brier_r, "slope_recal": sl_r, "intercept_recal": ic_r,
         "OOB_Brier_median": oob_brier,
         "OOB_Brier_q25": oob_brier_q25, "OOB_Brier_q75": oob_brier_q75,
         "OOB_slope_median": oob_slope,
         "OOB_slope_q25": oob_slope_q25, "OOB_slope_q75": oob_slope_q75,
         "OOB_intercept_median": oob_ic,
         "OOB_intercept_q25": oob_ic_q25, "OOB_intercept_q75": oob_ic_q75},
    ])
    summary.to_csv(out_summary, index=False, float_format="%.6f")
    print(f"  已保存: {out_summary}")

    # 逐例预测
    pred_df = pd.DataFrame({"outcome": y, "pred_prob": p,
                            "pred_prob_recal": p_r})
    if "patientunitstayid" in df.columns:
        pred_df.insert(0, "patientunitstayid", df["patientunitstayid"].values)
    pred_df.to_csv(out_pred, index=False, float_format="%.6f")
    print(f"  已保存: {out_pred}")

    # ============================================================
    # 图：ROC + calibration（与 Figure 2/S1/S2 完全统一）
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    # A: ROC
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

    # B: Calibration
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
# 跑 BE 版
# ============================================================
df_be, y_be, p_be, sum_be = run_version(
    "BE",
    os.path.join(RESULTS_DIR, "eicu_wide_be.csv"),
    os.path.join(RESULTS_DIR, "eicu_be_summary.csv"),
    os.path.join(RESULTS_DIR, "eicu_predictions_be.csv"),
    os.path.join(FIG_DIR_CHEST, "FigureS6_eicu_ROC_calibration.pdf"),
)

# ============================================================
# 跑 HCO3 版
# ============================================================
df_hco3, y_hco3, p_hco3, sum_hco3 = run_version(
    "HCO3",
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
dca_df.to_csv(os.path.join(RESULTS_DIR, "eicu_dca.csv"), index=False, float_format="%.4f")
print(f"  已保存: results/eicu_dca.csv ({len(dca_df)} 行)")

print("\n  代表阈值净获益：")
print(f"  {'版本':<8}{'阈值':>7}{'原始':>12}{'重校准':>12}{'TreatAll':>12}")
for version in ["BE", "HCO3"]:
    sub = dca_df[dca_df["Version"] == version]
    for pt in [0.10, 0.15, 0.20, 0.30]:
        row = sub.iloc[(sub["threshold"] - pt).abs().argmin()]
        print(f"  {version:<8}{row['threshold']:>7.2f}"
              f"{row['NB_original']:>12.4f}{row['NB_recalibrated']:>12.4f}"
              f"{row['NB_treat_all']:>12.4f}")

# ============================================================
# Figure S8: DCA（与 Figure 3 统一风格）
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
ax.text(0.20, max(0.25, prev_max * 1.35) * 0.96, "Exploratory threshold range (10%–30%)",
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