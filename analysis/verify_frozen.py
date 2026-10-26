# -*- coding: utf-8 -*-
"""
verify_frozen.py
从 results/FROZEN_v1/ 重算数字字典中的键，与字典值比对。
输出：
  verify_report.md      人读的对照表
  verify_report.csv     机器读的对照表
用法：python verify_frozen.py
"""
import os
import math
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(BASE, "results", "FROZEN_v1")

# ---------- 字典：键 -> (字典值, 容差) ----------
# 容差按数字精度设定：4 位小数的键容差 0.0005，2 位小数的键容差 0.05，整数容差 0。
EXPECTED = {
    # 开发队列
    "dev_n": (2028, 0),
    "dev_events": (226, 0),
    "dev_event_rate": (11.1, 0.05),
    "train_n": (1420, 0),
    "train_events": (154, 0),
    "intval_n": (608, 0),
    "intval_events": (72, 0),
    # 内部验证
    "intval_auc": (0.7237, 0.0005),
    "intval_auc_lo": (0.649, 0.0005),
    "intval_auc_hi": (0.770, 0.0005),
    "intval_brier": (0.099, 0.0005),
    "intval_slope_mean": (0.7814, 0.0005),
    "intval_intercept_mean": (-0.3802, 0.0005),
    "intval_100split_median": (0.766, 0.0005),
    "train_auc": (0.771, 0.0005),
    "train_brier": (0.087, 0.0005),
    # MIMIC-IV BE
    "mimic_n": (434, 0),
    "mimic_events": (88, 0),
    "mimic_event_rate": (20.3, 0.05),
    "mimic_auc": (0.622, 0.0005),
    "mimic_auc_lo": (0.553, 0.0005),
    "mimic_auc_hi": (0.682, 0.0005),
    "mimic_brier": (0.166, 0.0005),
    "mimic_brier_lo": (0.141, 0.0005),
    "mimic_brier_hi": (0.194, 0.0005),
    "mimic_slope": (0.420, 0.0005),
    "mimic_slope_lo": (0.151, 0.0005),
    "mimic_slope_hi": (0.661, 0.0005),
    "mimic_intercept": (-0.617, 0.0005),
    "mimic_intercept_lo": (-1.149, 0.0005),
    "mimic_intercept_hi": (-0.142, 0.0005),
    "mimic_recal_a": (0.4196, 0.00005),
    "mimic_recal_b": (-0.6172, 0.00005),
    "mimic_brier_recal": (0.158, 0.0005),
    "mimic_brier_change_pct": (-4.8, 0.05),
    "mimic_oob_brier": (0.159, 0.0005),
    "mimic_oob_slope": (1.003, 0.0005),
    "mimic_oob_intercept": (-0.030, 0.0005),
    # MIMIC-IV HCO3
    "mimic_hco3_n": (684, 0),
    "mimic_hco3_events": (109, 0),
    "mimic_hco3_auc": (0.642, 0.0005),
    "mimic_hco3_auc_lo": (0.588, 0.0005),
    "mimic_hco3_auc_hi": (0.697, 0.0005),
    "mimic_hco3_brier": (0.137, 0.0005),
    "mimic_hco3_brier_lo": (0.119, 0.0005),
    "mimic_hco3_brier_hi": (0.156, 0.0005),
    "mimic_hco3_slope": (0.486, 0.0005),
    "mimic_hco3_slope_lo": (0.294, 0.0005),
    "mimic_hco3_slope_hi": (0.709, 0.0005),
    "mimic_hco3_intercept": (-0.809, 0.0005),
    "mimic_hco3_intercept_lo": (-1.213, 0.0005),
    "mimic_hco3_intercept_hi": (-0.400, 0.0005),
    "mimic_hco3_recal_a": (0.4856, 0.00005),
    "mimic_hco3_recal_b": (-0.8090, 0.00005),
    "mimic_hco3_brier_recal": (0.131, 0.0005),
    "mimic_hco3_brier_change_pct": (-4.5, 0.05),
    "mimic_hco3_oob_brier": (0.131, 0.0005),
    "mimic_hco3_oob_slope": (1.0275, 0.0005),
    "mimic_hco3_oob_intercept": (0.0329, 0.0005),
    # eICU-CRD BE
    "eicu_full_n": (5094, 0),
    "eicu_be_n": (1505, 0),
    "eicu_be_events": (368, 0),
    "eicu_be_event_rate": (24.5, 0.05),
    "eicu_be_auc": (0.675, 0.0005),
    "eicu_be_auc_lo": (0.642, 0.0005),
    "eicu_be_auc_hi": (0.707, 0.0005),
    "eicu_be_brier": (0.176, 0.0005),
    "eicu_be_brier_lo": (0.165, 0.0005),
    "eicu_be_brier_hi": (0.188, 0.0005),
    "eicu_be_slope": (0.617, 0.0005),
    "eicu_be_intercept": (-0.484, 0.0005),
    "eicu_be_recal_a": (0.6171, 0.0005),
    "eicu_be_recal_b": (-0.4841, 0.0005),
    "eicu_hco3_recal_a": (0.6420, 0.0005),
    "eicu_hco3_recal_b": (-0.6126, 0.0005),
    "eicu_be_brier_recal": (0.171, 0.0005),
    "eicu_be_brier_change_pct": (-2.8, 0.05),
    "eicu_be_oob_brier": (0.171, 0.0005),
    "eicu_be_oob_slope": (0.999, 0.0005),
    "eicu_be_oob_intercept": (0.000, 0.0005),
    # eICU-CRD HCO3
    "eicu_hco3_n": (2434, 0),
    "eicu_hco3_events": (531, 0),
    "eicu_hco3_event_rate": (21.8, 0.05),
    "eicu_hco3_auc": (0.676, 0.0005),
    "eicu_hco3_auc_lo": (0.649, 0.0005),
    "eicu_hco3_auc_hi": (0.702, 0.0005),
    "eicu_hco3_brier": (0.165, 0.0005),
    "eicu_hco3_brier_lo": (0.157, 0.0005),
    "eicu_hco3_brier_hi": (0.174, 0.0005),
    "eicu_hco3_slope": (0.642, 0.0005),
    "eicu_hco3_intercept": (-0.613, 0.0005),
    "eicu_hco3_brier_recal": (0.158, 0.0005),
    "eicu_hco3_brier_change_pct": (-4.2, 0.05),
    "eicu_hco3_oob_brier": (0.159, 0.0005),
    "eicu_hco3_oob_slope": (1.004, 0.0005),
    "eicu_hco3_oob_intercept": (-0.002, 0.0005),
    # 覆盖率
    "mimic_cov_alb": (27.9, 0.05),
    "mimic_cov_bun": (93.7, 0.05),
    "mimic_cov_be": (49.0, 0.05),
    "mimic_cov_hco3": (93.7, 0.05),
    "eicu_cov_alb": (49.9, 0.05),
    "eicu_cov_bun": (93.9, 0.05),
    "eicu_cov_be": (55.2, 0.05),
    "eicu_cov_hco3": (91.0, 0.05),
    # ---- Table 5 OOB（2026-10-20 新增；summary_v2 与 eicu_summary 精确中位数与 IQR）----
    "mimic_oob_brier_median": (0.1593, 0.0005),
    "mimic_oob_brier_q25": (0.1492, 0.0005),
    "mimic_oob_brier_q75": (0.1694, 0.0005),
    "mimic_oob_slope_median": (1.0027, 0.0005),
    "mimic_oob_slope_q25": (0.6203, 0.0005),
    "mimic_oob_slope_q75": (1.5554, 0.0005),
    "mimic_oob_intercept_median": (-0.0302, 0.0005),
    "mimic_oob_intercept_q25": (-0.5243, 0.0005),
    "mimic_oob_intercept_q75": (0.7827, 0.0005),
    "mimic_hco3_oob_brier_median": (0.1321, 0.0005),
    "mimic_hco3_oob_brier_q25": (0.1240, 0.0005),
    "mimic_hco3_oob_brier_q75": (0.1402, 0.0005),
    "mimic_hco3_oob_slope_median": (1.0275, 0.0005),
    "mimic_hco3_oob_slope_q25": (0.7399, 0.0005),
    "mimic_hco3_oob_slope_q75": (1.4017, 0.0005),
    "mimic_hco3_oob_intercept_median": (0.0329, 0.0005),
    "mimic_hco3_oob_intercept_q25": (-0.4308, 0.0005),
    "mimic_hco3_oob_intercept_q75": (0.6337, 0.0005),
    "eicu_be_oob_brier_median": (0.171912, 0.0005),
    "eicu_be_oob_brier_q25": (0.166962, 0.0005),
    "eicu_be_oob_brier_q75": (0.176642, 0.0005),
    "eicu_be_oob_slope_median": (1.000885, 0.0005),
    "eicu_be_oob_slope_q25": (0.846834, 0.0005),
    "eicu_be_oob_slope_q75": (1.151568, 0.0005),
    "eicu_be_oob_intercept_median": (-0.001034, 0.0005),
    "eicu_be_oob_intercept_q25": (-0.176739, 0.0005),
    "eicu_be_oob_intercept_q75": (0.184719, 0.0005),
    "eicu_hco3_oob_brier_median": (0.158395, 0.0005),
    "eicu_hco3_oob_brier_q25": (0.154985, 0.0005),
    "eicu_hco3_oob_brier_q75": (0.162295, 0.0005),
    "eicu_hco3_oob_slope_median": (0.996253, 0.0005),
    "eicu_hco3_oob_slope_q25": (0.875993, 0.0005),
    "eicu_hco3_oob_slope_q75": (1.129930, 0.0005),
    "eicu_hco3_oob_intercept_median": (-0.012556, 0.0005),
    "eicu_hco3_oob_intercept_q25": (-0.166171, 0.0005),
    "eicu_hco3_oob_intercept_q75": (0.167044, 0.0005),
}

# ---------- 读文件 ----------
def load(name):
    path = os.path.join(FROZEN, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)


def get_scalar(df, col, prefer="value"):
    """从 metric/value 两列的 summary CSV 里取一个数。"""
    if df is None or df.empty:
        return None
    if prefer in df.columns and "metric" in df.columns:
        hit = df[df["metric"].astype(str) == col]
        if len(hit) > 0:
            return float(hit["value"].iloc[0])
    if col in df.columns:
        return float(df[col].iloc[0])
    return None


# ---------- 重算 ----------
results = []   # (key, expected, recomputed, status, source)


def record(key, recomputed, source):
    exp, tol = EXPECTED[key]
    if recomputed is None:
        status = "文件缺失"
        rec_str = "—"
    else:
        diff = abs(float(recomputed) - float(exp))
        status = "一致" if diff <= tol else f"不一致（差 {diff:.4f}）"
        rec_str = f"{recomputed:.4f}" if isinstance(recomputed, float) else str(recomputed)
    results.append((key, exp, rec_str, status, source))


# 1) 开发/内部验证：summary_for_manuscript.csv
sm = load("summary_for_manuscript.csv")
if sm is not None:
    # 兼容两种格式：metric/value 两列，或一行一指标
    metric_col = "metric" if "metric" in sm.columns else None
    def sm_get(name):
        return get_scalar(sm, name)
    for key, col in [
        ("dev_n", "dev_n"), ("dev_events", "dev_events"),
        ("train_n", "train_n"), ("train_events", "train_events"),
        ("intval_n", "intval_n"), ("intval_events", "intval_events"),
        ("train_auc", "train_auc"), ("train_brier", "train_brier"),
        ("intval_auc", "intval_auc"), ("intval_brier", "intval_brier"),
    ]:
        v = sm_get(col)
        if v is not None:
            record(key, v, "summary_for_manuscript.csv")
else:
    for key in ["dev_n", "dev_events", "train_n", "train_events",
                "intval_n", "intval_events", "train_auc", "train_brier",
                "intval_auc", "intval_brier"]:
        record(key, None, "summary_for_manuscript.csv")

# 2) 内部验证 50 套：internal_validation_50imp.csv
iv = load("internal_validation_50imp.csv")
if iv is not None:
    if "auc" in iv.columns:
        record("intval_auc", iv["auc"].mean(), "internal_validation_50imp.csv (mean)")
    elif "Metric" in iv.columns and "Mean" in iv.columns:
        _m = iv.loc[iv["Metric"] == "AUC", "Mean"]
        record("intval_auc", float(_m.iloc[0]) if len(_m) else None,
               "internal_validation_50imp.csv (Metric=AUC, Mean)")
    if "brier" in iv.columns:
        record("intval_brier", iv["brier"].mean(), "internal_validation_50imp.csv (mean)")
    if "cal_slope" in iv.columns:
        record("intval_slope_mean", iv["cal_slope"].mean(), "internal_validation_50imp.csv (mean)")
    if "cal_intercept" in iv.columns:
        record("intval_intercept_mean", iv["cal_intercept"].mean(), "internal_validation_50imp.csv (mean)")
else:
    for key in ["intval_slope_mean", "intval_intercept_mean"]:
        record(key, None, "internal_validation_50imp.csv")

# 3) 100 次划分：auc_100splits.csv
sp = load("auc_100splits.csv")
if sp is not None and "AUC" in sp.columns:
    record("intval_100split_median", sp["AUC"].median(), "auc_100splits.csv (median)")
elif sp is not None and "auc" in sp.columns:
    record("intval_100split_median", sp["auc"].median(), "auc_100splits.csv (median)")
else:
    record("intval_100split_median", None, "auc_100splits.csv")

# 4) MIMIC-IV HCO3 敏感性分析：external_hco3_sensitivity_v2.csv
h = load("external_hco3_sensitivity_v2.csv")
if h is not None:
    mapping = {
        "mimic_hco3_n": "n", "mimic_hco3_events": "events",
        "mimic_hco3_auc": "AUC", "mimic_hco3_auc_lo": "AUC_lo", "mimic_hco3_auc_hi": "AUC_hi",
        "mimic_hco3_brier": "Brier", "mimic_hco3_brier_lo": "Brier_lo", "mimic_hco3_brier_hi": "Brier_hi",
        "mimic_hco3_slope": "cal_slope", "mimic_hco3_slope_lo": "cal_slope_lo", "mimic_hco3_slope_hi": "cal_slope_hi",
        "mimic_hco3_intercept": "cal_intercept", "mimic_hco3_intercept_lo": "cal_intercept_lo", "mimic_hco3_intercept_hi": "cal_intercept_hi",
        "mimic_hco3_recal_a": "recal_a", "mimic_hco3_recal_b": "recal_b",
        "mimic_hco3_brier_recal": "Brier_recal",
        "mimic_hco3_brier_change_pct": "Brier_recal_change_pct",
        "mimic_hco3_oob_brier": "OOB_Brier_median",

    }
    for key, col in mapping.items():
        v = get_scalar(h, col)
        record(key, v, "external_hco3_sensitivity_v2.csv")
else:
 
        record(key, None, "external_hco3_sensitivity_v2.csv")

# 5) MIMIC-IV BE：external_predictions.csv（重算 AUC/Brier/slope/intercept 不可行时跳过）
ep = load("external_predictions.csv")
if ep is not None:
    # 若 CSV 有 outcome 和 pred 列，重算 AUC 与 Brier
    if "outcome" in ep.columns and "pred" in ep.columns:
        y = ep["outcome"].astype(float).values
        p = ep["pred"].astype(float).values
        record("mimic_n", len(ep), "external_predictions.csv")
        record("mimic_events", int(y.sum()), "external_predictions.csv")
        # AUC
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            record("mimic_auc", roc_auc_score(y, p), "external_predictions.csv (recomputed)")
            record("mimic_brier", brier_score_loss(y, p), "external_predictions.csv (recomputed)")
        except Exception:
            pass
else:
    record("mimic_n", None, "external_predictions.csv")
    record("mimic_events", None, "external_predictions.csv")

# 6) 重校准汇总：recal_summary_dual.csv（含 MIMIC BE 与 HCO3）
rs = load("recal_summary_dual.csv")
if rs is not None:
    # 假设列：model, stage, auc, brier, cal_slope, cal_intercept
    # 尝试取 MIMIC BE original / recal
    def pick(model_key, stage_key, col):
        m = rs
        for k, v in [("model", model_key), ("stage", stage_key)]:
            if k in m.columns:
                m = m[m[k].astype(str).str.contains(v, case=False, na=False)]
        if len(m) > 0 and col in m.columns:
            return float(m[col].iloc[0])
        return None
    v = pick("BE", "original", "cal_slope")
    if v is not None:
        record("mimic_slope", v, "recal_summary_dual.csv (BE original)")
    v = pick("BE", "original", "cal_intercept")
    if v is not None:
        record("mimic_intercept", v, "recal_summary_dual.csv (BE original)")
    v = pick("BE", "recal", "brier")
    if v is not None:
        record("mimic_brier_recal", v, "recal_summary_dual.csv (BE recal)")
    v = pick("HCO3", "recal", "brier")
    if v is not None:
        record("mimic_hco3_brier_recal", v, "recal_summary_dual.csv (HCO3 recal)")

# 7) Bootstrap 汇总：recal_bootstrap_summary_v2.csv（MIMIC-IV，含 IQR）
#    旧三键 mimic_oob_brier/slope/intercept 改读 v2 的 median 列，避免旧窄表列名不匹配
bs2 = load("recal_bootstrap_summary_v2.csv")
if bs2 is not None:
    def bs2_row(model):
        d = bs2[bs2["Version"].astype(str).str.upper() == model]
        return None if d.empty else d.iloc[0]

    r_be = bs2_row("BE")
    r_h  = bs2_row("HCO3")
    record("mimic_hco3_oob_slope",     None if r_h is None else float(r_h["slope_oob_median"]),     "recal_bootstrap_summary_v2.csv (HCO3 median)")
    record("mimic_hco3_oob_intercept", None if r_h is None else float(r_h["intercept_oob_median"]), "recal_bootstrap_summary_v2.csv (HCO3 median)")

    # 旧三键（BE median）——与 v2 的 Brier_oob_median 对齐
    if r_be is not None:
        record("mimic_oob_brier",     float(r_be["Brier_oob_median"]), "recal_bootstrap_summary_v2.csv (BE median)")
        record("mimic_oob_slope",     float(r_be["slope_oob_median"]), "recal_bootstrap_summary_v2.csv (BE median)")
        record("mimic_oob_intercept", float(r_be["intercept_oob_median"]), "recal_bootstrap_summary_v2.csv (BE median)")
    else:
        for k in ["mimic_oob_brier", "mimic_oob_slope", "mimic_oob_intercept"]:
            record(k, None, "recal_bootstrap_summary_v2.csv")

    # 36 新键：MIMIC BE + HCO3
    for prefix, r in [("mimic", r_be), ("mimic_hco3", r_h)]:
        if r is None:
            for suffix in ["brier_median","brier_q25","brier_q75",
                           "slope_median","slope_q25","slope_q75",
                           "intercept_median","intercept_q25","intercept_q75"]:
                record(f"{prefix}_oob_{suffix}", None, "recal_bootstrap_summary_v2.csv")
            continue
        record(f"{prefix}_oob_brier_median",     float(r["Brier_oob_median"]),     "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_brier_q25",        float(r["Brier_oob_q25"]),        "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_brier_q75",        float(r["Brier_oob_q75"]),        "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_slope_median",     float(r["slope_oob_median"]),     "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_slope_q25",        float(r["slope_oob_q25"]),        "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_slope_q75",        float(r["slope_oob_q75"]),        "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_intercept_median", float(r["intercept_oob_median"]), "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_intercept_q25",    float(r["intercept_oob_q25"]),    "recal_bootstrap_summary_v2.csv")
        record(f"{prefix}_oob_intercept_q75",    float(r["intercept_oob_q75"]),    "recal_bootstrap_summary_v2.csv")
else:
    for k in ["mimic_oob_brier","mimic_oob_slope","mimic_oob_intercept"]:
        record(k, None, "recal_bootstrap_summary_v2.csv")
    for prefix in ["mimic","mimic_hco3"]:
        for suffix in ["brier_median","brier_q25","brier_q75",
                       "slope_median","slope_q25","slope_q75",
                       "intercept_median","intercept_q25","intercept_q75"]:
            record(f"{prefix}_oob_{suffix}", None, "recal_bootstrap_summary_v2.csv")

# 7b) eICU-CRD OOB（含 IQR）：recal_bootstrap_eicu_summary.csv
be2 = load("recal_bootstrap_eicu_summary.csv")
if be2 is not None:
    def be2_row(model):
        d = be2[be2["Version"].astype(str).str.upper().str.replace("₃", "3") == model]
        return None if d.empty else d.iloc[0]

    for prefix, model in [("eicu_be", "BE"), ("eicu_hco3", "HCO3")]:
        r = be2_row(model)
        if r is None:
            for suffix in ["brier_median","brier_q25","brier_q75",
                           "slope_median","slope_q25","slope_q75",
                           "intercept_median","intercept_q25","intercept_q75"]:
                record(f"{prefix}_oob_{suffix}", None, "recal_bootstrap_eicu_summary.csv")
            continue
        record(f"{prefix}_oob_brier_median",     float(r["Brier_OOB_median"]),     "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_brier_q25",        float(r["Brier_OOB_q25"]),        "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_brier_q75",        float(r["Brier_OOB_q75"]),        "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_slope_median",     float(r["slope_OOB_median"]),     "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_slope_q25",        float(r["slope_OOB_q25"]),        "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_slope_q75",        float(r["slope_OOB_q75"]),        "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_intercept_median", float(r["intercept_OOB_median"]), "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_intercept_q25",    float(r["intercept_OOB_q25"]),    "recal_bootstrap_eicu_summary.csv")
        record(f"{prefix}_oob_intercept_q75",    float(r["intercept_OOB_q75"]),    "recal_bootstrap_eicu_summary.csv")
else:
    for prefix in ["eicu_be","eicu_hco3"]:
        for suffix in ["brier_median","brier_q25","brier_q75",
                       "slope_median","slope_q25","slope_q75",
                       "intercept_median","intercept_q25","intercept_q75"]:
            record(f"{prefix}_oob_{suffix}", None, "recal_bootstrap_eicu_summary.csv")

# 8) eICU BE / HCO3 summary
for fname, prefix in [("eicu_be_summary.csv", "eicu_be"),
                      ("eicu_hco3_summary.csv", "eicu_hco3")]:
    df = load(fname)
    if df is None:
        continue
    mapping = {
        f"{prefix}_n": "n", f"{prefix}_events": "events",
        f"{prefix}_auc": "AUC", f"{prefix}_auc_lo": "AUC_lo", f"{prefix}_auc_hi": "AUC_hi",
        f"{prefix}_brier": "Brier", f"{prefix}_brier_lo": "Brier_lo", f"{prefix}_brier_hi": "Brier_hi",
        f"{prefix}_slope": "cal_slope", f"{prefix}_intercept": "cal_intercept",
        f"{prefix}_brier_recal": "Brier_recal",
        f"{prefix}_oob_brier": "OOB_Brier_median",
        f"{prefix}_oob_slope": "OOB_slope_median",
        f"{prefix}_oob_intercept": "OOB_intercept_median",
    }
    for key, col in mapping.items():
        if key not in EXPECTED:
            continue
        v = get_scalar(df, col)
        record(key, v, fname)
    # Brier 变化百分比：用 Brier 和 Brier_recal 现算
    brier = get_scalar(df, "Brier")
    brier_recal = get_scalar(df, "Brier_recal")
    if brier is not None and brier_recal is not None and brier != 0:
        pct = (brier_recal - brier) / brier * 100
        record(f"{prefix}_brier_change_pct", pct, f"{fname} (computed)")
    else:
        record(f"{prefix}_brier_change_pct", None, fname)

# 9) 覆盖率：eicu_full_cohort.csv
fc = load("eicu_full_cohort.csv")
if fc is not None:
    record("eicu_full_n", len(fc), "eicu_full_cohort.csv")
    for key, col in [("eicu_cov_alb", "alb"), ("eicu_cov_bun", "bun"),
                     ("eicu_cov_be", "be"), ("eicu_cov_hco3", "hco3")]:
        if col in fc.columns:
            record(key, fc[col].notna().mean() * 100, "eicu_full_cohort.csv")

# 10) DCA：dca_results_dual.csv 与 eicu_dca.csv
dca_m = load("dca_results_dual.csv")
if dca_m is not None:
    # 假设列：version, threshold, NB_original, NB_recalibrated, NB_treat_all
    def dca_val(version, thr, col):
        m = dca_m
        if "version" in m.columns:
            m = m[m["version"].astype(str).str.upper() == version.upper()]
        m = m[np.isclose(m["threshold"].astype(float), thr)]
        if len(m) > 0 and col in m.columns:
            return float(m[col].iloc[0])
        return None
    for thr in [0.10, 0.15, 0.20, 0.30]:
        for col, key_suffix in [("NB_original", "orig"), ("NB_recalibrated", "recal"), ("NB_treat_all", "ta")]:
            for ver, ver_key in [("BE", "be"), ("HCO3", "hco3")]:
                key = f"mimic_dca_{ver_key}_{key_suffix}_{int(thr*100)}"
                if key in EXPECTED:
                    record(key, dca_val(ver, thr, col), "dca_results_dual.csv")

dca_e = load("eicu_dca.csv")
if dca_e is not None:
    def dca_e_val(version, thr, col):
        m = dca_e
        if "Version" in m.columns:
            m = m[m["Version"].astype(str).str.upper() == version.upper()]
        m = m[np.isclose(m["threshold"].astype(float), thr)]
        if len(m) > 0 and col in m.columns:
            return float(m[col].iloc[0])
        return None
    for thr in [0.10, 0.15, 0.20, 0.30]:
        for col, key_suffix in [("NB_original", "orig"), ("NB_recalibrated", "recal"), ("NB_treat_all", "ta")]:
            for ver, ver_key in [("BE", "be"), ("HCO3", "hco3")]:
                key = f"eicu_dca_{ver_key}_{key_suffix}_{int(thr*100)}"
                if key in EXPECTED:
                    record(key, dca_e_val(ver, thr, col), "eicu_dca.csv")

# ---------- 输出 ----------
report = pd.DataFrame(results, columns=["Key", "Expected", "Recomputed", "Status", "Source"])
report.to_csv(os.path.join(BASE, "verify_report.csv"), index=False)

lines = ["# verify_frozen 报告\n",
         f"冻结目录：{FROZEN}\n",
         f"总键数：{len(report)}\n",
         f"一致：{(report['Status'] == '一致').sum()}\n",
         f"不一致：{report['Status'].astype(str).str.startswith('不一致').sum()}\n",
         f"文件缺失/无法重算：{(report['Status'] == '文件缺失').sum()}\n",
         "\n## 逐项结果\n",
         "| Key | Expected | Recomputed | Status | Source |",
         "|---|---|---|---|---|"]
for _, r in report.iterrows():
    lines.append(f"| {r['Key']} | {r['Expected']} | {r['Recomputed']} | {r['Status']} | {r['Source']} |")
with open(os.path.join(BASE, "verify_report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"总键数 {len(report)}，一致 {(report['Status']=='一致').sum()}，"
      f"不一致 {report['Status'].astype(str).str.startswith('不一致').sum()}，"
      f"文件缺失 {(report['Status']=='文件缺失').sum()}")
print("报告：verify_report.md / verify_report.csv")