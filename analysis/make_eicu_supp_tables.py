# -*- coding: utf-8 -*-
"""
make_eicu_supp_tables.py

从 eICU-CRD 宽表生成补充表：
  Table S7b  : 开发队列 vs eICU-CRD 的 SMD（BE 版与 HCO3 版）
  Table S11  : eICU-CRD 基线特征（BE 版与 HCO3 版）
  Table S12  : eICU-CRD 纳入 vs 排除（BE 版与 HCO3 版）
  Table S13  : eICU-CRD ALB 测量 vs 未测量
  Table S6_row: 供 Table S6 "This study" 行使用的汇总数字

输入（默认路径，可在 BASE 下调整）：
  results/eicu_wide_be.csv      n=1,505
  results/eicu_wide_hco3.csv    n=2,434
  results/eicu_full_cohort.csv  n=5,099（含 alb 缺失行）

输出：
  results/TableS7b_dev_vs_eicu.csv
  results/TableS11_eicu_baseline.csv
  results/TableS12_eicu_incl_vs_excl.csv
  results/TableS13_eicu_alb_measured_vs_not.csv
  results/TableS6_this_study_row.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")

# 列名映射：按实际 CSV 列名修改
COLS = {
    "age": "age",
    "male": "male",
    "alb": "alb",
    "bun": "bun",
    "be": "be",
    "hco3": "hco3",
    "outcome": "outcome",
}

ID = "patientunitstayid"
# 开发队列参考值（来自 Table 1 / Table S7a）
DEV = {
    "age_mean": 76.0, "age_sd": 12.0,
    "male_pct": 71.1,
    "alb_mean": 34.44, "alb_sd": 5.31,
    "bun_mean": 8.73, "bun_sd": 6.35,
    "be_mean": 4.22, "be_sd": 7.80,
    "hco3_mean": None, "hco3_sd": None,   # 开发队列 HCO3 版见 Table 1（若有则填）
    "mort_pct": 11.1,
    "n": 2028,
}

# eICU 全队列人数（用于 S12/S13 的分母）
EICU_FULL_N = 5094
EICU_BE_N = 1505
EICU_HCO3_N = 2434


# ---------- 工具函数 ----------
def smd_continuous(m1, sd1, n1, m2, sd2, n2):
    """连续变量 SMD（绝对值）。"""
    if any(pd.isna([m1, sd1, m2, sd2])):
        return np.nan
    pooled = np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return abs(m1 - m2) / pooled


def smd_binary(p1, n1, p2, n2):
    """二分类变量 SMD（绝对值），p 为比例（0–1）。"""
    if any(pd.isna([p1, p2])):
        return np.nan
    denom = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if denom == 0:
        return np.nan
    return abs(p1 - p2) / denom


def welch_p(x, y):
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return stats.ttest_ind(x, y, equal_var=False).pvalue


def chi2_p(a, b, c, d):
    """2x2 卡方（含连续性校正），a/b 为组1 死亡/存活，c/d 为组2。"""
    table = np.array([[a, b], [c, d]])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return np.nan
    return stats.chi2_contingency(table, correction=True).pvalue


def fmt_mean_sd(s):
    s = pd.Series(s).dropna()
    if len(s) == 0:
        return "—"
    return f"{s.mean():.1f} ({s.std(ddof=1):.1f})"


def fmt_n_pct(n, d):
    if d == 0:
        return "—"
    return f"{int(n)} ({100.0 * n / d:.1f})"


def summarize(df, cols):
    """返回一个 dict：连续变量 mean/sd/n，二分类 n/pct，结局 n/pct。"""
    out = {}
    for key, col in cols.items():
        if col not in df.columns:
            out[key] = {"mean": np.nan, "sd": np.nan, "n": 0, "pct": np.nan}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if key == "male" or key == "outcome":
            n_valid = s.notna().sum()
            n_pos = (s == 1).sum()
            out[key] = {
                "mean": np.nan, "sd": np.nan,
                "n": int(n_pos), "pct": 100.0 * n_pos / n_valid if n_valid else np.nan,
                "n_valid": int(n_valid),
            }
        else:
            out[key] = {
                "mean": s.mean(), "sd": s.std(ddof=1),
                "n": int(s.notna().sum()), "pct": np.nan,
                "n_valid": int(s.notna().sum()),
            }
    out["n_total"] = len(df)
    return out


# ---------- 读入 ----------
def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少输入文件：{path}")
    df = pd.read_csv(path, low_memory=False)

    rename_map = {
        "ALB": "alb", "BUN": "bun", "BE": "be", "HCO3": "hco3",
        "gender": "male", "Age": "age",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "male" in df.columns and df["male"].dtype == object:
        df["male"] = df["male"].astype(str).str.strip().str.lower().map(
            {"male": 1, "female": 0}
        )

    oc = "outcome"
    if oc in df.columns and df[oc].dtype == object:
        df[oc] = df[oc].astype(str).str.strip().str.lower().map(
            {"expired": 1, "dead": 1, "death": 1, "1": 1,
             "alive": 0, "survived": 0, "0": 0}
        )
    return df

df_be = load(os.path.join(RESULTS, "eicu_wide_be.csv"))
df_hco3 = load(os.path.join(RESULTS, "eicu_wide_hco3.csv"))
df_full = load(os.path.join(RESULTS, "eicu_full_cohort.csv"))

# 一致性检查
assert len(df_be) == EICU_BE_N, f"BE 版 n={len(df_be)}，预期 {EICU_BE_N}"
assert len(df_hco3) == EICU_HCO3_N, f"HCO3 版 n={len(df_hco3)}，预期 {EICU_HCO3_N}"
assert len(df_full) == EICU_FULL_N, f"全队列 n={len(df_full)}，预期 {EICU_FULL_N}"

# 全队列拆分：排除 = 全队列 - 纳入
# 用索引或唯一 ID 做差集；此处假设两个宽表都来自全队列，且有唯一 ID 列 'id'
ID = "id" if "id" in df_full.columns else None
if ID:
    be_ids = set(df_be[ID])
    hco3_ids = set(df_hco3[ID])
    df_be_excl = df_full[~df_full[ID].isin(be_ids)].copy()
    df_hco3_excl = df_full[~df_full[ID].isin(hco3_ids)].copy()
else:
    # 无 ID 时，用行数兜底：排除组 = 全队列中 ALB 或 BE/HCO3 缺失者
    be_col = COLS["be"]
    hco3_col = COLS["hco3"]
    df_be_excl = df_full[df_full[[COLS["alb"], COLS["bun"], be_col]].isna().any(axis=1)].copy()
    df_hco3_excl = df_full[df_full[[COLS["alb"], COLS["bun"], hco3_col]].isna().any(axis=1)].copy()


# ---------- Table S7b ----------
def build_s7b():
    rows = []
    summ_be = summarize(df_be, COLS)
    summ_h = summarize(df_hco3, COLS)

    def add_row(label, dev_mean=None, dev_sd=None, dev_pct=None, key=None, is_binary=False):
        if is_binary:
            smd_be = smd_binary(dev_pct / 100.0 if dev_pct is not None else np.nan,
                                DEV["n"],
                                summ_be[key]["pct"] / 100.0 if key in summ_be and not pd.isna(summ_be[key]["pct"]) else np.nan,
                                summ_be["n_total"])
            smd_h = smd_binary(dev_pct / 100.0 if dev_pct is not None else np.nan,
                               DEV["n"],
                               summ_h[key]["pct"] / 100.0 if key in summ_h and not pd.isna(summ_h[key]["pct"]) else np.nan,
                               summ_h["n_total"])
            be_val = f"{summ_be[key]['pct']:.1f}" if key in summ_be and not pd.isna(summ_be[key]["pct"]) else "—"
            h_val = f"{summ_h[key]['pct']:.1f}" if key in summ_h and not pd.isna(summ_h[key]["pct"]) else "—"
            dev_val = f"{dev_pct:.1f}" if dev_pct is not None else "—"
        else:
            smd_be = smd_continuous(dev_mean, dev_sd, DEV["n"],
                                    summ_be[key]["mean"], summ_be[key]["sd"], summ_be["n_total"])
            smd_h = smd_continuous(dev_mean, dev_sd, DEV["n"],
                                   summ_h[key]["mean"], summ_h[key]["sd"], summ_h["n_total"])
            be_val = (f"{summ_be[key]['mean']:.1f} ({summ_be[key]['sd']:.1f})"
                      if key in summ_be and not pd.isna(summ_be[key]["mean"]) else "—")
            h_val = (f"{summ_h[key]['mean']:.1f} ({summ_h[key]['sd']:.1f})"
                     if key in summ_h and not pd.isna(summ_h[key]["mean"]) else "—")
            dev_val = f"{dev_mean:.1f} ({dev_sd:.1f})" if dev_mean is not None else "—"
        rows.append({
            "Variable": label,
            "Development (n = 2,028)": dev_val,
            "eICU-CRD BE (n = 1,505)": be_val,
            "SMD (BE)": f"{smd_be:.2f}" if not pd.isna(smd_be) else "—",
            "eICU-CRD HCO3 (n = 2,434)": h_val,
            "SMD (HCO3)": f"{smd_h:.2f}" if not pd.isna(smd_h) else "—",
        })

    add_row("Age, years, mean (SD)", DEV["age_mean"], DEV["age_sd"], key="age")
    add_row("Male, %", dev_pct=DEV["male_pct"], key="male", is_binary=True)
    add_row("ALB, g/L, mean (SD)", DEV["alb_mean"], DEV["alb_sd"], key="alb")
    add_row("BUN, mmol/L, mean (SD)", DEV["bun_mean"], DEV["bun_sd"], key="bun")
    add_row("BE, mmol/L, mean (SD)", DEV["be_mean"], DEV["be_sd"], key="be")
    add_row("HCO3, mmol/L, mean (SD)", DEV["hco3_mean"], DEV["hco3_sd"], key="hco3")
    add_row("In-hospital mortality, %", dev_pct=DEV["mort_pct"], key="outcome", is_binary=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "TableS7b_dev_vs_eicu.csv"), index=False)
    return out


# ---------- Table S11 ----------
def build_s11():
    summ_be = summarize(df_be, COLS)
    summ_h = summarize(df_hco3, COLS)

    def mean_sd(s, k):
        if k not in s or pd.isna(s[k]["mean"]):
            return "—"
        return f"{s[k]['mean']:.1f} ({s[k]['sd']:.1f})"

    def n_pct(s, k):
        if k not in s or pd.isna(s[k]["pct"]):
            return "—"
        return f"{s[k]['n']} ({s[k]['pct']:.1f})"

    rows = [
        {"Variable": "Age, years, mean (SD)",
         "eICU-CRD BE (n = 1,505)": mean_sd(summ_be, "age"),
         "eICU-CRD HCO3 (n = 2,434)": mean_sd(summ_h, "age")},
        {"Variable": "Male, n (%)",
         "eICU-CRD BE (n = 1,505)": n_pct(summ_be, "male"),
         "eICU-CRD HCO3 (n = 2,434)": n_pct(summ_h, "male")},
        {"Variable": "ALB, g/L, mean (SD)",
         "eICU-CRD BE (n = 1,505)": mean_sd(summ_be, "alb"),
         "eICU-CRD HCO3 (n = 2,434)": mean_sd(summ_h, "alb")},
        {"Variable": "BUN, mmol/L, mean (SD)",
         "eICU-CRD BE (n = 1,505)": mean_sd(summ_be, "bun"),
         "eICU-CRD HCO3 (n = 2,434)": mean_sd(summ_h, "bun")},
        {"Variable": "BE, mmol/L, mean (SD)",
         "eICU-CRD BE (n = 1,505)": mean_sd(summ_be, "be"),
         "eICU-CRD HCO3 (n = 2,434)": "—"},
        {"Variable": "HCO3, mmol/L, mean (SD)",
         "eICU-CRD BE (n = 1,505)": "—",
         "eICU-CRD HCO3 (n = 2,434)": mean_sd(summ_h, "hco3")},
        {"Variable": "In-hospital death, n (%)",
         "eICU-CRD BE (n = 1,505)": n_pct(summ_be, "outcome"),
         "eICU-CRD HCO3 (n = 2,434)": n_pct(summ_h, "outcome")},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "TableS11_eicu_baseline.csv"), index=False)
    return out


# ---------- Table S12 ----------
def build_s12():
    rows = []

    def one_version(incl, excl, version_label, x_col):
        si = summarize(incl, COLS)
        se = summarize(excl, COLS)
        # 年龄
        rows.append({
            "Version": version_label, "Variable": "Age, years, mean (SD)",
            "Included": f"{si['age']['mean']:.1f} ({si['age']['sd']:.1f})" if not pd.isna(si['age']['mean']) else "—",
            "Excluded": f"{se['age']['mean']:.1f} ({se['age']['sd']:.1f})" if not pd.isna(se['age']['mean']) else "—",
            "SMD": f"{smd_continuous(si['age']['mean'], si['age']['sd'], si['n_total'], se['age']['mean'], se['age']['sd'], se['n_total']):.2f}",
            "P value": f"{welch_p(incl[COLS['age']], excl[COLS['age']]):.3f}",
        })
        # 男性
        rows.append({
            "Version": version_label, "Variable": "Male, n (%)",
            "Included": f"{si['male']['n']} ({si['male']['pct']:.1f})",
            "Excluded": f"{se['male']['n']} ({se['male']['pct']:.1f})",
            "SMD": f"{smd_binary(si['male']['pct']/100.0, si['n_total'], se['male']['pct']/100.0, se['n_total']):.2f}",
            "P value": f"{chi2_p(si['male']['n'], si['n_total']-si['male']['n'], se['male']['n'], se['n_total']-se['male']['n']):.3f}",
        })
        # BUN
        rows.append({
            "Version": version_label, "Variable": "BUN, mmol/L, mean (SD)",
            "Included": f"{si['bun']['mean']:.1f} ({si['bun']['sd']:.1f})" if not pd.isna(si['bun']['mean']) else "—",
            "Excluded": f"{se['bun']['mean']:.1f} ({se['bun']['sd']:.1f})" if not pd.isna(se['bun']['mean']) else "—",
            "SMD": f"{smd_continuous(si['bun']['mean'], si['bun']['sd'], si['n_total'], se['bun']['mean'], se['bun']['sd'], se['n_total']):.2f}",
            "P value": f"{welch_p(incl[COLS['bun']], excl[COLS['bun']]):.3f}",
        })
        # 死亡率
        rows.append({
            "Version": version_label, "Variable": "In-hospital mortality, n (%)",
            "Included": f"{si['outcome']['n']} ({si['outcome']['pct']:.1f})",
            "Excluded": f"{se['outcome']['n']} ({se['outcome']['pct']:.1f})",
            "SMD": f"{smd_binary(si['outcome']['pct']/100.0, si['n_total'], se['outcome']['pct']/100.0, se['n_total']):.2f}",
            "P value": f"{chi2_p(si['outcome']['n'], si['n_total']-si['outcome']['n'], se['outcome']['n'], se['n_total']-se['outcome']['n']):.3f}",
        })

    one_version(df_be, df_be_excl, "BE", "be")
    one_version(df_hco3, df_hco3_excl, "HCO3", "hco3")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "TableS12_eicu_incl_vs_excl.csv"), index=False)
    return out


# ---------- Table S13 ----------
def build_s13():
    alb_col = COLS["alb"]
    measured = df_full[df_full[alb_col].notna()].copy()
    not_measured = df_full[df_full[alb_col].isna()].copy()

    sm = summarize(measured, COLS)
    sn = summarize(not_measured, COLS)

    rows = []
    def add(label, key, is_binary=False):
        if is_binary:
            rows.append({
                "Variable": label,
                f"ALB measured (n = {len(measured)})": f"{sm[key]['n']} ({sm[key]['pct']:.1f})" if key != "male" else f"{sm[key]['n']} ({sm[key]['pct']:.1f})",
                f"ALB not measured (n = {len(not_measured)})": f"{sn[key]['n']} ({sn[key]['pct']:.1f})",
                "SMD": f"{smd_binary(sm[key]['pct']/100.0, sm['n_total'], sn[key]['pct']/100.0, sn['n_total']):.2f}",
                "P value": (f"{chi2_p(sm[key]['n'], sm['n_total']-sm[key]['n'], sn[key]['n'], sn['n_total']-sn[key]['n']):.3f}"),
            })
        else:
            rows.append({
                "Variable": label,
                f"ALB measured (n = {len(measured)})": f"{sm[key]['mean']:.1f} ({sm[key]['sd']:.1f})" if not pd.isna(sm[key]['mean']) else "—",
                f"ALB not measured (n = {len(not_measured)})": f"{sn[key]['mean']:.1f} ({sn[key]['sd']:.1f})" if not pd.isna(sn[key]['mean']) else "—",
                "SMD": f"{smd_continuous(sm[key]['mean'], sm[key]['sd'], sm['n_total'], sn[key]['mean'], sn[key]['sd'], sn['n_total']):.2f}",
                "P value": f"{welch_p(measured[COLS[key]], not_measured[COLS[key]]):.3f}",
            })

    add("Age, years, mean (SD)", "age")
    add("Male, %", "male", is_binary=True)
    add("BUN, mmol/L, mean (SD)", "bun")
    add("BE, mmol/L, mean (SD)", "be")
    add("In-hospital mortality, %", "outcome", is_binary=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "TableS13_eicu_alb_measured_vs_not.csv"), index=False)
    return out


# ---------- Table S6 "This study" 行 ----------
def build_s6_row():
    row = pd.DataFrame([{
        "Model / Study": "This study (ALB + BUN + BE)",
        "Population": "Respiratory failure (J96)",
        "Derivation n": "2,028",
        "External validation": "Yes (MIMIC-IV, n = 434; eICU-CRD, n = 1,505)",
        "Discrimination (AUC)": "0.62 (MIMIC-IV); 0.68 (eICU-CRD); 0.64/0.68 with HCO3",
        "Calibration reported": "Yes (slope, intercept, bootstrap OOB)",
        "Recalibration performed": "Yes (slope + intercept)",
    }])
    row.to_csv(os.path.join(RESULTS, "TableS6_this_study_row.csv"), index=False)
    return row


if __name__ == "__main__":
    print("读入：")
    print(f"  eicu_wide_be.csv     n = {len(df_be)}")
    print(f"  eicu_wide_hco3.csv   n = {len(df_hco3)}")
    print(f"  eicu_full_cohort.csv n = {len(df_full)}")
    print(f"  排除（BE）n = {len(df_be_excl)}，排除（HCO3）n = {len(df_hco3_excl)}")

    print("\n生成 Table S7b ...")
    t7 = build_s7b()
    print(t7.to_string(index=False))

    print("\n生成 Table S11 ...")
    t11 = build_s11()
    print(t11.to_string(index=False))

    print("\n生成 Table S12 ...")
    t12 = build_s12()
    print(t12.to_string(index=False))

    print("\n生成 Table S13 ...")
    t13 = build_s13()
    print(t13.to_string(index=False))

    print("\n生成 Table S6 This study 行 ...")
    t6 = build_s6_row()
    print(t6.to_string(index=False))

    print("\n完成。输出位于 results/：")
    for f in ["TableS7b_dev_vs_eicu.csv", "TableS11_eicu_baseline.csv",
              "TableS12_eicu_incl_vs_excl.csv",
              "TableS13_eicu_alb_measured_vs_not.csv",
              "TableS6_this_study_row.csv"]:
        print("  ", f)