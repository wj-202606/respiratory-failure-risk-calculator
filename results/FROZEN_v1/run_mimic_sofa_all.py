# -*- coding: utf-8 -*-
"""
run_mimic_sofa_all.py v2.5
v2.5：新增独立 AUC 的单变量 bootstrap 95% CI（AUC_model_lo/hi, AUC_sofa_lo/hi）。
      保持 v2.4 的 ΔAUC 配对分层 bootstrap CI 和 DeLong 配对 P。
v2.4：HCO3 预测文件不存在时跳过（MIMIC 宽表无 HCO3 列）。
"""

import os
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
MIMIC_PATH = os.environ.get("MIMIC_PATH", r"D:\mimic-iv-3.1")
SOFA_CSV = os.environ.get("SOFA_CSV", r"D:\mimic-code\results\mimic_sofa_official.csv")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide_with_hco3.csv")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true).astype(int)

    def compute_midrank(x):
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=float)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(N, dtype=float)
        T2[J] = T
        return T2

    def fastDeLong(predictions_sorted_transposed, label_1_count):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]
        tx = np.empty([k, m])
        ty = np.empty([k, n])
        tz = np.empty([k, m + n])
        for r in range(k):
            tx[r, :] = compute_midrank(positive_examples[r, :])
            ty[r, :] = compute_midrank(negative_examples[r, :])
            tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov

    preds = np.vstack([y_pred1, y_pred2])
    order = np.argsort(-y_true)
    label_1_count = int(y_true.sum())
    preds_sorted = preds[:, order]
    aucs, delongcov = fastDeLong(preds_sorted, label_1_count)

    l = np.array([[1, -1]])
    variance = float(np.dot(np.dot(l, delongcov), l.T).flatten()[0])
    delta = float(aucs[0] - aucs[1])
    if not np.isfinite(variance) or variance <= 0:
        z_value = 0.0 if abs(delta) < 1e-12 else float("inf")
        p_value = 1.0 if abs(delta) < 1e-12 else 0.0
    else:
        z_value = abs(delta) / np.sqrt(variance)
        p_value = 2 * norm.sf(z_value)
    return aucs[0], aucs[1], float(z_value), float(p_value)


def bootstrap_auc_ci(y_true, p_pred, n_boot=2000, seed=20261107,
                     confidence_level=0.95):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pred, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("AUC CI bootstrap input length mismatch or empty")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("AUC CI bootstrap requires both 0 and 1")
    if not np.isfinite(p).all():
        raise ValueError("AUC CI bootstrap contains non-finite predictions")

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        aucs[b] = roc_auc_score(y[idx], p[idx])

    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(aucs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def bootstrap_delta_auc_ci(y_true, pred_model, pred_sofa,
                           n_boot=2000, seed=20261103,
                           confidence_level=0.95):
    y = np.asarray(y_true, dtype=int)
    p_model = np.asarray(pred_model, dtype=float)
    p_sofa = np.asarray(pred_sofa, dtype=float)

    if not (len(y) == len(p_model) == len(p_sofa)) or len(y) == 0:
        raise ValueError("delta AUC bootstrap input length mismatch or empty")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("delta AUC bootstrap requires both 0 and 1")
    if (not np.isfinite(p_model).all() or
            not np.isfinite(p_sofa).all()):
        raise ValueError("delta AUC bootstrap contains non-finite predictions")
    if ((p_model < 0) | (p_model > 1)).any():
        raise ValueError("model predictions must be in [0, 1]")

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    delta = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        delta[b] = (
            roc_auc_score(y[idx], p_model[idx])
            - roc_auc_score(y[idx], p_sofa[idx])
        )

    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(delta, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def validate_prediction_file(path, coef_path, mimic_hash, label):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    required = ["hadm_id", "outcome", "pred_prob", "coef_sha256", "mimic_sha256"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{label} prediction file missing v6.2 columns: {missing}")
    if df["hadm_id"].isna().any() or df["hadm_id"].duplicated().any():
        raise ValueError(f"{label} prediction hadm_id must be non-missing and unique")
    if not (df["coef_sha256"].astype(str) == file_sha256(coef_path)).all():
        raise ValueError(f"{label} prediction coef_sha256 mismatch")
    if not (df["mimic_sha256"].astype(str) == mimic_hash).all():
        raise ValueError(f"{label} prediction mimic_sha256 mismatch")
    df["outcome"] = pd.to_numeric(df["outcome"], errors="raise").astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="raise").astype(float)
    if set(df["outcome"].unique()) != {0, 1} or not np.isfinite(df["pred_prob"]).all() \
            or ((df["pred_prob"] < 0) | (df["pred_prob"] > 1)).any():
        raise ValueError(f"{label} prediction outcome/pred_prob illegal")
    return df


def load_sofa_official():
    if not os.path.exists(SOFA_CSV):
        raise FileNotFoundError(
            f"Cannot find official SOFA: {SOFA_CSV}\n"
            f"Run postgres-make-concepts.sql and export first."
        )
    sofa = pd.read_csv(SOFA_CSV)
    required = ["stay_id", "sofa"]
    missing = [c for c in required if c not in sofa.columns]
    if missing:
        raise KeyError(f"mimic_sofa_official.csv missing columns {missing}; "
                       f"current columns: {sofa.columns.tolist()}")
    out = sofa[["stay_id", "sofa"]].rename(columns={"sofa": "sofa_total"}).copy()
    out["sofa_total"] = pd.to_numeric(out["sofa_total"], errors="coerce")
    out = out.dropna(subset=["stay_id", "sofa_total"])
    if out["stay_id"].duplicated().any():
        raise ValueError("official SOFA file has duplicate stay_id")
    return out


def build_hadm_to_stay():
    icustays = pd.read_csv(
        os.path.join(MIMIC_PATH, "icu", "icustays.csv.gz"),
        usecols=["hadm_id", "stay_id", "intime"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    hadm_to_stay = (icustays.sort_values(["hadm_id", "intime"])
                            .groupby("hadm_id")["stay_id"].first().reset_index())
    if hadm_to_stay["hadm_id"].duplicated().any():
        raise ValueError("hadm_id to stay_id mapping not unique")
    return hadm_to_stay


def main():
    print("=" * 70)
    print("run_mimic_sofa_all.py v2.5 - 3-variable model vs official first_day_sofa")
    print("=" * 70)

    print("\n[1] Load official first_day_sofa ...")
    sofa_df = load_sofa_official()
    print(f"  n = {len(sofa_df)}, "
          f"sofa median (IQR) = {sofa_df['sofa_total'].median():.1f} "
          f"({sofa_df['sofa_total'].quantile(0.25):.1f}-"
          f"{sofa_df['sofa_total'].quantile(0.75):.1f})")

    print("\n[2] hadm_id -> stay_id mapping ...")
    hadm_to_stay = build_hadm_to_stay()
    print(f"  unique hadm_id in icustays: {len(hadm_to_stay)}")

    print("\n[3] Load predictions ...")
    if not os.path.exists(MIMIC_CSV):
        raise FileNotFoundError(f"Cannot find MIMIC wide CSV: {MIMIC_CSV}")
    mimic_hash = file_sha256(MIMIC_CSV)
    coef_be_path = os.path.join(RESULTS_DIR, "external_validation_coef.csv")
    coef_hco3_path = os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv")
    be_pred = validate_prediction_file(
        os.path.join(RESULTS_DIR, "external_predictions.csv"),
        coef_be_path, mimic_hash, "BE")
    if be_pred is None:
        raise FileNotFoundError(
            "Missing BE prediction file: results/external_predictions.csv; "
            "run run_model.py v6.2.10 first"
        )
    hco3_pred = validate_prediction_file(
        os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv"),
        coef_hco3_path, mimic_hash, "HCO3")
    if hco3_pred is None:
        print("  HCO3 prediction file missing, skipping HCO3 SOFA comparison")
    print(f"  BE predictions: n = {len(be_pred)}")
    if hco3_pred is not None:
        print(f"  HCO3 predictions: n = {len(hco3_pred)}")

    print("\n[4] Merge + DeLong + AUC/delta-AUC bootstrap CI ...")
    summary_rows = []
    merge_diag_rows = []

    for name, df in [("BE", be_pred), ("HCO3", hco3_pred)]:
        if df is None:
            continue
        print(f"\n=== {name} model vs official first_day_sofa ===")
        prob_col = "pred_prob"

        if "hadm_id" not in df.columns or "outcome" not in df.columns:
            raise KeyError(f"{name} prediction file must contain hadm_id and outcome")
        if df["hadm_id"].duplicated().any():
            raise ValueError(f"{name} prediction hadm_id duplicated")
        merged = df.merge(hadm_to_stay, on="hadm_id", how="left", validate="one_to_one")
        n_no_stay = int(merged["stay_id"].isna().sum())

        merged = merged.merge(sofa_df, on="stay_id", how="left", validate="many_to_one")
        n_no_sofa_total = int(merged["sofa_total"].isna().sum())
        n_no_sofa_among_mapped = int((merged["stay_id"].notna() & merged["sofa_total"].isna()).sum())

        n_total = len(merged)
        n_with_sofa = int(merged["sofa_total"].notna().sum())
        print(f"  input n = {n_total}")
        print(f"  cannot map to stay_id: {n_no_stay}")
        print(f"  mapped but no SOFA: {n_no_sofa_among_mapped}")
        print(f"  SOFA available n = {n_with_sofa}")

        merge_diag_rows.append({
            "Version": name,
            "n_input": n_total,
            "n_no_stay_id": n_no_stay,
            "n_no_sofa": n_no_sofa_total,
            "n_no_sofa_among_mapped": n_no_sofa_among_mapped,
            "n_with_sofa": n_with_sofa,
        })

        merged = merged.dropna(subset=["sofa_total", prob_col])
        if len(merged) == 0:
            print(f"  [skip] no usable samples")
            continue

        if merged["outcome"].nunique() != 2:
            raise ValueError("SOFA comparison sample must contain both 0 and 1")
        y = merged["outcome"].values.astype(int)
        p_model = merged[prob_col].values.astype(float)
        p_sofa = merged["sofa_total"].values.astype(float)
        if not np.isfinite(p_model).all() or not np.isfinite(p_sofa).all():
            raise ValueError("SOFA paired comparison has non-finite values")

        auc_model = roc_auc_score(y, p_model)
        auc_sofa = roc_auc_score(y, p_sofa)
        auc1, auc2, z, p_val = delong_roc_test(y, p_model, p_sofa)

        seed_auc_model = 20261111 if name == "BE" else 20261113
        seed_auc_sofa  = 20261112 if name == "BE" else 20261114
        seed_delta     = 20261103 if name == "BE" else 20261104

        auc_model_lo, auc_model_hi = bootstrap_auc_ci(
            y, p_model, n_boot=2000, seed=seed_auc_model)
        auc_sofa_lo, auc_sofa_hi = bootstrap_auc_ci(
            y, p_sofa, n_boot=2000, seed=seed_auc_sofa)
        delta_lo, delta_hi = bootstrap_delta_auc_ci(
            y, p_model, p_sofa, n_boot=2000, seed=seed_delta)

        print(f"  final n = {len(merged)}, events = {int(y.sum())}")
        print(f"  3-variable model AUC: {auc_model:.4f} "
              f"(95% CI {auc_model_lo:.4f} to {auc_model_hi:.4f})")
        print(f"  SOFA AUC:            {auc_sofa:.4f} "
              f"(95% CI {auc_sofa_lo:.4f} to {auc_sofa_hi:.4f})")
        print(f"  delta AUC = {auc_model - auc_sofa:+.4f} "
              f"(paired bootstrap 95% CI {delta_lo:+.4f} to {delta_hi:+.4f})")
        print(f"  paired DeLong z = {z:.4f}, p = {p_val:.4e}")

        summary_rows.append({
            "Version": name,
            "n": len(merged),
            "events": int(y.sum()),
            "AUC_model": round(auc_model, 4),
            "AUC_model_lo": round(auc_model_lo, 4),
            "AUC_model_hi": round(auc_model_hi, 4),
            "AUC_sofa": round(auc_sofa, 4),
            "AUC_sofa_lo": round(auc_sofa_lo, 4),
            "AUC_sofa_hi": round(auc_sofa_hi, 4),
            "delta_AUC": round(auc_model - auc_sofa, 4),
            "delta_AUC_lo": round(delta_lo, 4),
            "delta_AUC_hi": round(delta_hi, 4),
            "delta_AUC_bootstrap_reps": 2000,
            "delta_AUC_bootstrap_method": "paired_outcome_stratified_percentile",
            "delong_z": round(z, 4),
            "delong_p": p_val,
        })

    print("\n[5] Write outputs ...")
    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_df.to_csv(os.path.join(RESULTS_DIR, "mimic_sofa_vs_model.csv"),
                      index=False, float_format="%.6f")
        print(f"  saved results/mimic_sofa_vs_model.csv")
        print(out_df.to_string(index=False))

    if merge_diag_rows:
        pd.DataFrame(merge_diag_rows).to_csv(
            os.path.join(RESULTS_DIR, "mimic_sofa_merge_diagnostics.csv"),
            index=False)
        print(f"  saved results/mimic_sofa_merge_diagnostics.csv")

    provenance = []
    for p, label in [
        (SOFA_CSV, "mimic_sofa_official.csv"),
        (os.path.join(RESULTS_DIR, "external_predictions.csv"), "external_predictions.csv"),
        (os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv"), "external_predictions_hco3_v2.csv"),
        (os.path.join(MIMIC_PATH, "icu", "icustays.csv.gz"), "icustays.csv.gz"),
    ]:
        if os.path.exists(p):
            provenance.append({
                "file": label, "path": p, "exists": True,
                "bytes": os.path.getsize(p), "sha256": file_sha256(p),
            })
        else:
            provenance.append({
                "file": label, "path": p, "exists": False,
                "bytes": np.nan, "sha256": "",
            })
    pd.DataFrame(provenance).to_csv(
        os.path.join(RESULTS_DIR, "mimic_sofa_input_provenance.csv"), index=False)
    print(f"  saved results/mimic_sofa_input_provenance.csv")

    print("\n" + "=" * 70)
    print("Methodological reminder: MIMIC first_day_sofa is the ICU first-day score;")
    print("if model prediction is at hospital admission + 24h, the time windows do not")
    print("fully align. Document this in Methods / Table S14; do NOT call it a strict")
    print("same-landmark comparison.")
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()