# -*- coding: utf-8 -*-
"""
make_external_stats.py v1.2 (v6.2.10)
单一统计来源：BE 与 HCO3 的外部验证 AUC / Brier / slope / intercept 及其 CI。

v1.2：
  - load_pred 同时校验 coef_sha256 与当前冻结系数文件一致。
    mimic_sha256 只保证同一宽表；coef_sha256 保证同一套系数。
  - bootstrap_ci_auc_brier 返回 dict；四种指标分别记录有效重复次数。
  - MIN_VALID_BOOT=1900 对四种指标逐一强制，任一不达即中止。
  - 逐次重抽样结果落盘为 results/external_bootstrap_replicates_<v>.npz。

v1.1：
  - load_pred 校验 hadm_id 唯一、mimic_sha256、model_version、landmark。
  - bootstrap_ci_auc_brier 改为返回 dict（shared_calib v1.3）。
"""
import os, json, hashlib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from shared_calib import (
    to_logit, calib_slope_intercept, bootstrap_ci_auc_brier,
)
from ci_config import (
    CI_SEED, N_BOOT_CI, CI_METHOD, CI_LEVEL, MIN_VALID_BOOT,
    AUC_DECIMALS, BRIER_DECIMALS, CALIB_DECIMALS,
    SAMPLING_SCHEME_AUC,
)

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
PRED_BE     = os.path.join(RESULTS_DIR, "external_predictions.csv")
PRED_HCO3   = os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv")
OUT_CSV     = os.path.join(RESULTS_DIR, "external_auc_stats.csv")
OUT_JSON    = os.path.join(RESULTS_DIR, "external_auc_stats.json")

COEF_BE_PATH   = os.path.join(RESULTS_DIR, "external_validation_coef.csv")
COEF_HCO3_PATH = os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv")

EXPECTED_MIMIC_SHA256  = "b151612f75c568fce52f5a472206608682daaecd1a8ae2302595c45a0b3a72eb"
EXPECTED_MODEL_VERSION = "v6.2.2"
EXPECTED_LANDMARK      = "hospital_admission_plus_24h"

REQUIRED_PRED_COLUMNS = [
    "hadm_id", "outcome", "pred_prob",
    "coef_sha256", "mimic_sha256", "model_version", "landmark",
]


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_pred(path, label, expected_coef_sha256):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label}: prediction file not found: {path}")
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_PRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: missing columns {missing}")

    if df["hadm_id"].isna().any():
        raise ValueError(f"{label}: hadm_id contains NaN")
    if df["hadm_id"].duplicated().any():
        dup = int(df["hadm_id"].duplicated(keep=False).sum())
        raise ValueError(f"{label}: hadm_id not unique ({dup} rows involved)")

    if not (df["mimic_sha256"].astype(str) == EXPECTED_MIMIC_SHA256).all():
        raise ValueError(f"{label}: mimic_sha256 mismatch")

    if not (df["model_version"].astype(str) == EXPECTED_MODEL_VERSION).all():
        raise ValueError(f"{label}: model_version mismatch")

    if not (df["landmark"].astype(str) == EXPECTED_LANDMARK).all():
        raise ValueError(f"{label}: landmark mismatch")

    # v1.2: coef_sha256 must equal the currently frozen coefficient file
    observed_coef_sha = df["coef_sha256"].astype(str).unique().tolist()
    if len(observed_coef_sha) != 1:
        raise ValueError(
            f"{label}: coef_sha256 not unique across rows: {observed_coef_sha}")
    if observed_coef_sha[0] != expected_coef_sha256:
        raise ValueError(
            f"{label}: coef_sha256 mismatch\n"
            f"  expected: {expected_coef_sha256}\n"
            f"  observed: {observed_coef_sha[0]}")

    y = pd.to_numeric(df["outcome"], errors="raise").to_numpy(dtype=int)
    p = pd.to_numeric(df["pred_prob"], errors="raise").to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{label}: outcome must contain both 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f"{label}: pred_prob must be finite in [0,1]")
    return df, y, p


def compute_one(label, path, expected_coef_sha256):
    df, y, p = load_pred(path, label, expected_coef_sha256)

    auc   = float(roc_auc_score(y, p))
    brier = float(brier_score_loss(y, p))
    lp    = to_logit(p)
    slope, intercept = calib_slope_intercept(y, lp)

    boot = bootstrap_ci_auc_brier(y, p, n_boot=N_BOOT_CI, seed=CI_SEED)
    auc_lo, auc_hi     = boot["auc_lo"],   boot["auc_hi"]
    brier_lo, brier_hi = boot["brier_lo"], boot["brier_hi"]
    n_valid_auc        = boot["n_valid_auc"]
    n_valid_brier      = boot["n_valid_brier"]
    auc_replicates     = boot["auc_replicates"]
    brier_replicates   = boot["brier_replicates"]

    rng = np.random.default_rng(CI_SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    slopes     = np.full(N_BOOT_CI, np.nan, dtype=float)
    intercepts = np.full(N_BOOT_CI, np.nan, dtype=float)
    for b in range(N_BOOT_CI):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        yb, lb = y[idx], lp[idx]
        if yb.min() == yb.max():
            continue
        s, i = calib_slope_intercept(yb, lb)
        slopes[b]     = s
        intercepts[b] = i

    n_valid_slope     = int(np.isfinite(slopes).sum())
    n_valid_intercept = int(np.isfinite(intercepts).sum())

    for name, nv in [("AUC", n_valid_auc), ("Brier", n_valid_brier),
                     ("slope", n_valid_slope), ("intercept", n_valid_intercept)]:
        if nv < MIN_VALID_BOOT:
            raise RuntimeError(
                f"{label}: {name} valid replicates {nv} < "
                f"MIN_VALID_BOOT {MIN_VALID_BOOT}; refusing to write stats")

    alpha = 1.0 - CI_LEVEL
    slope_lo, slope_hi = np.nanpercentile(slopes,     [alpha/2*100, (1-alpha/2)*100])
    int_lo,   int_hi   = np.nanpercentile(intercepts, [alpha/2*100, (1-alpha/2)*100])

    replicate_path = os.path.join(
        RESULTS_DIR, f"external_bootstrap_replicates_{label}.npz")
    np.savez_compressed(
        replicate_path,
        auc_replicates=auc_replicates,
        brier_replicates=brier_replicates,
        slope_replicates=slopes,
        intercept_replicates=intercepts,
        n_boot=N_BOOT_CI,
        n_valid_auc=n_valid_auc,
        n_valid_brier=n_valid_brier,
        n_valid_slope=n_valid_slope,
        n_valid_intercept=n_valid_intercept,
        ci_seed=CI_SEED,
        ci_method=CI_METHOD,
        ci_level=CI_LEVEL,
        label=label,
    )
    print(f"[bootstrap replicates] {replicate_path}  "
          f"n_valid_auc={n_valid_auc}  n_valid_brier={n_valid_brier}  "
          f"n_valid_slope={n_valid_slope}  n_valid_intercept={n_valid_intercept}")

    return {
        "Version":              label,
        "n":                    int(len(y)),
        "events":               int(y.sum()),
        "AUC":                  auc,
        "AUC_lo":               round(float(auc_lo),   AUC_DECIMALS),
        "AUC_hi":               round(float(auc_hi),   AUC_DECIMALS),
        "Brier":                brier,
        "Brier_lo":             round(float(brier_lo), BRIER_DECIMALS),
        "Brier_hi":             round(float(brier_hi), BRIER_DECIMALS),
        "slope":                slope,
        "slope_lo":             round(float(slope_lo),  CALIB_DECIMALS),
        "slope_hi":             round(float(slope_hi),  CALIB_DECIMALS),
        "intercept":            intercept,
        "intercept_lo":         round(float(int_lo),    CALIB_DECIMALS),
        "intercept_hi":         round(float(int_hi),    CALIB_DECIMALS),
        "n_boot":               N_BOOT_CI,
        "n_boot_valid_auc":     n_valid_auc,
        "n_boot_valid_brier":   n_valid_brier,
        "n_boot_valid_slope":   n_valid_slope,
        "n_boot_valid_intercept": n_valid_intercept,
        "ci_seed":              CI_SEED,
        "ci_method":            CI_METHOD,
        "ci_level":             CI_LEVEL,
        "min_valid_boot":       MIN_VALID_BOOT,
        "sampling_scheme":      SAMPLING_SCHEME_AUC,
        "pred_file":            os.path.basename(path),
        "pred_sha256":          file_sha256(path),
    }


if not os.path.exists(COEF_BE_PATH):
    raise FileNotFoundError(f"Missing BE coefficient file: {COEF_BE_PATH}")
if not os.path.exists(COEF_HCO3_PATH):
    raise FileNotFoundError(f"Missing HCO3 coefficient file: {COEF_HCO3_PATH}")

EXPECTED_BE_COEF_SHA256   = file_sha256(COEF_BE_PATH)
EXPECTED_HCO3_COEF_SHA256 = file_sha256(COEF_HCO3_PATH)

print(f"[coef SHA256] BE   = {EXPECTED_BE_COEF_SHA256}")
print(f"[coef SHA256] HCO3 = {EXPECTED_HCO3_COEF_SHA256}")

rows = [compute_one("BE", PRED_BE, EXPECTED_BE_COEF_SHA256)]
if os.path.exists(PRED_HCO3):
    rows.append(compute_one("HCO3", PRED_HCO3, EXPECTED_HCO3_COEF_SHA256))

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2, default=str)

print(f"[external_auc_stats] {OUT_CSV}")
print(pd.DataFrame(rows).to_string(index=False))