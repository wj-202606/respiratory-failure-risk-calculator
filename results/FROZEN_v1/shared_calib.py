# -*- coding: utf-8 -*-
"""
shared_calib.py v1.3 (v6.2.10)

v1.3：
  - bootstrap_ci_auc_brier 改为 outcome-stratified 重抽样。
  - bootstrap_ci_auc_brier 返回 dict，含 auc_lo/auc_hi/brier_lo/brier_hi、
    n_valid_auc/n_valid_brier、auc_replicates/brier_replicates。
  - 删除函数内 50% 有效重复阈值；阈值由调用方按 MIN_VALID_BOOT 执行。

被以下脚本 import：
  - recal_bootstrap.py
  - eicu_validate.py
  - make_external_stats.py
  - run_model.py
  - fig2_be.py
  - figS2_hco3.py
"""

import numpy as np
import pandas as pd
from functools import lru_cache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

OOB_MIN_N         = 20
OOB_MIN_EVENTS    = 3
OOB_MIN_NONEVENTS = 3
DEFAULT_SEED      = 123
DEFAULT_N_BOOT    = 1000
DEFAULT_N_CI      = 2000


def to_logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@lru_cache(maxsize=1)
def _supports_penalty_none():
    try:
        probe = LogisticRegression(penalty=None, solver="lbfgs", max_iter=50)
        probe.fit(np.asarray([[0.0], [1.0], [2.0], [3.0]]),
                  np.asarray([0, 0, 1, 1]))
        return True
    except (TypeError, ValueError):
        return False


def _make_unpenalized_logit():
    if _supports_penalty_none():
        return LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    return LogisticRegression(C=1e12, solver="lbfgs", max_iter=5000)


def calib_slope_intercept(y, logit_pred):
    m = _make_unpenalized_logit()
    m.fit(np.asarray(logit_pred).reshape(-1, 1), np.asarray(y).astype(int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def calib_slope_intercept_legacy(y, logit_pred):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    m.fit(np.asarray(logit_pred).reshape(-1, 1), np.asarray(y).astype(int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def net_benefit(y_true, p_pred, pt):
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    y_pred = (p_pred >= pt).astype(int)
    n = len(y_true)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    if pt >= 1:
        return 0.0
    return tp / n - fp / n * (pt / (1 - pt))


# ============================================================
# Bootstrap：AUC / Brier 的 95% CI（v6.2.10：分层 + 返回 dict）
# ============================================================
def bootstrap_ci_auc_brier(y, p, n_boot=DEFAULT_N_CI, seed=DEFAULT_SEED):
    """
    Percentile bootstrap 95% CI for AUC and Brier score.
    v6.2.10: outcome-stratified resampling (previously: ordinary bootstrap).
    Returns a dict with CIs, per-metric valid replicate counts, and the
    full replicate arrays for audit.

    Caller must enforce MIN_VALID_BOOT on n_valid_auc and n_valid_brier.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("bootstrap_ci_auc_brier: length mismatch or empty input")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("bootstrap_ci_auc_brier: outcome must contain both 0 and 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("bootstrap_ci_auc_brier: pred must be finite in [0,1]")

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    aucs   = np.full(n_boot, np.nan, dtype=float)
    briers = np.full(n_boot, np.nan, dtype=float)

    for b in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        yb, pb = y[idx], p[idx]
        aucs[b]   = roc_auc_score(yb, pb)
        briers[b] = brier_score_loss(yb, pb)

    n_valid_auc   = int(np.isfinite(aucs).sum())
    n_valid_brier = int(np.isfinite(briers).sum())

    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])
    brier_lo, brier_hi = np.percentile(briers, [2.5, 97.5])

    return {
        "auc_lo": float(auc_lo),
        "auc_hi": float(auc_hi),
        "brier_lo": float(brier_lo),
        "brier_hi": float(brier_hi),
        "n_valid_auc": n_valid_auc,
        "n_valid_brier": n_valid_brier,
        "auc_replicates": aucs,
        "brier_replicates": briers,
    }


# ============================================================
# Bootstrap OOB 重校准（保持不变）
# ============================================================
def bootstrap_oob_recal(y, p, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED):
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("OOB重校准要求y与p长度相同且非空")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("OOB预测概率必须为有限的[0,1]数值")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("OOB重校准结局必须同时包含0和1")
    n = len(y)
    logit_p = to_logit(p)
    rows = []
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if (len(oob) < OOB_MIN_N or
                y[oob].sum() < OOB_MIN_EVENTS or
                (len(oob) - y[oob].sum()) < OOB_MIN_NONEVENTS):
            continue
        m = _make_unpenalized_logit()
        m.fit(logit_p[idx].reshape(-1, 1), y[idx])
        a = float(m.coef_[0, 0]); b = float(m.intercept_[0])
        logit_oob = a * logit_p[oob] + b
        p_oob = 1.0 / (1.0 + np.exp(-logit_oob))
        try:
            brier = brier_score_loss(y[oob], p_oob)
            sl, ic = calib_slope_intercept(y[oob], logit_oob)
        except Exception:
            continue
        rows.append({
            "iter": i, "n_oob": len(oob),
            "a": a, "b": b,
            "Brier_oob": brier,
            "slope_oob": sl,
            "intercept_oob": ic,
        })
    return pd.DataFrame(rows)


def summarize_oob(boot_df):
    keys = ["Brier_OOB", "slope_OOB", "intercept_OOB"]
    out = {}
    if boot_df is None or len(boot_df) == 0:
        for k in keys:
            out[f"{k}_median"] = np.nan
            out[f"{k}_q25"]    = np.nan
            out[f"{k}_q75"]    = np.nan
        return out
    mapping = {
        "Brier_OOB":     "Brier_oob",
        "slope_OOB":     "slope_oob",
        "intercept_OOB": "intercept_oob",
    }
    for k_out, k_in in mapping.items():
        s = boot_df[k_in]
        out[f"{k_out}_median"] = float(s.median())
        out[f"{k_out}_q25"]    = float(s.quantile(0.25))
        out[f"{k_out}_q75"]    = float(s.quantile(0.75))
    return out