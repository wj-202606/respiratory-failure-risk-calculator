# -*- coding: utf-8 -*-
"""
ci_config.py v1.1 (v6.2.10)
统一 AUC / Brier / 校准参数置信区间的 bootstrap 参数。
任何脚本需要做 CI 时，必须 from ci_config import ...，不得自写常量。

v1.1：
  - SAMPLING_SCHEME_AUC 与 shared_calib 实际实现对齐（分层重抽样）。
  - MIN_VALID_BOOT 提升到 1900，且 AUC / Brier / slope / intercept
    分别统计有效次数，任一低于 1900 即中止。
"""
CI_SEED   = 20261101
N_BOOT_CI = 2000
CI_METHOD = "percentile"
CI_LEVEL  = 0.95

MIN_VALID_BOOT = int(0.95 * N_BOOT_CI)   # 1900

AUC_DECIMALS   = 3
BRIER_DECIMALS = 3
CALIB_DECIMALS = 3

SAMPLING_SCHEME_AUC   = "outcome_stratified_resample"
SAMPLING_SCHEME_DELTA = "paired_outcome_stratified"

SOFA_APACHE_SEEDS = {
    "sofa_be_model_auc":   20261111,
    "sofa_be_sofa_auc":    20261112,
    "sofa_be_delta":       20261103,
    "sofa_hco3_model_auc": 20261113,
    "sofa_hco3_sofa_auc":  20261114,
    "sofa_hco3_delta":     20261104,
}