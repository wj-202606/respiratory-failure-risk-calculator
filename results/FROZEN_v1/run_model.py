# -*- coding: utf-8 -*-
"""
Respiratory failure prediction model - Python ML comparison + SHAP
v6.2.10: Figure 2 moved to fig2_be.py to remove circular dependency
         (run_model -> make_external_stats -> Figure 2).
         Per-patient predictions for internal and external validation
         are dumped to ml_predictions_*_v6.2.10.csv (audit / DeLong).
v6.2.9 : Figure 2 AUC CI read from external_auc_stats.csv (unified CI_SEED=20261101).
v6.2.8 : MIMIC wide SHA256 hard guard; manifest gains wide rows/cols/hco3 source.
v6.2.2 : HCO3 frozen predictions skipped if mimic_wide.csv has no HCO3 column.
v6.2.1 : Fixed XGBoost early stopping.
v6.2   : Generate BE / HCO3 admission-level frozen predictions.
"""
import os
import sys
import inspect
import hashlib
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from scipy import stats

from shared_calib import (
    to_logit,
    calib_slope_intercept,
    bootstrap_ci_auc_brier,
    DEFAULT_N_CI,
    DEFAULT_SEED,
)
from ci_config import CI_SEED, N_BOOT_CI, CI_METHOD

BASE = os.environ.get("PROJECT_BASE", os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
FIGURES_DIR = os.path.join(BASE, "figures")
FIG_DIR_CHEST = os.path.join(BASE, "论文图")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(FIG_DIR_CHEST, exist_ok=True)

TRAIN_CSV = os.path.join(RESULTS_DIR, "traindata_descriptive.csv")
VAL_CSV   = os.path.join(RESULTS_DIR, "valdata_descriptive.csv")
MIMIC_CSV = os.environ.get("MIMIC_WIDE_CSV", r"D:\mimic_wide_with_hco3.csv")

EXPECTED_MIMIC_SHA256 = (
    "b151612f75c568fce52f5a472206608682daaecd1a8ae2302595c45a0b3a72eb"
)

DEPLOY_COEF_BE   = os.path.join(RESULTS_DIR, "external_validation_coef.csv")
DEPLOY_COEF_HCO3 = os.path.join(RESULTS_DIR, "external_validation_coef_hco3.csv")

RNG = DEFAULT_SEED
FEATURES = ["ALB", "BUN", "BE"]

print("=" * 70)
print("Python:", sys.executable)
print("BASE:", BASE)
print("MIMIC_CSV:", MIMIC_CSV)
print(f"CI_SEED={CI_SEED}  N_BOOT_CI={N_BOOT_CI}  CI_METHOD={CI_METHOD}")
print("=" * 70)

sys.path.insert(0, BASE)
try:
    from chest_style import apply_style
    apply_style()
    print("[style] chest_style applied")
except Exception as _e:
    print(f"[style] chest_style not found, using default: {_e}")

plt.rcParams['axes.edgecolor']  = 'black'
plt.rcParams['axes.linewidth']  = 1.0
plt.rcParams['xtick.color']     = 'black'
plt.rcParams['ytick.color']     = 'black'
plt.rcParams['text.color']      = 'black'
plt.rcParams['axes.labelcolor'] = 'black'

def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def assert_mimic_wide_locked(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"MIMIC wide table not found: {path}")
    actual = file_sha256(path)
    if actual != EXPECTED_MIMIC_SHA256:
        sys.exit(
            "[FATAL] MIMIC wide SHA256 mismatch.\n"
            f"  expected: {EXPECTED_MIMIC_SHA256}\n"
            f"  actual  : {actual}\n"
            f"  path    : {path}\n"
            "If the wide table was intentionally replaced, update "
            "EXPECTED_MIMIC_SHA256 in all scripts and VERSION files."
        )
    print(f"[OK] MIMIC wide SHA256 verified: {actual}")
    return actual

def load_deploy_coef(path, version_label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Deploy coefficient file not found: {path}\n"
            f"Generate it in R_analysis_v5_4_1.R section 8 / 8b-1."
        )
    df = pd.read_csv(path)
    term_col = "Term" if "Term" in df.columns else df.columns[0]
    beta_col = "Beta" if "Beta" in df.columns else (
        "estimate" if "estimate" in df.columns else df.columns[1])
    coef = dict(zip(df[term_col].astype(str), df[beta_col].astype(float)))
    if "(Intercept)" in coef and "Intercept" not in coef:
        coef["Intercept"] = coef.pop("(Intercept)")
    for need in ["Intercept", "ALB", "BUN"]:
        if need not in coef:
            raise KeyError(f"{version_label} missing coef {need}: {list(coef.keys())}")
    expected = ["Intercept", "ALB", "BUN", ("BE" if version_label == "BE" else "HCO3")]
    missing = [k for k in expected if k not in coef]
    if missing:
        raise KeyError(f"{version_label} missing required terms: {missing}")
    extra = [k for k in coef if k not in expected]
    if extra:
        raise ValueError(f"{version_label} has unexpected terms: {extra}")
    if not all(np.isfinite(v) for v in coef.values()):
        raise ValueError(f"{version_label} contains non-finite coefficient")
    return coef

DEPLOY_BE = load_deploy_coef(DEPLOY_COEF_BE, "BE")
print(f"[coef] BE deploy coefficients: {DEPLOY_BE}")
if not os.path.exists(DEPLOY_COEF_HCO3):
    raise FileNotFoundError(f"Missing HCO3 frozen coef: {DEPLOY_COEF_HCO3}")
DEPLOY_HCO3 = load_deploy_coef(DEPLOY_COEF_HCO3, "HCO3")
print(f"[coef] HCO3 deploy coefficients: {DEPLOY_HCO3}")

def predict_from_coef(df, coef, x_col):
    lp = (coef["Intercept"]
          + coef["ALB"] * df["ALB"].values
          + coef["BUN"] * df["BUN"].values
          + coef[x_col] * df[x_col].values)
    return 1.0 / (1.0 + np.exp(-lp))

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not installed")

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not installed")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[warn] shap not installed")

def fit_xgb_with_early_stopping(model, X_train, y_train,
                                X_valid, y_valid, rounds=50):
    fit_params = inspect.signature(model.fit).parameters
    fitted = False

    if "early_stopping_rounds" in fit_params:
        try:
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      early_stopping_rounds=rounds, verbose=False)
            fitted = True
        except (TypeError, ValueError):
            fitted = False

    if not fitted:
        try:
            model.set_params(early_stopping_rounds=rounds)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)], verbose=False)
            fitted = True
        except (TypeError, ValueError):
            fitted = False

    if not fitted:
        try:
            from xgboost.callback import EarlyStopping as XGBEarlyStopping
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost early stopping not supported by installed version"
            ) from exc
        model.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  callbacks=[XGBEarlyStopping(rounds=rounds, save_best=False)],
                  verbose=False)
        fitted = True

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        raise RuntimeError("XGBoost early stopping did not set best_iteration")
    best_n_estimators = int(best_iteration) + 1
    if best_n_estimators < 1:
        raise RuntimeError("XGBoost n_estimators is invalid")
    return model, best_n_estimators

print("\n[1/6] Reading data ...")
if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")
if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(f"Validation CSV not found: {VAL_CSV}")

train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)

required_cols = FEATURES + ["Outcome"]
for label, frame in [("train", train), ("validation", val)]:
    missing = [c for c in required_cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{label} file missing columns: {missing}")
    if frame[required_cols].isna().any().any():
        raise ValueError(f"{label} has missing ML features or Outcome")
    if set(frame["Outcome"].unique()) != {0, 1}:
        raise ValueError(f"{label} Outcome must contain both 0 and 1")
common_id = next((c for c in ["patient_id", "subject_id", "hadm_id"]
                  if c in train.columns and c in val.columns), None)
if common_id and set(train[common_id].dropna()).intersection(set(val[common_id].dropna())):
    raise ValueError(f"Train and validation share {common_id}")

X_train = train[FEATURES].to_numpy(dtype=float)
y_train = train["Outcome"].to_numpy(dtype=int)
X_val   = val[FEATURES].to_numpy(dtype=float)
y_val   = val["Outcome"].to_numpy(dtype=int)
if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
    raise ValueError("ML features contain Inf or non-finite values")

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_val_s   = scaler.transform(X_val)

print("\n[2/6] Training models ...")
models = {
    "Logistic": LogisticRegression(max_iter=2000, C=1.0, random_state=RNG),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, max_depth=6, min_samples_leaf=10,
        random_state=RNG, n_jobs=-1),
    "SVM": SVC(probability=True, kernel="rbf", C=1.0, gamma="scale",
               random_state=RNG),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=1e-3,
        max_iter=2000, early_stopping=True, validation_fraction=0.15,
        random_state=RNG),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1,
        eval_metric="logloss", random_state=RNG, n_jobs=-1)
if HAS_LGB:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        num_leaves=8, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=RNG, verbose=-1)

SCALED = {"Logistic", "SVM", "MLP"}

def early_stop_by_inner_cv(name, Xtr, ytr, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RNG)
    best_iters = []
    for tr_idx, es_idx in skf.split(Xtr, ytr):
        Xt, Xe = Xtr[tr_idx], Xtr[es_idx]
        yt, ye = ytr[tr_idx], ytr[es_idx]
        if name == "XGBoost":
            m = XGBClassifier(
                n_estimators=1000, learning_rate=0.03, max_depth=3,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.1,
                eval_metric="logloss", random_state=RNG, n_jobs=-1)
            _, best_n = fit_xgb_with_early_stopping(m, Xt, yt, Xe, ye, rounds=50)
            best_iters.append(int(best_n))
        elif name == "LightGBM":
            m = LGBMClassifier(
                n_estimators=1000, learning_rate=0.03, max_depth=3,
                num_leaves=8, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, random_state=RNG, verbose=-1)
            try:
                m.fit(Xt, yt, eval_set=[(Xe, ye)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            except TypeError:
                m.fit(Xt, yt, eval_set=[(Xe, ye)],
                      early_stopping_rounds=50, verbose=False)
            best_iters.append(getattr(m, "best_iteration_", 1000) or 1000)
    return best_iters

fitted = {}
best_iters_dict = {}
results = []
for name, model in models.items():
    print(f"  -> {name} ...", end=" ")
    Xtr = X_train_s if name in SCALED else X_train
    Xva = X_val_s   if name in SCALED else X_val
    try:
        if name in ("XGBoost", "LightGBM"):
            inner_iters = early_stop_by_inner_cv(name, Xtr, y_train, n_splits=5)
            best_iters_dict[name] = inner_iters
            final_n = max(1, int(np.median(inner_iters)))
            print(f"[inner-CV iters={inner_iters}, median={final_n}] ", end="")
            if name == "XGBoost":
                final_model = XGBClassifier(
                    n_estimators=final_n, learning_rate=0.03, max_depth=3,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, reg_alpha=0.1,
                    eval_metric="logloss", random_state=RNG, n_jobs=-1)
                final_model.fit(Xtr, y_train, verbose=False)
                model = final_model
            else:
                final_model = LGBMClassifier(
                    n_estimators=final_n, learning_rate=0.03, max_depth=3,
                    num_leaves=8, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=RNG, verbose=-1)
                final_model.fit(Xtr, y_train)
                model = final_model
        else:
            model.fit(Xtr, y_train)

        p_val = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        brier = brier_score_loss(y_val, p_val)
        fitted[name] = model
        results.append({"Model": name,
                        "AUC_val": round(auc, 4),
                        "Brier_val": round(brier, 4),
                        "best_iter_inner_cv": np.nan if name not in best_iters_dict
                                              else int(np.median(best_iters_dict[name]))})
        print(f"AUC={auc:.4f}")
    except Exception as e:
        print(f"failed: {e}")

res_df = pd.DataFrame(results).sort_values("AUC_val", ascending=False)
res_df.to_csv(os.path.join(RESULTS_DIR, "python_model_comparison.csv"), index=False)

if best_iters_dict:
    rows = []
    for name, iters in best_iters_dict.items():
        for fold_i, it in enumerate(iters, start=1):
            rows.append({"Model": name, "fold": fold_i, "best_iter": it})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "python_ml_inner_cv_iters.csv"), index=False)

print("\n[3/6] External validation (MIMIC-IV) ...")
ext_df = None
mimic_cc = None
p_ext_frozen = None
p_ext_python_logit = None
y_mimic = None

if not os.path.exists(MIMIC_CSV):
    raise FileNotFoundError(
        f"MIMIC wide table not found: {MIMIC_CSV}. "
        "v6.2 requires a successful generation of both BE and HCO3 frozen predictions."
    )
else:
    mimic_hash = assert_mimic_wide_locked(MIMIC_CSV)
    mimic = pd.read_csv(MIMIC_CSV)
    mimic_n_rows = int(len(mimic))
    mimic_n_cols = int(mimic.shape[1])

    coef_be_hash = file_sha256(DEPLOY_COEF_BE)
    coef_hco3_hash = file_sha256(DEPLOY_COEF_HCO3)

    if "hadm_id" not in mimic.columns:
        raise KeyError("MIMIC wide table missing hadm_id")
    if mimic["hadm_id"].isna().any():
        raise ValueError("MIMIC wide table hadm_id has missing values")
    if mimic["hadm_id"].duplicated().any():
        dup_n = int(mimic["hadm_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"MIMIC wide table has duplicate hadm_id ({dup_n} rows involved)")
    col_alb = "alb_g_l"  if "alb_g_l"  in mimic.columns else "ALB"
    col_bun = "bun_mmol_l" if "bun_mmol_l" in mimic.columns else "BUN"
    col_be  = "be_mmol_l" if "be_mmol_l" in mimic.columns else "BE"
    col_out = "outcome" if "outcome" in mimic.columns else "Outcome"

    need = [col_alb, col_bun, col_be, col_out]
    missing_be = [c for c in need if c not in mimic.columns]
    if missing_be:
        raise KeyError(f"MIMIC wide table missing columns for BE: {missing_be}")

    mimic_cc = mimic.dropna(subset=need).copy()
    print(f"  complete cases: {len(mimic_cc)}, events: {int(mimic_cc[col_out].sum())}")

    X_mimic = mimic_cc[[col_alb, col_bun, col_be]].values
    y_mimic = pd.to_numeric(mimic_cc[col_out], errors="raise").values.astype(int)
    if set(np.unique(y_mimic)) != {0, 1}:
        raise ValueError("MIMIC BE complete-case outcome must contain both 0 and 1")
    X_mimic_s = scaler.transform(X_mimic)

    mimic_df_for_coef = pd.DataFrame({
        "ALB": X_mimic[:, 0], "BUN": X_mimic[:, 1], "BE": X_mimic[:, 2]
    })

    ext_rows = []
    p_ext_frozen = predict_from_coef(mimic_df_for_coef, DEPLOY_BE, "BE")
    ext_rows.append({
        "Model": "FrozenR_Logistic",
        "AUC_ext": round(roc_auc_score(y_mimic, p_ext_frozen), 4),
        "Brier_ext": round(brier_score_loss(y_mimic, p_ext_frozen), 4),
        "Note": "R Rubin pooled coefficients from external_validation_coef.csv",
    })
    if "Logistic" in fitted:
        p_ext_python_logit = fitted["Logistic"].predict_proba(X_mimic_s)[:, 1]
        ext_rows.append({
            "Model": "Python_Logistic",
            "AUC_ext": round(roc_auc_score(y_mimic, p_ext_python_logit), 4),
            "Brier_ext": round(brier_score_loss(y_mimic, p_ext_python_logit), 4),
            "Note": "sklearn LogisticRegression (L2, C=1.0) predict_proba",
        })
    for name, model in fitted.items():
        if name == "Logistic":
            continue
        Xm = X_mimic_s if name in SCALED else X_mimic
        p_ext = model.predict_proba(Xm)[:, 1]
        ext_rows.append({
            "Model": name,
            "AUC_ext": round(roc_auc_score(y_mimic, p_ext), 4),
            "Brier_ext": round(brier_score_loss(y_mimic, p_ext), 4),
            "Note": "",
        })
    ext_df = pd.DataFrame(ext_rows).sort_values("AUC_ext", ascending=False)
    ext_df.to_csv(os.path.join(RESULTS_DIR, "python_external_validation.csv"), index=False)
    print(ext_df.to_string(index=False))

    if p_ext_frozen is not None:
        if mimic_cc["hadm_id"].isna().any() or mimic_cc["hadm_id"].duplicated().any():
            raise ValueError("BE prediction hadm_id must be non-missing and unique")
        pred_df = pd.DataFrame({
            "hadm_id": mimic_cc["hadm_id"].values,
            "outcome": y_mimic,
            "pred_prob": p_ext_frozen,
            "coef_sha256": coef_be_hash,
            "mimic_sha256": mimic_hash,
            "model_version": "v6.2.2",
            "landmark": "hospital_admission_plus_24h",
        })
        pred_df.to_csv(os.path.join(RESULTS_DIR, "external_predictions.csv"), index=False)

    col_hco3 = None
    if "hco3_mmol_l" in mimic.columns:
        col_hco3 = "hco3_mmol_l"
    elif "HCO3" in mimic.columns:
        col_hco3 = "HCO3"
    elif "hco3" in mimic.columns:
        col_hco3 = "hco3"

    pred_hco3_df = None
    hco3_source = "not_available"
    if col_hco3 is None:
        print("[HCO3 frozen predictions] mimic_wide.csv has no HCO3 column, skipping")
    else:
        need_hco3 = [col_alb, col_bun, col_hco3, col_out, "hadm_id"]
        missing_hco3 = [c for c in need_hco3 if c not in mimic.columns]
        if missing_hco3:
            print(f"[HCO3 frozen predictions] mimic_wide.csv missing {missing_hco3}, skipping")
        else:
            mimic_hco3_cc = mimic.dropna(subset=need_hco3).copy()
            if mimic_hco3_cc["hadm_id"].isna().any() or mimic_hco3_cc["hadm_id"].duplicated().any():
                raise ValueError("HCO3 complete-case hadm_id must be non-missing and unique")
            y_hco3 = pd.to_numeric(mimic_hco3_cc[col_out], errors="raise").to_numpy(dtype=int)
            if set(np.unique(y_hco3)) != {0, 1}:
                raise ValueError("MIMIC HCO3 complete-case outcome must contain both 0 and 1")
            hco3_input = pd.DataFrame({
                "ALB": pd.to_numeric(mimic_hco3_cc[col_alb], errors="raise"),
                "BUN": pd.to_numeric(mimic_hco3_cc[col_bun], errors="raise"),
                "HCO3": pd.to_numeric(mimic_hco3_cc[col_hco3], errors="raise"),
            })
            p_hco3_frozen = predict_from_coef(hco3_input, DEPLOY_HCO3, "HCO3")
            if not np.isfinite(p_hco3_frozen).all() or ((p_hco3_frozen < 0) | (p_hco3_frozen > 1)).any():
                raise ValueError("MIMIC HCO3 frozen prediction probability illegal")
            pred_hco3_df = pd.DataFrame({
                "hadm_id": mimic_hco3_cc["hadm_id"].values,
                "outcome": y_hco3,
                "pred_prob": p_hco3_frozen,
                "coef_sha256": coef_hco3_hash,
                "mimic_sha256": mimic_hash,
                "model_version": "v6.2.2",
                "landmark": "hospital_admission_plus_24h",
            })
            pred_hco3_df.to_csv(os.path.join(RESULTS_DIR, "external_predictions_hco3_v2.csv"), index=False)
            hco3_source = f"mimic_wide_with_hco3.csv column={col_hco3}"
            print(f"[HCO3 frozen predictions] n={len(pred_hco3_df)}, events={int(y_hco3.sum())}; saved")

    manifest_rows = [
        {"Version":"BE", "n":len(pred_df), "events":int(y_mimic.sum()),
         "coef_sha256":coef_be_hash, "mimic_sha256":mimic_hash,
         "landmark":"hospital_admission_plus_24h", "model_version":"v6.2.2",
         "mimic_n_rows": mimic_n_rows, "mimic_n_cols": mimic_n_cols,
         "hco3_source": "not_used_by_BE"},
    ]
    if pred_hco3_df is not None:
        manifest_rows.append({
            "Version":"HCO3", "n":len(pred_hco3_df), "events":int(pred_hco3_df["outcome"].sum()),
            "coef_sha256":coef_hco3_hash, "mimic_sha256":mimic_hash,
            "landmark":"hospital_admission_plus_24h", "model_version":"v6.2.2",
            "mimic_n_rows": mimic_n_rows, "mimic_n_cols": mimic_n_cols,
            "hco3_source": hco3_source,
        })
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(RESULTS_DIR, "mimic_prediction_manifest.csv"), index=False)

    if p_ext_python_logit is not None:
        pd.DataFrame({
            "outcome": y_mimic,
            "pred_prob": p_ext_python_logit,
        }).to_csv(os.path.join(RESULTS_DIR,
                               "external_predictions_python_logistic.csv"), index=False)

def _compute_midrank(x):
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

def _fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def paired_delong(y_true, p1, p2):
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")
    y_sorted = y_true[order]
    preds = np.vstack((np.asarray(p1)[order], np.asarray(p2)[order]))
    m = int(y_sorted.sum())
    aucs, cov = _fastDeLong(preds, m)
    var = np.diag(cov)
    if var[0] + var[1] - 2 * cov[0, 1] <= 0:
        return float(aucs[0]), float(aucs[1]), 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var[0] + var[1] - 2 * cov[0, 1])
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)

delong_internal_rows = []
if "Logistic" in fitted and len(res_df) > 1:
    best_ml = [m for m in res_df["Model"] if m != "Logistic"][0]
    p_log = fitted["Logistic"].predict_proba(X_val_s)[:, 1]
    X_ml = X_val_s if best_ml in SCALED else X_val
    p_ml = fitted[best_ml].predict_proba(X_ml)[:, 1]
    a1, a2, z, p = paired_delong(y_val, p_log, p_ml)
    delong_internal_rows.append({
        "Comparison": f"Python_Logistic vs {best_ml}",
        "AUC_1": round(a1, 4), "AUC_2": round(a2, 4),
        "z": round(z, 4), "p_paired_delong": round(p, 4),
        "n": len(y_val),
    })
if (p_ext_frozen is not None) and (p_ext_python_logit is not None):
    a1, a2, z, p = paired_delong(y_mimic, p_ext_frozen, p_ext_python_logit)
    delong_internal_rows.append({
        "Comparison": "FrozenR_Logistic vs Python_Logistic (external)",
        "AUC_1": round(a1, 4), "AUC_2": round(a2, 4),
        "z": round(z, 4), "p_paired_delong": round(p, 4),
        "n": len(y_mimic),
    })
if delong_internal_rows:
    pd.DataFrame(delong_internal_rows).to_csv(
        os.path.join(RESULTS_DIR, "delong_paired_results.csv"), index=False)
    print("\n[DeLong paired]")
    print(pd.DataFrame(delong_internal_rows).to_string(index=False))

from matplotlib.colors import LinearSegmentedColormap
CMAP_SHAP = LinearSegmentedColormap.from_list(
    "blue_red",
    ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"],
    N=256
)
if HAS_SHAP:
    tree_candidates = []
    if "XGBoost" in fitted:    tree_candidates.append("XGBoost")
    if "LightGBM" in fitted:   tree_candidates.append("LightGBM")
    if "RandomForest" in fitted: tree_candidates.append("RandomForest")
    if tree_candidates:
        best_tree = tree_candidates[0]
        explainer = shap.TreeExplainer(fitted[best_tree])
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
        plt.figure(figsize=(8, 5.0))
        shap.summary_plot(shap_values, X_train,
                          feature_names=FEATURES, plot_type="dot",
                          cmap=CMAP_SHAP, show=False, plot_size=None)
        plt.xlabel("SHAP value (log-odds)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR_CHEST, "FigureS3_SHAP_summary.pdf"),
                    dpi=1200, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(6, 4))
        shap.summary_plot(shap_values, X_train,
                          feature_names=FEATURES, plot_type="bar",
                          color="#1F77B4", show=False, plot_size=None)
        plt.xlabel("mean(|SHAP value|)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR_CHEST, "FigureS3b_SHAP_bar.pdf"),
                    dpi=1200, bbox_inches="tight")
        plt.close()

# ============================================================
# 6b. v6.2.10: dump per-patient predictions (audit + DeLong)
# ============================================================
_MODEL_VERSION = "v6.2.10"
_PREDICTION_TS = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _sha_or_empty(path):
    return file_sha256(path) if os.path.exists(path) else ""

# ---- internal validation (n=608) ----
if "Logistic" in fitted:
    internal_cols = {
        "patient_index": np.arange(len(y_val)),
        "outcome": y_val,
    }
    for name in ["Logistic", "RandomForest", "XGBoost", "LightGBM", "SVM", "MLP"]:
        if name not in fitted:
            continue
        X_eval = X_val_s if name in SCALED else X_val
        internal_cols[name] = fitted[name].predict_proba(X_eval)[:, 1]

    internal_df = pd.DataFrame(internal_cols)
    internal_path = os.path.join(
        RESULTS_DIR, f"ml_predictions_internal_{_MODEL_VERSION}.csv")

    with open(internal_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "# NOTE: patient_index is the 0-based row index within "
            "results/valdata_descriptive.csv.\n"
            "#       The source CSV does not carry patient IDs, so no "
            "synthetic clinical identifier is used.\n"
            f"# model_version: {_MODEL_VERSION}\n"
            f"# prediction_timestamp: {_PREDICTION_TS}\n"
            f"# train_csv_sha256: {_sha_or_empty(TRAIN_CSV)}\n"
            f"# val_csv_sha256:   {_sha_or_empty(VAL_CSV)}\n"
            f"# coef_be_sha256:   {_sha_or_empty(DEPLOY_COEF_BE)}\n"
            f"# n: {len(y_val)}\n"
            f"# events: {int(y_val.sum())}\n"
        )
        internal_df.to_csv(f, index=False)
    print(f"[predictions] internal: {internal_path} "
          f"(n={len(internal_df)}, events={int(y_val.sum())})")

# ---- external validation (n=434, BE complete-case) ----
if (p_ext_frozen is not None) and (mimic_cc is not None):
    external_cols = {
        "hadm_id": mimic_cc["hadm_id"].values,
        "outcome": y_mimic,
        "FrozenR_Logistic": p_ext_frozen,
    }
    if p_ext_python_logit is not None:
        external_cols["Python_Logistic"] = p_ext_python_logit
    for name in ["RandomForest", "XGBoost", "LightGBM", "SVM", "MLP"]:
        if name not in fitted:
            continue
        X_eval = X_mimic_s if name in SCALED else X_mimic
        external_cols[name] = fitted[name].predict_proba(X_eval)[:, 1]

    external_df = pd.DataFrame(external_cols)
    external_path = os.path.join(
        RESULTS_DIR, f"ml_predictions_external_{_MODEL_VERSION}.csv")

    with open(external_path, "w", encoding="utf-8", newline="") as f:
        f.write(
            "# NOTE: hadm_id is the MIMIC-IV hospital admission identifier.\n"
            f"# model_version: {_MODEL_VERSION}\n"
            f"# prediction_timestamp: {_PREDICTION_TS}\n"
            f"# mimic_sha256:    {mimic_hash}\n"
            f"# coef_be_sha256:  {_sha_or_empty(DEPLOY_COEF_BE)}\n"
            f"# n: {len(y_mimic)}\n"
            f"# events: {int(y_mimic.sum())}\n"
        )
        external_df.to_csv(f, index=False)
    print(f"[predictions] external: {external_path} "
          f"(n={len(external_df)}, events={int(y_mimic.sum())})")

print("\n" + "=" * 70)
print("Done. Next: run make_external_stats.py, then fig2_be.py and figS2_hco3.py.")
print("=" * 70)