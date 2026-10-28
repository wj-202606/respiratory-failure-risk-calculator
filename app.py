# -*- coding: utf-8 -*-
"""
Respiratory Failure In-Hospital Mortality Risk Calculator (v4.3)

Models:
  Primary (BE):      LP = 1.7825 - 0.1227*ALB + 0.0343*BUN - 0.0585*BE
  Sensitivity (HCO3): LP = 3.1936 - 0.1209*ALB + 0.0374*BUN - 0.0604*HCO3_

External validation (MIMIC-IV v3.1):
  BE model:   n=434, 88 deaths, AUC 0.622 (0.553-0.682)
              recalibration: logit(p_new) = 0.4196*logit(p) - 0.6172
  HCO3 model: n=684, 109 deaths, AUC 0.642 (0.588-0.697)
              recalibration: logit(p_new) = 0.4856*logit(p) - 0.8090

eICU-CRD validation (ICU-admission time origin, Plan B):
  BE model:   n=1,458, 363 deaths, AUC 0.670 (0.638-0.703)
              recalibration: logit(p_new) = 0.5993*logit(p) - 0.4863
  HCO3 model: n=2,287, 521 deaths, AUC 0.670 (0.643-0.696)
              recalibration: logit(p_new) = 0.6189*logit(p) - 0.5914

DCA:
  Recalibrated BE model: positive net benefit vs treat-all at 15-30% in MIMIC-IV
    and 20-30% in eICU-CRD; at 10% the difference is negligible in MIMIC-IV
    and slightly below treat-all in eICU-CRD.
  Recalibrated HCO3 model: positive net benefit vs treat-all at 10% in MIMIC-IV,
    and 15-30% in eICU-CRD.

Note:
  The calculator displays the original and MIMIC-IV-recalibrated probabilities.
  eICU-CRD recalibration parameters are reported in the manuscript (Table 7) 
  and are not implemented here.
"""

import streamlit as st
import numpy as np

# ========== Page configuration ==========
st.set_page_config(
    page_title="Respiratory Failure Mortality Risk Calculator",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Respiratory Failure In-Hospital Mortality Risk Calculator")
st.caption(
    "A 3-variable logistic model developed at a single center in China "
    "and externally evaluated in MIMIC-IV v3.1 and eICU-CRD v2.0. "
    "This calculator is a research-use tool, not a validated clinical device."
)
st.markdown("---")

# ========== Model coefficients ==========
# Primary model: BE version (original scale)
BE_INTERCEPT = 1.7825
BE_COEF = {
    "ALB": -0.1227,
    "BUN":  0.0343,
    "BE":  -0.0585,
}
# BE recalibration parameters (MIMIC-IV)
BE_RECAL_A = 0.4196
BE_RECAL_B = -0.6172

# Sensitivity model: HCO3- version (original scale, 50-imputation pooled)
HCO3_INTERCEPT = 3.1936
HCO3_COEF = {
    "ALB":   -0.1209,
    "BUN":    0.0374,
    "HCO3_": -0.0604,
}
# HCO3- recalibration parameters
HCO3_RECAL_A = 0.4856
HCO3_RECAL_B = -0.8090

# ========== Sidebar ==========
st.sidebar.header("📊 Model Information")

st.sidebar.markdown("**Development cohort**")
st.sidebar.markdown(
    "- Single tertiary center, China\n"
    "- n = 2,028 (226 deaths, 11.1%)\n"
    "- Predictors selected by LASSO across 50 imputed datasets"
)

st.sidebar.markdown("**External validation (MIMIC-IV v3.1)**")
st.sidebar.markdown(
    "- **BE model**: n = 434, 88 deaths\n"
    "  - AUC 0.622 (95% CI 0.553–0.682)\n"
    "  - Calibration slope 0.420, intercept −0.617\n"
    "  - After recalibration: slope 0.999, intercept −0.001; Brier 0.166 → 0.158\n"
    "- **HCO₃⁻ model**: n = 684, 109 deaths\n"
    "  - AUC 0.642 (95% CI 0.588–0.697)\n"
    "  - Calibration slope 0.486, intercept −0.809\n"
    "  - After recalibration: slope 1.002, intercept 0.004; Brier 0.137 → 0.131"
)

st.sidebar.markdown("**External evaluation (eICU-CRD v2.0)**")
st.sidebar.markdown(
    "- **BE model**: n = 1,458, 363 deaths\n"
    "  - AUC 0.670 (95% CI 0.638–0.703)\n"
    "  - Calibration slope 0.599, intercept −0.486\n"
    "  - After recalibration: Brier 0.179 → 0.174\n"
    "- **HCO₃⁻ model**: n = 2,287, 521 deaths\n"
    "  - AUC 0.670 (95% CI 0.643–0.696)\n"
    "  - Calibration slope 0.619, intercept −0.591\n"
    "  - After recalibration: Brier 0.171 → 0.164\n"
    "- *Note: The eICU-CRD cohort uses ICU admission as the time origin.*"
)

st.sidebar.markdown("**Clinical utility (DCA, exploratory)**")
st.sidebar.markdown(
    "- Recalibrated **BE model** shows positive net benefit vs treat-all at **15%–30%** in MIMIC-IV and **20%–30%** in eICU-CRD.\n"
    "- Recalibrated **HCO₃⁻ model** shows positive net benefit vs treat-all at **10%** in MIMIC-IV and **15%–30%** in eICU-CRD.\n"
    "- *The net-benefit findings were not uniform across cohorts and thresholds.*"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Research tool only. Recalibration was apparent (fitted and evaluated "
    "on the same external cohort) and supported by bootstrap OOB; prospective "
    "external validation is still required before routine clinical use."
)

# ========== Model selection ==========
st.subheader("1️⃣ Select model")
model_choice = st.radio(
    "Choose the acid–base marker available at your institution:",
    options=[
        "BE model (Base Excess) — primary model",
        "HCO₃⁻ model (Bicarbonate) — sensitivity analysis",
    ],
    index=0,
    help=(
        "BE is the LASSO-selected marker in the primary model, but is not "
        "routinely measured at all centers. HCO₃⁻ has higher availability "
        "(93.7% vs 49.0% in MIMIC-IV) and comparable external discrimination."
    ),
)
use_be = model_choice.startswith("BE")

st.markdown("---")

# ========== Input ==========
st.subheader("2️⃣ Enter patient data")

col1, col2 = st.columns(2)

with col1:
    alb = st.number_input(
        "Albumin ALB (g/L)",
        min_value=5.0, max_value=60.0, value=35.0, step=0.5,
        help="Normal range: 40–55 g/L"
    )
    bun = st.number_input(
        "Blood urea nitrogen BUN (mmol/L)",
        min_value=0.5, max_value=80.0, value=7.0, step=0.5,
        help="Normal range: 3.1–8.0 mmol/L"
    )

    if use_be:
        be = st.number_input(
            "Base excess BE (mmol/L)",
            min_value=-40.0, max_value=40.0, value=0.0, step=0.5,
            help="Normal range: −2 to +3 mmol/L"
        )
        hco3 = None
    else:
        hco3 = st.number_input(
            "Bicarbonate HCO₃⁻ (mmol/L)",
            min_value=5.0, max_value=50.0, value=24.0, step=0.5,
            help="Normal range: 22–29 mmol/L"
        )
        be = None

# ========== Prediction functions ==========
def linear_predictor_be(alb, bun, be):
    return BE_INTERCEPT + BE_COEF["ALB"] * alb + BE_COEF["BUN"] * bun + BE_COEF["BE"] * be

def linear_predictor_hco3(alb, bun, hco3):
    return (HCO3_INTERCEPT + HCO3_COEF["ALB"] * alb
            + HCO3_COEF["BUN"] * bun + HCO3_COEF["HCO3_"] * hco3)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def recalibrate(p, a, b):
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    logit_p = np.log(p / (1 - p))
    return sigmoid(a * logit_p + b)

# ========== Output ==========
with col2:
    st.subheader("3️⃣ Predicted mortality")

    if st.button("🔮 Calculate Risk", type="primary", use_container_width=True):
        if use_be:
            lp = linear_predictor_be(alb, bun, be)
            p_original = sigmoid(lp)
            p_recal = recalibrate(p_original, BE_RECAL_A, BE_RECAL_B)
            model_label = "BE model"
        else:
            lp = linear_predictor_hco3(alb, bun, hco3)
            p_original = sigmoid(lp)
            p_recal = recalibrate(p_original, HCO3_RECAL_A, HCO3_RECAL_B)
            model_label = "HCO₃⁻ model"

        st.markdown(f"**{model_label} · recalibrated probability**")
        st.markdown(
            f"<h1 style='text-align:center; color:#1f77b4;'>{p_recal:.2%}</h1>",
            unsafe_allow_html=True
        )

        if p_recal < 0.10:
            risk_level = "🟢 Low risk (<10%)"
            risk_note = ("Below the exploratory DCA range; the model showed no "
                         "net benefit over treat-all in this range.")
        elif p_recal < 0.20:
            risk_level = "🟡 Moderate risk (10–20%)"
            risk_note = ("Within the exploratory DCA range; the recalibrated model "
                         "showed higher net benefit than treat-all at 15%–20% in MIMIC-IV. "
                         "Whether this translates into changes in management has "
                         "not been prospectively evaluated.")
        elif p_recal < 0.30:
            risk_level = "🟠 High risk (20–30%)"
            risk_note = ("Within the exploratory DCA range; net benefit in this "
                         "range is smaller than at 15%–20%, and for the HCO₃⁻ "
                         "model it is slightly lower than the original model.")
        else:
            risk_level = "🔴 Very high risk (≥30%)"
            risk_note = ("Above 30%; DCA net benefit was marginal or negative in "
                         "this range; interpret with caution.")

        st.markdown(f"**{risk_level}**")
        st.caption(risk_note)

        with st.expander("🔍 Original vs recalibrated probability"):
            st.markdown(
                f"- **Original model (before recalibration):** {p_original:.2%}\n"
                f"- **Recalibrated model:** {p_recal:.2%}\n"
                f"- **Difference:** {(p_recal - p_original)*100:+.2f} percentage points"
            )
            st.caption(
                "In the MIMIC-IV external cohort, the original model produced "
                "overly extreme predictions and a substantial departure from ideal "
                "calibration (jointly estimated calibration intercept: BE model −0.62; "
                "HCO₃⁻ model −0.81). The recalibrated probability is shown for "
                "research purposes only; it has not been prospectively validated "
                "for clinical decision-making."
            )

        with st.expander("📐 Researcher details"):
            a = BE_RECAL_A if use_be else HCO3_RECAL_A
            b = BE_RECAL_B if use_be else HCO3_RECAL_B
            sign = "+" if b >= 0 else "−"
            st.markdown(
                f"- Linear predictor (LP): **{lp:.4f}**\n"
                f"- Original probability: **{p_original:.6f}**\n"
                f"- Recalibrated probability: **{p_recal:.6f}**\n"
                f"- Recalibration formula (MIMIC-IV): "
                f"`logit(p_new) = {a:.4f} × logit(p) {sign} {abs(b):.4f}`"
            )

st.markdown("---")

# ========== Footer ==========
st.caption(
    "📌 **Model**: Logistic regression, 3 predictors (ALB + BUN + BE), developed "
    "on 2,028 patients and externally evaluated in MIMIC-IV v3.1 (n = 434) and eICU-CRD v2.0 (n = 1,458)."
)
st.caption(
    "📌 **External validation (MIMIC-IV, BE)**: AUC 0.622 (95% CI 0.553–0.682); calibration "
    "slope 0.420, intercept −0.617 before recalibration; after two-parameter "
    "recalibration, slope 0.999 and intercept −0.001."
)
st.caption(
    "📌 **External evaluation (eICU-CRD, BE)**: AUC 0.670 (95% CI 0.638–0.703); calibration "
    "slope 0.599, intercept −0.486 before recalibration; after two-parameter "
    "recalibration, Brier score decreased from 0.179 to 0.174."
)
st.caption(
    "📌 **Clinical utility (BE)**: Decision curve analysis showed higher net benefit "
    "than treat-all at threshold probabilities of 15%–30% after recalibration "
    "in MIMIC-IV (at 10% the difference was negligible) and 20%–30% in eICU-CRD. "
    "These thresholds are exploratory and were not derived from a validated "
    "clinical decision pathway."
)
st.caption(
    "📌 **Sensitivity analysis (HCO₃⁻ model)**: MIMIC-IV (n=684) AUC 0.642; "
    "eICU-CRD (n=2,287) AUC 0.670. Net-benefit findings were not uniform across "
    "cohorts and thresholds."
)
st.caption(
    "📌 **Limitations**: Recalibration was apparent (fitted and evaluated on the "
    "same external cohort); prospective external validation is required. External "
    "discrimination is modest. This tool is for research use only and does "
    "not replace clinical judgment."
)