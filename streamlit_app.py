# -*- coding: utf-8 -*-
"""
Respiratory Failure In-Hospital Mortality Risk Calculator

Models:
  Primary (BE):      LP = 1.7933 - 0.1256*ALB + 0.0390*BUN - 0.0387*BE
  Sensitivity (HCO3): LP = 3.2028 - 0.1277*ALB + 0.0353*BUN - 0.0523*HCO3_

External validation (MIMIC-IV v3.1):
  BE model:   n=434, 88 deaths, AUC 0.6187 (0.5517-0.6815)
              recalibration: logit(p_new) = 0.4042*logit(p) - 0.6358
  HCO3 model: n=684, 109 deaths, AUC 0.6410 (0.5862-0.6949)
              recalibration: logit(p_new) = 0.4808*logit(p) - 0.1779

DCA: recalibrated model shows positive net benefit vs treat-all
     across threshold probabilities 10%-30% (greatest gain at 15%-20%)

Limitations:
  - Recalibration was apparent (fitted and evaluated on the same external cohort),
    validated by bootstrap OOB; prospective external validation still required.
  - External discrimination is moderate.
  - Research tool only; does not replace clinical judgment.
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
    "and externally validated in MIMIC-IV v3.1, with recalibration for clinical use."
)
st.markdown("---")

# ========== Model coefficients ==========
# Primary model: BE version (original scale)
BE_INTERCEPT = 1.7933
BE_COEF = {
    "ALB": -0.1256,
    "BUN":  0.0390,
    "BE":  -0.0387,
}
# BE recalibration parameters (estimated on the MIMIC-IV external cohort)
BE_RECAL_A = 0.4042
BE_RECAL_B = -0.6358

# Sensitivity model: HCO3- version (original scale, not standardized)
HCO3_INTERCEPT = 3.2028
HCO3_COEF = {
    "ALB":   -0.1277,
    "BUN":    0.0353,
    "HCO3_": -0.0523,
}
# HCO3- recalibration parameters
HCO3_RECAL_A = 0.4808
HCO3_RECAL_B = -0.1779

# ========== Sidebar: model information ==========
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
    "  - AUC 0.619 (95% CI 0.552–0.682)\n"
    "  - Calibration slope 0.404, intercept −0.636 (before recalibration)\n"
    "  - After recalibration: slope 1.000, intercept 0.000; Brier 0.166 → 0.158\n"
    "- **HCO₃⁻ model**: n = 684, 109 deaths\n"
    "  - AUC 0.641 (95% CI 0.586–0.695)\n"
    "  - Calibration slope 0.480, intercept −0.804 (before recalibration)\n"
    "  - After recalibration: slope 1.001, intercept 0.001; Brier 0.143 → 0.131"
)

st.sidebar.markdown("**Clinical utility (DCA)**")
st.sidebar.markdown(
    "- Recalibrated model shows positive net benefit vs treat-all\n"
    "- Useful threshold range: **10%–30%** (greatest gain at 15%–20%)"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Research tool only. Recalibration was apparent (fitted and evaluated "
    "on the same external cohort) and validated by bootstrap OOB; prospective "
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
    """logit(p_new) = a * logit(p) + b"""
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

        # Recalibrated probability as the primary output
        st.markdown(f"**{model_label} · recalibrated probability**")
        st.markdown(
            f"<h1 style='text-align:center; color:#1f77b4;'>{p_recal:.2%}</h1>",
            unsafe_allow_html=True
        )

        # Risk stratification based on the DCA-supported threshold range
        if p_recal < 0.10:
            risk_level = "🟢 Low risk (<10%)"
            risk_note = ("Below the clinically useful threshold range; DCA showed "
                         "no net benefit over treat-all.")
        elif p_recal < 0.20:
            risk_level = "🟡 Moderate risk (10–20%)"
            risk_note = ("Within the DCA-supported range; recalibrated model "
                         "outperforms treat-all.")
        elif p_recal < 0.30:
            risk_level = "🟠 High risk (20–30%)"
            risk_note = ("Within the DCA-supported range; greatest net benefit "
                         "observed here.")
        else:
            risk_level = "🔴 Very high risk (≥30%)"
            risk_note = ("Above 30%; DCA net benefit was marginal in this range, "
                         "interpret with caution.")

        st.markdown(f"**{risk_level}**")
        st.caption(risk_note)

        # Original vs recalibrated probability
        with st.expander("🔍 Original vs recalibrated probability"):
            st.markdown(
                f"- **Original model (before recalibration):** {p_original:.2%}\n"
                f"- **Recalibrated model:** {p_recal:.2%}\n"
                f"- **Difference:** {(p_recal - p_original)*100:+.2f} percentage points"
            )
            st.caption(
                "The original model systematically overestimated mortality in the "
                "MIMIC-IV external cohort (calibration intercept −0.64). "
                "The recalibrated probability is the recommended output for clinical use."
            )

        # Researcher details
        with st.expander("📐 Researcher details"):
            a = BE_RECAL_A if use_be else HCO3_RECAL_A
            b = BE_RECAL_B if use_be else HCO3_RECAL_B
            sign = "+" if b >= 0 else "−"
            st.markdown(
                f"- Linear predictor (LP): **{lp:.4f}**\n"
                f"- Original probability: **{p_original:.6f}**\n"
                f"- Recalibrated probability: **{p_recal:.6f}**\n"
                f"- Recalibration formula: "
                f"`logit(p_new) = {a:.4f} × logit(p) {sign} {abs(b):.4f}`"
            )

st.markdown("---")

# ========== Footer ==========
st.caption(
    "📌 **Model**: Logistic regression, 3 predictors (ALB + BUN + BE), developed "
    "on 2,028 patients and externally validated in MIMIC-IV v3.1 (n = 434)."
)
st.caption(
    "📌 **External validation**: AUC 0.619 (95% CI 0.552–0.682); calibration slope "
    "0.404, intercept −0.636 before recalibration; after a single logistic "
    "recalibration, slope 1.000 and intercept 0.000 (bootstrap OOB confirmed, "
    "difference < 0.002)."
)
st.caption(
    "📌 **Clinical utility**: Decision curve analysis showed higher net benefit "
    "than treat-all across threshold probabilities of 10%–30% after recalibration."
)
st.caption(
    "📌 **Limitations**: Recalibration was apparent (fitted and evaluated on the "
    "same external cohort); prospective external validation is required. External "
    "discrimination is moderate. This tool is for clinical reference only and does "
    "not replace clinical judgment."
)
st.caption(
    "📌 **Sensitivity analysis (HCO₃⁻ model)**: n = 684, AUC 0.641 (95% CI "
    "0.586–0.695); after recalibration, positive net benefit at 10%–20% thresholds."
)