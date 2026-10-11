# -*- coding: utf-8 -*-
"""
Respiratory Failure In-Hospital Mortality Risk Calculator
呼吸衰竭院内死亡风险计算器

模型来源：
  主模型（BE 版）：LP = 1.7933 − 0.1256*ALB + 0.0390*BUN − 0.0387*BE
    - 开发队列：单中心 2,028 例（事件 226）
    - 外部验证：MIMIC-IV v3.1，n=434，事件 88
    - 外部 AUC 0.6187 (95% CI 0.5517–0.6815)
    - 原始校准斜率 0.4042，截距 −0.6358（系统性高估风险）
    - 重校准：logit(p_new) = 0.4042 * logit(p) − 0.6358
      → 重校准后斜率 1.000、截距 0.000，Brier 0.166→0.158

  敏感性模型（HCO3- 版）：LP = 3.2028 − 0.1277*ALB + 0.0353*BUN − 0.0523*HCO3_
    - 原始尺度系数（非标准化）
    - 外部验证：MIMIC-IV v3.1，n=684，事件 109
    - 外部 AUC 0.6410 (95% CI 0.5862–0.6949)
    - 原始校准斜率 0.4799，截距 −0.8043
    - 重校准：logit(p_new) = 0.4808 * logit(p) − 0.1779
      → 重校准后斜率 1.001、截距 0.001，Brier 0.143→0.131

DCA 结果（重校准后净获益 vs 全部治疗）：
  BE 版：10%/15%/20%/30% 四阈值均优于 treat-all
  HCO3- 版：10%/15%/20% 三阈值优于 treat-all，30% 略负
  临床有用区间：10%–30%（重点 15%–20%）

⚠️ 局限性：
  - 重校准为表观重校准（同一外部队列拟合+评估），虽经 bootstrap OOB 验证，
    但前瞻性外部验证仍是必要的
  - 外部 AUC 0.62–0.64，区分度中等
  - 本工具仅供临床参考，不替代临床判断
"""

import streamlit as st
import numpy as np

# ========== 页面设置 ==========
st.set_page_config(
    page_title="Respiratory Failure Mortality Risk Calculator",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Respiratory Failure In-Hospital Mortality Risk Calculator")
st.caption("呼吸衰竭院内死亡风险计算器 · 基于单中心开发 + MIMIC-IV 外部验证 + 重校准")
st.markdown("---")

# ========== 模型系数 ==========
# --- 主模型：BE 版（原始尺度） ---
BE_INTERCEPT = 1.7933
BE_COEF = {
    "ALB": -0.1256,
    "BUN": 0.0390,
    "BE":  -0.0387,
}
# BE 版重校准参数（来自 MIMIC-IV 外部队列）
BE_RECAL_A = 0.4042
BE_RECAL_B = -0.6358

# --- 敏感性模型：HCO3- 版（原始尺度，非标准化） ---
HCO3_INTERCEPT = 3.2028
HCO3_COEF = {
    "ALB":   -0.1277,
    "BUN":    0.0353,
    "HCO3_": -0.0523,
}
# HCO3- 版重校准参数
HCO3_RECAL_A = 0.4808
HCO3_RECAL_B = -0.1779

# ========== 侧边栏：模型信息 ==========
st.sidebar.header("📊 Model Information")

st.sidebar.markdown("**Development cohort**")
st.sidebar.markdown(
    "- Single tertiary center, China\n"
    "- n = 2,028 (226 deaths, 11.1%)\n"
    "- Predictors selected by LASSO across 50 imputed datasets"
)

st.sidebar.markdown("**External validation (MIMIC-IV v3.1)**")
st.sidebar.markdown(
    "- BE model: n = 434, 88 deaths\n"
    "  - AUC **0.619** (95% CI 0.552–0.682)\n"
    "  - Calibration slope **0.404**, intercept **−0.636**\n"
    "  - After recalibration: slope 1.000, intercept 0.000, Brier 0.166→0.158\n"
    "- HCO₃⁻ model: n = 684, 109 deaths\n"
    "  - AUC **0.641** (95% CI 0.586–0.695)\n"
    "  - Calibration slope **0.480**, intercept **−0.804**\n"
    "  - After recalibration: slope 1.001, intercept 0.001, Brier 0.143→0.131"
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

# ========== 模型选择 ==========
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

# ========== 输入界面 ==========
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

# ========== 预测函数 ==========
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

# ========== 结果显示 ==========
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

        # 用重校准后的概率作为主输出
        st.markdown(f"**{model_label} · recalibrated probability**")
        st.markdown(
            f"<h1 style='text-align:center; color:#1f77b4;'>{p_recal:.2%}</h1>",
            unsafe_allow_html=True
        )

        # 风险分层（基于 DCA 的临床有用区间）
        if p_recal < 0.10:
            risk_level = "🟢 Low risk (<10%)"
            risk_note = "Below the clinically useful threshold range; DCA showed no net benefit over treat-all."
        elif p_recal < 0.20:
            risk_level = "🟡 Moderate risk (10–20%)"
            risk_note = "Within the DCA-supported range; recalibrated model outperforms treat-all."
        elif p_recal < 0.30:
            risk_level = "🟠 High risk (20–30%)"
            risk_note = "Within the DCA-supported range; greatest net benefit observed here."
        else:
            risk_level = "🔴 Very high risk (≥30%)"
            risk_note = "Above 30%; DCA net benefit was marginal in this range, interpret with caution."

        st.markdown(f"**{risk_level}**")
        st.caption(risk_note)

        # 原始 vs 重校准对比
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

        # 线性预测值（研究者信息）
        with st.expander("📐 Researcher details"):
            st.markdown(
                f"- Linear predictor (LP): **{lp:.4f}**\n"
                f"- Original probability: **{p_original:.6f}**\n"
                f"- Recalibrated probability: **{p_recal:.6f}**\n"
                f"- Recalibration formula: "
                f"`logit(p_new) = {BE_RECAL_A if use_be else HCO3_RECAL_A:.4f} × logit(p) "
                f"{'+' if (BE_RECAL_B if use_be else HCO3_RECAL_B) >= 0 else '−'} "
                f"{abs(BE_RECAL_B if use_be else HCO3_RECAL_B):.4f}`"
            )

st.markdown("---")

# ========== 页脚 ==========
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
