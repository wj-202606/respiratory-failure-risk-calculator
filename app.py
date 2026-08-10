import streamlit as st
import numpy as np

# ========== 页面设置 ==========
st.set_page_config(
    page_title="Respiratory Failure In-Hospital Mortality Risk Calculator",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Respiratory Failure In-Hospital Mortality Risk Calculator")
st.markdown("---")

# ========== Model Information（侧边栏） ==========
st.sidebar.header("📊 Model Information")
st.sidebar.markdown("""
- **Model**: Logistic Regression
- **AUC**: 0.735 (95% CI: 0.678-0.790)
- **Validation Set**: 668cases
- **Predictors**: 6variables
""")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ This tool is for clinical reference only; please make a comprehensive assessment based on the specific condition of each patient.")

# ========== 模型系数（使用 R 列线图的实际系数） ==========
INTERCEPT = -0.84589834
COEFFICIENTS = {
    'ALB': -0.08250486,
    'DD': 0.03643274,
    'APTT': 0.02809351,
    'BUN': 0.02673668,
    'BE': -0.03161960,
    'LAC': 0.09061898
}

# ========== 输入界面 ==========
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Enter patient data")
    
    alb = st.number_input(
        "Albumin ALB (g/L)",
        min_value=5.0, max_value=60.0, value=35.0, step=0.5,
        help="normal range: 35-55 g/L"
    )
    
    dd = st.number_input(
        "D-dimer DD (mg/L)",
        min_value=0.0, max_value=100.0, value=1.0, step=0.1,
        help="normal range: < 0.5 mg/L"
    )
    
    aptt = st.number_input(
        "Activated partial thromboplastin time APTT (秒)",
        min_value=10.0, max_value=150.0, value=35.0, step=0.5,
        help="normal range: 25-35秒"
    )
    
    bun = st.number_input(
        "Blood urea nitrogen BUN (mmol/L)",
        min_value=0.0, max_value=80.0, value=7.0, step=0.5,
        help="normal range: 3.2-7.1 mmol/L"
    )
    
    be = st.number_input(
        "Buffuer excess BE (mmol/L)",
        min_value=-40.0, max_value=40.0, value=0.0, step=0.5,
        help="normal range: -2 到 +2 mmol/L"
    )
    
    lac = st.number_input(
        "Lactate LAC (mmol/L)",
        min_value=0.0, max_value=20.0, value=2.0, step=0.1,
        help="normal range: 0.5-2.0 mmol/L"
    )

# ========== 预测计算 ==========
def calculate_risk(alb, dd, aptt, bun, be, lac):
    """Calculate Risk"""
    linear_pred = (
        INTERCEPT +
        COEFFICIENTS['ALB'] * alb +
        COEFFICIENTS['DD'] * dd +
        COEFFICIENTS['APTT'] * aptt +
        COEFFICIENTS['BUN'] * bun +
        COEFFICIENTS['BE'] * be +
        COEFFICIENTS['LAC'] * lac
    )
    probability = 1 / (1 + np.exp(-linear_pred))
    return probability

with col2:
    st.subheader("📊 Predicted Mortality Probability")
    
    if st.button("🔮 Calculate Risk", type="primary", use_container_width=True):
        prob = calculate_risk(alb, dd, aptt, bun, be, lac)
        
        if prob < 0.1:
            risk_level = "🟢 Low Risk"
            risk_color = "green"
        elif prob < 0.3:
            risk_level = "🟡 Moderate Risk"
            risk_color = "orange"
        elif prob < 0.5:
            risk_level = "🟠 High Risk"
            risk_color = "orange"
        else:
            risk_level = "🔴 Very High Risk"
            risk_color = "red"
        
        st.markdown("### Predicted Mortality Probability")
        st.markdown(
            f"<h1 style='text-align: center; color: {risk_color};'>"
            f"{prob:.2%}"
            f"</h1>",
            unsafe_allow_html=True
        )
        
        st.markdown("### Risk Stratification")
        st.markdown(
            f"<h2 style='text-align: center;'>{risk_level}</h2>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("📌 This tool is based on a logistic regression model and has undergone internal validation (Bootstrap, 1,000 iterations).")
st.caption("📌 AUC = 0.735 (95% CI: 0.678-0.790)，calibration is good")
st.caption("📌 predictor variable: ALB, DD, APTT, BUN, BE, LAC")
