import streamlit as st
import numpy as np

# ========== 页面设置 ==========
st.set_page_config(
    page_title="呼吸衰竭院内死亡风险预测",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 呼吸衰竭患者院内死亡风险预测")
st.markdown("---")

# ========== 模型信息（侧边栏） ==========
st.sidebar.header("📊 模型信息")
st.sidebar.markdown("""
- **模型**: Logistic回归
- **AUC**: 0.735 (95% CI: 0.678-0.790)
- **验证集**: 668例
- **预测变量**: 6个
""")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ 本工具仅供临床参考，请结合患者具体情况综合判断")

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
    st.subheader("📋 输入患者指标")
    
    alb = st.number_input(
        "白蛋白 ALB (g/L)",
        min_value=5.0, max_value=60.0, value=35.0, step=0.5,
        help="正常范围: 35-55 g/L"
    )
    
    dd = st.number_input(
        "D-二聚体 DD (mg/L)",
        min_value=0.0, max_value=100.0, value=1.0, step=0.1,
        help="正常范围: < 0.5 mg/L"
    )
    
    aptt = st.number_input(
        "活化部分凝血活酶时间 APTT (秒)",
        min_value=10.0, max_value=150.0, value=35.0, step=0.5,
        help="正常范围: 25-35秒"
    )
    
    bun = st.number_input(
        "血尿素氮 BUN (mmol/L)",
        min_value=0.0, max_value=80.0, value=7.0, step=0.5,
        help="正常范围: 3.2-7.1 mmol/L"
    )
    
    be = st.number_input(
        "碱剩余 BE (mmol/L)",
        min_value=-40.0, max_value=40.0, value=0.0, step=0.5,
        help="正常范围: -2 到 +2 mmol/L"
    )
    
    lac = st.number_input(
        "乳酸 LAC (mmol/L)",
        min_value=0.0, max_value=20.0, value=2.0, step=0.1,
        help="正常范围: 0.5-2.0 mmol/L"
    )

# ========== 预测计算 ==========
def calculate_risk(alb, dd, aptt, bun, be, lac):
    """计算院内死亡预测概率"""
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
    st.subheader("📊 预测结果")
    
    if st.button("🔮 计算死亡风险", type="primary", use_container_width=True):
        prob = calculate_risk(alb, dd, aptt, bun, be, lac)
        
        if prob < 0.1:
            risk_level = "🟢 低风险"
            risk_color = "green"
        elif prob < 0.3:
            risk_level = "🟡 中风险"
            risk_color = "orange"
        elif prob < 0.5:
            risk_level = "🟠 高风险"
            risk_color = "orange"
        else:
            risk_level = "🔴 极高风险"
            risk_color = "red"
        
        st.markdown("### 预测死亡概率")
        st.markdown(
            f"<h1 style='text-align: center; color: {risk_color};'>"
            f"{prob:.2%}"
            f"</h1>",
            unsafe_allow_html=True
        )
        
        st.markdown("### 风险分层")
        st.markdown(
            f"<h2 style='text-align: center;'>{risk_level}</h2>",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("📌 本工具基于Logistic回归模型，已通过内部验证（Bootstrap 1000次）")
st.caption("📌 AUC = 0.735 (95% CI: 0.678-0.790)，校准良好")
st.caption("📌 预测变量: ALB, DD, APTT, BUN, BE, LAC")