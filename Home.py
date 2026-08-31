"""
Home.py — Vital Health Alert XAI
Landing Page with Interactive Clinical Deterioration Sandbox & Dynamic Routing
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Theme import inject_base_css, wave_divider, render_auth_header
from predict_utils import predict_patient

st.set_page_config(
    page_title="Vital Health Alert XAI — Early Deterioration Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_base_css()
render_auth_header()

# Custom Hero Styling
st.markdown(
    """
    <style>
    .hero-container {
        background: linear-gradient(145deg, rgba(6, 44, 56, 0.9) 0%, rgba(4, 27, 36, 0.95) 100%);
        border: 1px solid rgba(2, 195, 154, 0.35);
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 35px rgba(2, 195, 154, 0.12);
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 800;
        color: #02C39A;
        background: rgba(2, 195, 154, 0.15);
        border: 1px solid rgba(2, 195, 154, 0.4);
        border-radius: 999px;
        padding: 0.35rem 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.8rem;
        line-height: 1.15;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #FFFFFF;
    }
    .hero-title .highlight {
        background: linear-gradient(90deg, #00A896, #02C39A, #F0F3BD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-desc {
        color: rgba(245, 250, 248, 0.9);
        font-size: 1.08rem;
        line-height: 1.65;
        margin-bottom: 1.6rem;
    }
    .sandbox-card {
        background: rgba(4, 30, 40, 0.9);
        border: 1px solid rgba(2, 195, 154, 0.35);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.8rem 0;
    }
    .vital-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: rgba(240, 243, 189, 0.1);
        border: 1px solid rgba(240, 243, 189, 0.25);
        border-radius: 8px;
        padding: 0.35rem 0.65rem;
        color: #F0F3BD;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DEMO_PRESETS = {
    "🟢 Normal Baseline": {
        "name": "Elena (Healthy Baseline)",
        "vitals": {
            'Heart Rate': 72, 'Respiratory Rate': 15, 'Body Temperature': 36.8,
            'Oxygen Saturation': 98.5, 'Systolic Blood Pressure': 118, 'Diastolic Blood Pressure': 76,
            'Age': 34, 'Gender': 0, 'Weight (kg)': 62.0, 'Height (m)': 1.68,
            'Derived_HRV': 0.12, 'Derived_Pulse_Pressure': 42, 'Derived_BMI': 21.97, 'Derived_MAP': 90.0
        },
        "description": "Standard resting vitals within clinical reference ranges."
    },
    "🔴 Sepsis Emergency": {
        "name": "David (Severe Sepsis Alert)",
        "vitals": {
            'Heart Rate': 136, 'Respiratory Rate': 28, 'Body Temperature': 40.2,
            'Oxygen Saturation': 89.0, 'Systolic Blood Pressure': 82, 'Diastolic Blood Pressure': 48,
            'Age': 64, 'Gender': 1, 'Weight (kg)': 78.0, 'Height (m)': 1.76,
            'Derived_HRV': 0.03, 'Derived_Pulse_Pressure': 34, 'Derived_BMI': 25.18, 'Derived_MAP': 59.33
        },
        "description": "High fever, profound tachycardia, hypotension, and tachypnea indicative of septic shock."
    },
    "🟠 Hypertensive Crisis": {
        "name": "Sarah (Hypertensive Emergency)",
        "vitals": {
            'Heart Rate': 92, 'Respiratory Rate': 18, 'Body Temperature': 36.8,
            'Oxygen Saturation': 97.0, 'Systolic Blood Pressure': 205, 'Diastolic Blood Pressure': 125,
            'Age': 58, 'Gender': 0, 'Weight (kg)': 82.0, 'Height (m)': 1.65,
            'Derived_HRV': 0.07, 'Derived_Pulse_Pressure': 80, 'Derived_BMI': 30.12, 'Derived_MAP': 151.67
        },
        "description": "Severely elevated SBP > 200 mmHg & DBP > 120 mmHg requiring urgent pressure reduction."
    },
    "🔴 Severe Hypoxia / COPD": {
        "name": "Robert (Acute Hypoxia Flare)",
        "vitals": {
            'Heart Rate': 122, 'Respiratory Rate': 32, 'Body Temperature': 38.1,
            'Oxygen Saturation': 81.5, 'Systolic Blood Pressure': 145, 'Diastolic Blood Pressure': 90,
            'Age': 71, 'Gender': 1, 'Weight (kg)': 70.0, 'Height (m)': 1.72,
            'Derived_HRV': 0.04, 'Derived_Pulse_Pressure': 55, 'Derived_BMI': 23.66, 'Derived_MAP': 108.33
        },
        "description": "Critical oxygen desaturation (81.5%) and severe tachypnea requiring immediate oxygenation."
    }
}


def render_gauge_chart(score, category, color):
    """
    Renders Plotly Gauge for Health Risk Index with explicit key.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{category}</b>", 'font': {'size': 15, 'color': color, 'family': 'Space Grotesk'}},
        number={'suffix': "/100", 'font': {'size': 32, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.4)"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(255,255,255,0.06)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.15)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(2, 195, 154, 0.18)'},
                {'range': [30, 60], 'color': 'rgba(240, 243, 189, 0.18)'},
                {'range': [60, 80], 'color': 'rgba(242, 166, 90, 0.18)'},
                {'range': [80, 100], 'color': 'rgba(232, 84, 62, 0.28)'}
            ],
            'threshold': {'line': {'color': color, 'width': 4}, 'thickness': 0.85, 'value': score}
        }
    ))
    fig.update_layout(
        height=190,
        margin=dict(l=15, r=15, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#FFFFFF"}
    )
    return fig


# ============================================================
# HERO SECTION
# ============================================================
st.markdown('<div class="hero-container">', unsafe_allow_html=True)

hero_col1, hero_col2 = st.columns([1.1, 0.9], gap="large")

with hero_col1:
    st.markdown('<div class="hero-badge">🫀 Clinical Decision Support · Explainable AI</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">Detect Early.<br>Explain Clearly.<br><span class="highlight">Act Confidently.</span></h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-desc">Vital Health Alert XAI combines multi-vital machine learning (LightGBM & Random Forest), '
        'NHS NEWS2 physiological early-warning protocols, and SHAP Explainable AI to identify patient deterioration hours '
        'before critical collapse — providing clinicians and patients with transparent reasoning for every alert.</p>',
        unsafe_allow_html=True
    )

    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.button("🚀 Live Predictor", type="primary", use_container_width=True, key="hero_pred_btn"):
            st.switch_page("pages/2_Predict.py")
    with btn_c2:
        if st.button("🚑 Symptom Check", use_container_width=True, key="hero_symptom_btn"):
            st.switch_page("pages/1_Symptom_Check.py")
    with btn_c3:
        if st.button("🤖 AI Assistant", use_container_width=True, key="hero_ai_btn"):
            st.switch_page("pages/3_AI_Assistant.py")

    st.markdown(
        """
        <div style="margin-top:1.4rem; font-size:0.85rem; color:#FFFFFF; display:flex; align-items:center; gap:12px; font-weight:600;">
            <span>🛡️ Validated on NEWS2 Protocol</span>
            <span>•</span>
            <span>🧠 SHAP Feature Attribution</span>
            <span>•</span>
            <span>🚨 Anomaly Detection</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_col2:
    st.markdown('<div class="sandbox-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#FFFFFF; margin-top:0;'>⚡ Interactive Clinical Sandbox</h4>", unsafe_allow_html=True)
    st.caption("Select a real patient clinical case to see live AI prediction & SHAP explanation:")

    selected_preset = st.selectbox(
        "Select Case Scenario:",
        list(DEMO_PRESETS.keys()),
        index=1,
        label_visibility="collapsed",
        key="home_sandbox_select"
    )

    case_info = DEMO_PRESETS[selected_preset]
    vitals_data = case_info["vitals"]

    # Run real prediction
    res = predict_patient(vitals_data)

    st.plotly_chart(
        render_gauge_chart(res["risk_score"], res["risk_category"], res["risk_color"]),
        use_container_width=True,
        key="home_sandbox_gauge",
        config={'displayModeBar': False}
    )

    # Vitals chips
    st.markdown(
        f"""
        <div class="chip-container">
            <span class="vital-chip">❤ HR: {vitals_data['Heart Rate']} bpm</span>
            <span class="vital-chip">🫁 SpO₂: {vitals_data['Oxygen Saturation']}%</span>
            <span class="vital-chip">🩸 BP: {vitals_data['Systolic Blood Pressure']}/{vitals_data['Diastolic Blood Pressure']}</span>
            <span class="vital-chip">🌡 Temp: {vitals_data['Body Temperature']}°C</span>
            <span class="vital-chip">⏱ RR: {vitals_data['Respiratory Rate']}</span>
            <span class="vital-chip" style="border-color:{res['risk_color']}; color:{res['risk_color']};">NEWS2: {res['news2_score']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(f"<p style='font-size:0.85rem; color:#FFFFFF; margin-bottom:0.8rem;'><i>{case_info['description']}</i></p>", unsafe_allow_html=True)

    if st.button("🔬 Load & Inspect in Full Predictor →", use_container_width=True, key="sandbox_inspect"):
        st.session_state["loaded_preset"] = vitals_data
        st.session_state["loaded_patient_name"] = case_info["name"]
        st.switch_page("pages/2_Predict.py")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

wave_divider()

# ============================================================
# WHY IT MATTERS
# ============================================================
st.markdown('<div class="section-label" style="text-align:center; display:block;">Clinical Significance</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title" style="text-align:center;">Three Pillars of Clinical Early Warning</h2>', unsafe_allow_html=True)

v1, v2, v3 = st.columns(3)
pillars = [
    ("⏱️ Early Deterioration Flagging", "Predicts physiological instability hours before critical alarms trigger, enabling proactive clinical intervention.", "#02C39A"),
    ("🧠 Explainable AI (SHAP)", "No black-box guesses. Shows exact mathematical feature contributions (e.g. SpO₂ drop vs BP spike) for every alert.", "#F0F3BD"),
    ("🏥 Standardized NEWS2 Triaging", "Integrates National Early Warning Score protocols for seamless escalation to ward doctors and ICU response teams.", "#F2A65A")
]

for col, (title, desc, accent_color) in zip([v1, v2, v3], pillars):
    with col:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 3px solid {accent_color};">
                <h4 style="color:{accent_color}; margin-top:0;">{title}</h4>
                <p style="color:#FFFFFF; font-size:0.92rem; line-height:1.6; margin-bottom:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

wave_divider()

# ============================================================
# PIPELINE ARCHITECTURE
# ============================================================
st.markdown('<div class="section-label">End-to-End Pipeline</div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-title">From Raw Vitals to Actionable Clinical Triage</h2>', unsafe_allow_html=True)

p1, a1, p2, a2, p3, a3, p4 = st.columns([2, 0.3, 2, 0.3, 2, 0.3, 2])
pipe_steps = [
    ("01", "Vitals Acquisition", "Multi-parameter inputs (HR, BP, SpO₂, Temp, RR, MAP, BMI, HRV) from wearables or clinical monitors."),
    ("02", "Ensemble ML Scoring", "LightGBM + Random Forest calibrated ensemble trained on 60,000+ clinical deterioration profiles."),
    ("03", "Explainable AI (SHAP)", "TreeExplainer computes precise directional SHAP force values attributing risk to specific biomarkers."),
    ("04", "Clinical Decision Support", "Harmonized Health Risk Index (0-100), Isolation Forest anomaly flags, and NHS NEWS2 escalation pathways.")
]

for col, (num, title, desc) in [(p1, pipe_steps[0]), (p2, pipe_steps[1]), (p3, pipe_steps[2]), (p4, pipe_steps[3])]:
    with col:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-family:'JetBrains Mono'; font-weight:800; color:#02C39A; font-size:1.3rem; margin-bottom:0.4rem;">{num}</div>
                <h4 style="margin-bottom:0.4rem; font-size:1rem; color:#FFFFFF;">{title}</h4>
                <p style="color:#FFFFFF; font-size:0.86rem; line-height:1.55; margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
for a in (a1, a2, a3):
    with a:
        st.markdown('<div style="text-align:center; padding-top:2.2rem; color:#02C39A; font-size:1.4rem; font-weight:700;">→</div>', unsafe_allow_html=True)

wave_divider()

# ============================================================
# FINAL CTA & FOOTER
# ============================================================
st.markdown(
    """
    <div style="text-align:center; background:linear-gradient(135deg, rgba(0,168,150,0.22), rgba(2,195,154,0.12)); border:1px solid rgba(2,195,154,0.4); border-radius:22px; padding:2.5rem 1.8rem; margin:2rem 0;">
        <h2 style="margin-bottom:0.6rem; color:#FFFFFF;">Ready to Run Patient Risk Predictions?</h2>
        <p style="color:#FFFFFF; max-width:580px; margin:0 auto 1.5rem auto; font-size:0.98rem;">
            Experience explainable AI with real-time SHAP feature attribution, anomaly alerts, full patient history trajectory charts, and the Clinical AI Copilot.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

fc1, fc2, fc3 = st.columns([1, 1.2, 1])
with fc2:
    if st.button("🚀 Open Vitals Predictor Now", type="primary", use_container_width=True, key="bottom_cta"):
        st.switch_page("pages/2_Predict.py")

st.markdown(
    """
    <div style="border-top:1px solid rgba(240,243,189,0.18); margin-top:3rem; padding-top:1.5rem; text-align:center; font-size:0.88rem; color:rgba(255,255,255,0.7);">
        <strong style="color:#FFFFFF;">Vital Health Alert XAI</strong> — Explainable AI Clinical Decision Support System<br>
        Academic Research Prototype • Final Year Major Project
    </div>
    """,
    unsafe_allow_html=True
)