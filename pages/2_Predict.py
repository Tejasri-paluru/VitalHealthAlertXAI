"""
2_Predict.py — Clinical Intelligence & Explainable AI Prediction Platform
Vital Health Alert XAI

Features:
- Dual Modes:
  1. Advanced Clinical Decision Support (Full parameters, 1-click presets, SHAP XAI, NEWS2, Anomaly)
  2. Smart Patient Health Check (User-friendly wizard, wearable smartwatch data, dynamic symptom estimation)
  3. Patient Deterioration History & Trajectories (Pre-seeded benchmark cases & time-series trend graphs)
- Fully unique Plotly chart keys to prevent StreamlitDuplicateElementId errors.
- High-contrast, pure white typography and parameter labels.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Theme import inject_base_css, wave_divider, render_auth_header
from predict_utils import (
    predict_patient, save_prediction, get_prediction_history, get_patient_timeline,
    estimate_vitals_from_symptoms
)
from auth_utils import save_user_vitals

st.set_page_config(
    page_title="Vitals Predictor & XAI — Vital Health Alert",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_base_css()
render_auth_header()

# Header banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(6, 44, 56, 0.85) 0%, rgba(4, 27, 36, 0.95) 100%); border: 1px solid rgba(2, 195, 154, 0.25); border-radius: 20px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="section-label">Explainable Clinical Decision Support</div>
        <h1 style="margin: 0.2rem 0 0.5rem 0; font-size:2.2rem;">🏥 Vital Health Predictor & Explainable AI</h1>
        <p style="color:rgba(245,250,248,0.72); margin:0; font-size:0.95rem;">
            Early patient deterioration detection powered by LightGBM, Random Forest, NHS NEWS2 clinical scoring,
            SHAP feature attribution, and Isolation Forest anomaly analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Clinical Presets
CLINICAL_PRESETS = {
    "🟢 Healthy Baseline": {
        "name": "Elena Rostova (Wellness Check)",
        "hr": 72, "rr": 15, "temp": 36.8, "spo2": 98.5,
        "sbp": 118, "dbp": 76, "age": 34, "gender": "Female",
        "w": 62.0, "h": 1.68, "desc": "Optimal baseline vitals with zero clinical warning flags."
    },
    "🔴 Sepsis Emergency": {
        "name": "David Miller (ICU Sepsis Case)",
        "hr": 136, "rr": 28, "temp": 40.2, "spo2": 89.0,
        "sbp": 82, "dbp": 48, "age": 64, "gender": "Male",
        "w": 78.0, "h": 1.76, "desc": "Septic shock state: High fever (40.2°C), severe tachycardia (136 bpm), hypotension (82/48), hypoxemia (89%)."
    },
    "🟠 Hypertensive Emergency": {
        "name": "Sarah Jenkins (Hypertensive Crisis)",
        "hr": 92, "rr": 18, "temp": 36.8, "spo2": 97.0,
        "sbp": 205, "dbp": 125, "age": 58, "gender": "Female",
        "w": 82.0, "h": 1.65, "desc": "Stage 3 Hypertensive Crisis with SBP 205 / DBP 125 mmHg requiring rapid pressure reduction."
    },
    "🔴 Acute Hypoxia / COPD": {
        "name": "Robert Chen (COPD Flare-up)",
        "hr": 122, "rr": 32, "temp": 38.1, "spo2": 81.5,
        "sbp": 145, "dbp": 90, "age": 71, "gender": "Male",
        "w": 70.0, "h": 1.72, "desc": "Critical respiratory desaturation (SpO2 81.5%) with severe tachypnea (32 breaths/min)."
    },
    "🟡 Post-Op Cardiac Tachycardia": {
        "name": "Maria Gonzalez (Post-Op Cardiac)",
        "hr": 142, "rr": 24, "temp": 37.8, "spo2": 93.5,
        "sbp": 150, "dbp": 95, "age": 60, "gender": "Female",
        "w": 68.0, "h": 1.64, "desc": "Post-surgical tachyarrhythmia with elevated heart rate (142 bpm) and mild hypoxemia."
    }
}


def render_risk_gauge(score, category, color):
    """
    Renders Plotly circular gauge for Health Risk Index.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{category}</b>", 'font': {'size': 16, 'color': color, 'family': 'Space Grotesk'}},
        number={'suffix': "/100", 'font': {'size': 38, 'color': '#FFFFFF', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.4)"},
            'bar': {'color': color, 'thickness': 0.32},
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
        height=220,
        margin=dict(l=15, r=15, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#F5FAF8"}
    )
    return fig


def render_shap_bar_chart(explanation_df):
    """
    Renders interactive SHAP attribution bar chart.
    """
    top_df = explanation_df.head(6).copy()
    top_df['Color'] = top_df['Contribution'].apply(lambda x: '#E8543E' if x > 0 else '#02C39A')
    top_df = top_df.iloc[::-1]

    fig = go.Figure(go.Bar(
        x=top_df['Contribution'],
        y=top_df['Feature'],
        orientation='h',
        marker=dict(color=top_df['Color'], line=dict(width=1, color='rgba(255,255,255,0.2)')),
        text=top_df['Contribution'].apply(lambda x: f"{x:+.2f}"),
        textposition='outside',
        textfont=dict(family='JetBrains Mono', color='#FFFFFF', size=11)
    ))

    fig.update_layout(
        title="<b>SHAP Feature Attribution (Why this prediction?)</b>",
        title_font=dict(size=14, family='Space Grotesk', color='#FFFFFF'),
        xaxis_title="SHAP Impact Value (Directional Risk Contribution)",
        xaxis=dict(gridcolor='rgba(255,255,255,0.08)', zerolinecolor='rgba(255,255,255,0.3)', tickfont=dict(color='#FFFFFF')),
        yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(color='#FFFFFF', size=12)),
        height=260,
        margin=dict(l=10, r=40, t=40, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', family='Inter')
    )
    return fig


# Top Tabs
tab_clinician, tab_patient, tab_history = st.tabs([
    "🏥 Advanced Clinical Decision Support",
    "👤 Smart Patient Health Check",
    "📈 Patient Deterioration History & Trajectories"
])


# ============================================================
# TAB 1: CLINICAL DECISION SUPPORT
# ============================================================
with tab_clinician:
    st.markdown("### 🔬 Advanced Clinical Decision Support")
    st.caption("Select a standardized clinical scenario or adjust individual physiological biomarkers:")

    # One-click preset scenario buttons
    st.markdown("<p style='color:#FFFFFF; font-weight:700; margin-bottom:0.4rem;'>⚡ Quick Clinical Case Scenarios:</p>", unsafe_allow_html=True)
    preset_cols = st.columns(len(CLINICAL_PRESETS))
    
    if "loaded_preset" in st.session_state:
        default_preset = st.session_state.pop("loaded_preset")
        default_name = st.session_state.pop("loaded_patient_name", "Anonymous")
    else:
        default_preset = None
        default_name = "Anonymous"

    for col, (preset_title, p_data) in zip(preset_cols, CLINICAL_PRESETS.items()):
        with col:
            if st.button(preset_title, use_container_width=True, key=f"btn_preset_{preset_title}"):
                st.session_state["active_preset"] = p_data

    active = st.session_state.get("active_preset", CLINICAL_PRESETS["🟢 Healthy Baseline"])
    if default_preset:
        active = {
            "name": default_name,
            "hr": default_preset.get("Heart Rate", 72),
            "rr": default_preset.get("Respiratory Rate", 15),
            "temp": default_preset.get("Body Temperature", 36.8),
            "spo2": default_preset.get("Oxygen Saturation", 98.5),
            "sbp": default_preset.get("Systolic Blood Pressure", 118),
            "dbp": default_preset.get("Diastolic Blood Pressure", 76),
            "age": default_preset.get("Age", 34),
            "gender": "Male" if default_preset.get("Gender", 1) == 1 else "Female",
            "w": default_preset.get("Weight (kg)", 70.0),
            "h": default_preset.get("Height (m)", 1.75),
            "desc": "Custom loaded scenario"
        }

    st.info(f"📋 **Selected Patient Case:** **{active['name']}** — *{active['desc']}*")

    # Patient Details and Vitals Input Form
    with st.expander("🛠️ Customize Physiological Parameters & Biomarkers", expanded=True):
        f_c1, f_c2, f_c3 = st.columns(3)

        with f_c1:
            st.markdown("<h4 style='color:#FFFFFF; margin-bottom:0.5rem;'>Primary Vital Signs</h4>", unsafe_allow_html=True)
            hr_val = st.slider("Heart Rate (Pulse bpm)", 30, 200, int(active["hr"]), key="sl_hr", help="Normal: 60 - 100 bpm")
            spo2_val = st.slider("Oxygen Saturation (SpO₂ %)", 65.0, 100.0, float(active["spo2"]), step=0.5, key="sl_spo2", help="Normal: 95 - 100%")
            temp_val = st.slider("Body Temperature (°C)", 33.0, 42.0, float(active["temp"]), step=0.1, key="sl_temp", help="Normal: 36.5 - 37.5°C")
            rr_val = st.slider("Respiratory Rate (breaths/min)", 6, 45, int(active["rr"]), key="sl_rr", help="Normal: 12 - 20 breaths/min")

        with f_c2:
            st.markdown("<h4 style='color:#FFFFFF; margin-bottom:0.5rem;'>Hemodynamics & Blood Pressure</h4>", unsafe_allow_html=True)
            sbp_val = st.slider("Systolic Blood Pressure (mmHg)", 60, 260, int(active["sbp"]), key="sl_sbp", help="Normal: 100 - 130 mmHg")
            dbp_val = st.slider("Diastolic Blood Pressure (mmHg)", 35, 160, int(active["dbp"]), key="sl_dbp", help="Normal: 60 - 85 mmHg")
            age_val = st.number_input("Patient Age (Years)", 1, 110, int(active["age"]), key="num_age")
            gender_val = st.selectbox("Biological Sex", ["Male", "Female"], index=0 if active["gender"] == "Male" else 1, key="sel_gender")

        with f_c3:
            st.markdown("<h4 style='color:#FFFFFF; margin-bottom:0.5rem;'>Anthropometrics & Derived Indices</h4>", unsafe_allow_html=True)
            weight_val = st.number_input("Weight (kg)", 20.0, 220.0, float(active["w"]), step=0.5, key="num_weight")
            height_val = st.number_input("Height (m)", 1.0, 2.3, float(active["h"]), step=0.01, key="num_height")

            pulse_pressure = sbp_val - dbp_val
            bmi_val = round(weight_val / (height_val ** 2), 2)
            map_val = round(dbp_val + (pulse_pressure / 3.0), 2)
            hrv_val = 0.04 if hr_val > 115 else 0.11

            st.markdown(
                f"""
                <div style="background:rgba(6, 44, 56, 0.95); border:1px solid rgba(2, 195, 154, 0.4); border-radius:12px; padding:0.9rem; margin-top:0.5rem;">
                    <div style="font-family:'JetBrains Mono'; font-size:0.82rem; color:#02C39A; font-weight:800;">AUTO-COMPUTED INDICES</div>
                    <div style="font-size:0.88rem; color:#FFFFFF; margin-top:0.3rem;">• <b>Pulse Pressure:</b> {pulse_pressure} mmHg</div>
                    <div style="font-size:0.88rem; color:#FFFFFF;">• <b>Body Mass Index (BMI):</b> {bmi_val} kg/m²</div>
                    <div style="font-size:0.88rem; color:#FFFFFF;">• <b>Mean Arterial Pressure (MAP):</b> {map_val} mmHg</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        patient_name_input = st.text_input("Patient ID / Name for Record", active["name"], key="inp_patient_name")

    # Input dict for model
    input_dict = {
        'Heart Rate': hr_val,
        'Respiratory Rate': rr_val,
        'Body Temperature': temp_val,
        'Oxygen Saturation': spo2_val,
        'Systolic Blood Pressure': sbp_val,
        'Diastolic Blood Pressure': dbp_val,
        'Age': age_val,
        'Gender': 1 if gender_val == "Male" else 0,
        'Weight (kg)': weight_val,
        'Height (m)': height_val,
        'Derived_HRV': hrv_val,
        'Derived_Pulse_Pressure': pulse_pressure,
        'Derived_BMI': bmi_val,
        'Derived_MAP': map_val
    }

    result = predict_patient(input_dict)

    wave_divider()

    # Results & XAI Output
    st.markdown("### 📊 Clinical Evaluation & Explainable AI Output")

    col_res_l, col_res_r = st.columns([1, 1.3], gap="medium")

    with col_res_l:
        st.plotly_chart(
            render_risk_gauge(result["risk_score"], result["risk_category"], result["risk_color"]),
            use_container_width=True,
            key="clinician_risk_gauge_chart",
            config={'displayModeBar': False}
        )

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {result['risk_color']}; padding:1.1rem; margin-top:0.5rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:'JetBrains Mono'; font-size:0.88rem; color:#02C39A; font-weight:700;">NEWS2 Clinical Score:</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:800; font-size:1.15rem; color:{result['risk_color']};">{result['news2_score']} ({result['news2_level']})</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.6rem;">
                    <span style="font-family:'JetBrains Mono'; font-size:0.88rem; color:#02C39A; font-weight:700;">Anomaly Detector (IsoForest):</span>
                    <span style="font-family:'JetBrains Mono'; font-weight:800; font-size:0.95rem; color:{'#E8543E' if result['is_anomaly'] == 'Yes' else '#02C39A'};">
                        {'🚨 Atypical Pattern Flagged' if result['is_anomaly'] == 'Yes' else '✅ Standard Physiological Pattern'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_res_r:
        st.plotly_chart(
            render_shap_bar_chart(result["explanation_df"]),
            use_container_width=True,
            key="clinician_shap_bar_chart",
            config={'displayModeBar': False}
        )

    # AI Clinical Summary Card
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 4px solid {result['risk_color']}; margin-top:1rem;">
            <div style="font-family:'JetBrains Mono'; color:#02C39A; font-size:0.8rem; font-weight:800; letter-spacing:0.1em; text-transform:uppercase;">
                AI Clinical Decision Support Summary
            </div>
            <h3 style="margin: 0.3rem 0 0.6rem 0; font-size:1.3rem; color:{result['risk_color']};">
                {result['ai_summary']['status_headline']}
            </h3>
            <p style="font-size:0.95rem; line-height:1.6; color:#FFFFFF; margin-bottom:0.8rem;">
                <b>Clinical Findings:</b> {result['ai_summary']['findings']}
            </p>
            <div style="background:rgba(255,255,255,0.08); border-radius:10px; padding:0.9rem 1.1rem; font-size:0.92rem; color:#FFFFFF;">
                {result['ai_summary']['action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Biomarker Table
    st.markdown("<h4 style='color:#FFFFFF;'>🩺 Biomarker Clinical Range Breakdown</h4>", unsafe_allow_html=True)
    
    biomarkers_data = [
        ("Oxygen Saturation (SpO₂)", f"{spo2_val}%", "95 – 100%", "#E8543E" if spo2_val < 90 else ("#F2A65A" if spo2_val < 94 else "#02C39A"), "Severe Hypoxemia" if spo2_val < 90 else ("Mild Hypoxia" if spo2_val < 94 else "Optimal")),
        ("Systolic Blood Pressure", f"{sbp_val} mmHg", "100 – 130 mmHg", "#E8543E" if sbp_val >= 180 or sbp_val < 90 else ("#F2A65A" if sbp_val >= 140 else "#02C39A"), "Crisis / Shock" if sbp_val >= 180 or sbp_val < 90 else ("Elevated" if sbp_val >= 140 else "Normal")),
        ("Diastolic Blood Pressure", f"{dbp_val} mmHg", "60 – 85 mmHg", "#E8543E" if dbp_val >= 120 or dbp_val < 50 else ("#F2A65A" if dbp_val >= 90 else "#02C39A"), "Severe" if dbp_val >= 120 or dbp_val < 50 else ("Stage 1" if dbp_val >= 90 else "Normal")),
        ("Heart Rate (Pulse)", f"{hr_val} bpm", "60 – 100 bpm", "#E8543E" if hr_val > 130 or hr_val < 45 else ("#F2A65A" if hr_val > 100 or hr_val < 55 else "#02C39A"), "Severe Tachy/Brady" if hr_val > 130 or hr_val < 45 else ("Elevated" if hr_val > 100 else "Normal")),
        ("Body Temperature", f"{temp_val}°C", "36.5 – 37.5°C", "#E8543E" if temp_val >= 39.5 or temp_val < 35.0 else ("#F2A65A" if temp_val >= 38.0 else "#02C39A"), "High Fever / Hypothermia" if temp_val >= 39.5 or temp_val < 35.0 else ("Fever" if temp_val >= 38.0 else "Normal")),
        ("Respiratory Rate", f"{rr_val} breaths/min", "12 – 20 breaths/min", "#E8543E" if rr_val >= 25 or rr_val <= 8 else ("#F2A65A" if rr_val >= 21 else "#02C39A"), "Severe Tachypnea" if rr_val >= 25 else ("Elevated" if rr_val >= 21 else "Normal")),
    ]

    rows_html = "".join(
        f"<tr><td style='padding:0.6rem; color:#FFFFFF; border-bottom:1px solid rgba(255,255,255,0.1);'><b>{name}</b></td>"
        f"<td style='padding:0.6rem; color:#FFFFFF; border-bottom:1px solid rgba(255,255,255,0.1);'>{val}</td>"
        f"<td style='padding:0.6rem; color:#FFFFFF; border-bottom:1px solid rgba(255,255,255,0.1);'>{ref}</td>"
        f"<td style='padding:0.6rem; color:{color}; font-weight:700; border-bottom:1px solid rgba(255,255,255,0.1);'>{status}</td></tr>"
        for name, val, ref, color, status in biomarkers_data
    )

    st.markdown(
        f"""
        <div style="background:rgba(6,44,56,0.85); border:1px solid rgba(2,195,154,0.3); border-radius:14px; padding:0.5rem; overflow-x:auto;">
            <table style="width:100%; border-collapse:collapse; font-family:'JetBrains Mono',monospace; font-size:0.86rem;">
                <thead>
                    <tr style="border-bottom:2px solid #02C39A; text-align:left;">
                        <th style="padding:0.6rem; color:#02C39A;">Biomarker</th>
                        <th style="padding:0.6rem; color:#02C39A;">Patient Value</th>
                        <th style="padding:0.6rem; color:#02C39A;">Clinical Reference Range</th>
                        <th style="padding:0.6rem; color:#02C39A;">Clinical Status</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    if st.button("💾 Save Prediction to History Log", type="primary", key="save_clinician_btn"):
        pid = save_prediction(input_dict, result, patient_name_input)
        
        # Also save to user's vitals table if logged in
        cur_email = st.session_state.get("current_user_email", "")
        if cur_email:
            save_user_vitals(cur_email, hr_val, sbp_val, dbp_val, spo2_val, temp_val, rr_val, result['risk_score'], result['risk_category'])
            
        st.success(f"✅ Prediction for '{patient_name_input}' successfully saved to database (Record ID: {pid})!")


# ============================================================
# TAB 2: SMART PATIENT HEALTH CHECK
# ============================================================
with tab_patient:
    st.markdown("### 👤 Smart Patient Health Check")
    st.caption("Designed for non-medical users. Answer simple questions or enter smartwatch data to receive an instant plain-language risk assessment:")

    p_col1, p_col2 = st.columns([1.1, 0.9], gap="large")

    with p_col1:
        st.markdown("<h4 style='color:#FFFFFF;'>Step 1: Patient Profile</h4>", unsafe_allow_html=True)
        p_name = st.text_input("Your Name / Nickname", "Patient", key="p_mode_name")
        p_age = st.number_input("Your Age", 18, 100, 32, key="p_mode_age")
        p_gender = st.radio("Biological Sex", ["Male", "Female"], horizontal=True, key="p_mode_gender")

        st.markdown("<h4 style='color:#FFFFFF; margin-top:1rem;'>Step 2: Available Devices</h4>", unsafe_allow_html=True)
        st.caption("Check any devices you have available at home:")
        has_smartwatch = st.checkbox("⌚ Smartwatch / Fitness Band (Apple Watch, Fitbit, Samsung)", value=True, key="p_dev_watch")
        has_oximeter = st.checkbox("🫁 Fingertip Pulse Oximeter", value=False, key="p_dev_oxi")
        has_bp_cuff = st.checkbox("🩸 Home Blood Pressure Monitor (BP Cuff)", value=False, key="p_dev_bp")
        has_thermometer = st.checkbox("🌡 Digital Thermometer", value=True, key="p_dev_temp")

        st.markdown("<h4 style='color:#FFFFFF; margin-top:1rem;'>Step 3: Enter Your Measurements</h4>", unsafe_allow_html=True)

        base_p_hr = 74.0
        base_p_spo2 = 98.0
        base_p_sbp = 120.0
        base_p_dbp = 78.0
        base_p_temp = 36.8
        base_p_rr = 15.0

        if has_smartwatch or has_oximeter:
            base_p_hr = float(st.number_input("Heart Rate from watch/oximeter (bpm)", 40, 190, 78, key="p_inp_hr"))
        if has_oximeter:
            base_p_spo2 = float(st.number_input("Oxygen Level (SpO₂ %) from oximeter", 65.0, 100.0, 98.0, step=0.5, key="p_inp_spo2"))
        if has_bp_cuff:
            bp_c1, bp_c2 = st.columns(2)
            with bp_c1:
                base_p_sbp = float(st.number_input("Top BP Number (Systolic)", 70, 240, 120, key="p_inp_sbp"))
            with bp_c2:
                base_p_dbp = float(st.number_input("Bottom BP Number (Diastolic)", 40, 140, 80, key="p_inp_dbp"))
        if has_thermometer:
            base_p_temp = float(st.number_input("Body Temperature (°C)", 33.0, 42.0, 36.8, step=0.1, key="p_inp_temp"))

        st.markdown("<h4 style='color:#FFFFFF; margin-top:1rem;'>Step 4: Are you experiencing any symptoms?</h4>", unsafe_allow_html=True)
        sym_sob = st.checkbox("Shortness of breath / difficulty breathing", key="p_sym_sob")
        sym_chest = st.checkbox("Chest tightness, heaviness, or pain", key="p_sym_chest")
        sym_dizzy = st.checkbox("Feeling lightheaded, dizzy, or weak", key="p_sym_dizzy")
        sym_fever = st.checkbox("Chills, feverish feeling, or hot sweats", key="p_sym_fever")
        sym_cyanosis = st.checkbox("Bluish or pale lips/fingertips", key="p_sym_cyanosis")
        sym_confuse = st.checkbox("Sudden confusion or extreme lethargy", key="p_sym_confuse")

        # Dynamic symptom estimation
        symptoms_map = {
            'sob': sym_sob,
            'chest': sym_chest,
            'dizzy': sym_dizzy,
            'fever': sym_fever,
            'cyanosis': sym_cyanosis,
            'confusion': sym_confuse
        }

        est_hr, est_spo2, est_sbp, est_dbp, est_temp, est_rr = estimate_vitals_from_symptoms(
            base_p_hr, base_p_spo2, base_p_sbp, base_p_dbp, base_p_temp, base_p_rr, symptoms_map
        )

    with p_col2:
        st.markdown("<h4 style='color:#FFFFFF;'>💡 Your Personalized Health Assessment</h4>", unsafe_allow_html=True)

        p_pulse_pressure = est_sbp - est_dbp
        p_bmi = 23.5
        p_map = round(est_dbp + (p_pulse_pressure / 3.0), 2)
        p_hrv = 0.04 if est_hr > 115 else 0.10

        p_input_dict = {
            'Heart Rate': est_hr,
            'Respiratory Rate': est_rr,
            'Body Temperature': est_temp,
            'Oxygen Saturation': est_spo2,
            'Systolic Blood Pressure': est_sbp,
            'Diastolic Blood Pressure': est_dbp,
            'Age': p_age,
            'Gender': 1 if p_gender == "Male" else 0,
            'Weight (kg)': 70.0,
            'Height (m)': 1.72,
            'Derived_HRV': p_hrv,
            'Derived_Pulse_Pressure': p_pulse_pressure,
            'Derived_BMI': p_bmi,
            'Derived_MAP': p_map
        }

        p_res = predict_patient(p_input_dict)

        st.plotly_chart(
            render_risk_gauge(p_res["risk_score"], p_res["risk_category"], p_res["risk_color"]),
            use_container_width=True,
            key="patient_mode_risk_gauge_chart",
            config={'displayModeBar': False}
        )

        if p_res["risk_score"] <= 30:
            user_guidance = "🟢 **Optimal Vital Health:** Your readings and reported responses indicate healthy physiological status. Continue routine healthy habits."
        elif p_res["risk_score"] <= 60:
            user_guidance = "🟡 **Moderate Alert:** Some mild physiological strain or symptoms detected. Rest, stay well hydrated, and re-check in a few hours. If symptoms persist for over 24 hours, consult your physician."
        elif p_res["risk_score"] <= 80:
            user_guidance = "🟠 **High Risk Alert:** Notable physiological instability detected (such as elevated heart rate, high blood pressure, or fever). A timely doctor consultation is strongly recommended."
        else:
            user_guidance = "🔴 **Critical Emergency Warning:** Your vital signs or reported symptoms indicate severe distress. Please seek immediate emergency medical care (call 112 / 911 / 999 or visit the nearest ER)."

        st.markdown(
            f"""
            <div class="glass-card" style="border-left:4px solid {p_res['risk_color']};">
                <h4 style="color:{p_res['risk_color']}; margin-top:0;">Assessment Summary</h4>
                <p style="font-size:0.95rem; line-height:1.6; color:#FFFFFF; margin-bottom:0.8rem;">{user_guidance}</p>
                <div style="font-size:0.85rem; color:#02C39A; font-weight:700;">
                    • Primary Driving Factor: {p_res['top_factors'][0][0]} ({p_res['top_factors'][0][4]})
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("💾 Save My Health Assessment", type="primary", key="save_patient_mode_btn"):
            pid = save_prediction(p_input_dict, p_res, p_name)
            cur_email = st.session_state.get("current_user_email", "")
            if cur_email:
                save_user_vitals(cur_email, est_hr, est_sbp, est_dbp, est_spo2, est_temp, est_rr, p_res['risk_score'], p_res['risk_category'])
            st.success(f"✅ Your health assessment was saved successfully (Record ID: {pid})!")


# ============================================================
# TAB 3: PATIENT HISTORY & TRAJECTORIES
# ============================================================
with tab_history:
    st.markdown("### 📈 Patient Deterioration History & Trajectories")
    st.caption("Track deterioration trajectories across patient cohorts and historical visits:")

    history_df = get_prediction_history()
    unique_patients = history_df["Patient_Name"].dropna().unique().tolist()

    t_col1, t_col2 = st.columns([1, 2], gap="medium")

    with t_col1:
        st.markdown("<h4 style='color:#FFFFFF;'>Select Patient Record</h4>", unsafe_allow_html=True)
        selected_patient = st.selectbox(
            "Choose a Patient Record:",
            unique_patients,
            index=0 if unique_patients else 0,
            key="hist_patient_selector"
        )

        p_subset = history_df[history_df["Patient_Name"] == selected_patient]
        if not p_subset.empty:
            latest_rec = p_subset.iloc[0]
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:4px solid #02C39A;">
                    <h4 style="margin-top:0; color:#02C39A;">{selected_patient}</h4>
                    <div style="font-size:0.88rem; line-height:1.7; color:#FFFFFF;">
                        • <b>Cohort:</b> {latest_rec['Case_Type']}<br>
                        • <b>Total Recorded Checkups:</b> {len(p_subset)}<br>
                        • <b>Latest Risk Score:</b> {latest_rec['Risk_Score']}/100<br>
                        • <b>Category:</b> {latest_rec['Category']}<br>
                        • <b>Vitals:</b> SpO₂ {latest_rec['SpO2_pct']}% | HR {latest_rec['HR_bpm']} bpm | BP {latest_rec['SBP']}/{latest_rec['DBP']} mmHg
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with t_col2:
        st.markdown("<h4 style='color:#FFFFFF;'>📉 Deterioration Trajectory Curve</h4>", unsafe_allow_html=True)
        timeline_df = get_patient_timeline(selected_patient)

        if not timeline_df.empty and len(timeline_df) > 1:
            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['predicted_at'],
                y=timeline_df['risk_score'],
                mode='lines+markers',
                name='Health Risk Index',
                line=dict(color='#E8543E', width=3),
                marker=dict(size=8, color='#E8543E')
            ))

            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['predicted_at'],
                y=timeline_df['oxygen_saturation'],
                mode='lines+markers',
                name='SpO₂ (%)',
                line=dict(color='#02C39A', width=2, dash='dot'),
                marker=dict(size=6, color='#02C39A')
            ))

            fig_timeline.add_trace(go.Scatter(
                x=timeline_df['predicted_at'],
                y=timeline_df['heart_rate'],
                mode='lines+markers',
                name='Heart Rate (bpm)',
                line=dict(color='#F0F3BD', width=2, dash='dash'),
                marker=dict(size=6, color='#F0F3BD')
            ))

            fig_timeline.update_layout(
                title=f"<b>Risk & Vital Signs Trajectory — {selected_patient}</b>",
                title_font=dict(size=14, family='Space Grotesk', color='#FFFFFF'),
                xaxis_title="Checkup Timestamp",
                yaxis_title="Metric Value",
                xaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#FFFFFF')),
                yaxis=dict(gridcolor='rgba(255,255,255,0.08)', tickfont=dict(color='#FFFFFF')),
                height=280,
                margin=dict(l=10, r=10, t=40, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#FFFFFF', family='Inter'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#FFFFFF'))
            )
            st.plotly_chart(
                fig_timeline,
                use_container_width=True,
                key="patient_trajectory_timeline_chart",
                config={'displayModeBar': False}
            )
        else:
            st.info("Single snapshot recorded for this patient. Add more predictions to render multi-point trajectory plots.")

    wave_divider()

    # Searchable Database Table
    st.markdown("<h4 style='color:#FFFFFF;'>🗄️ Full Prediction Database</h4>", unsafe_allow_html=True)
    st.dataframe(
        history_df,
        use_container_width=True,
        column_config={
            "Risk_Score": st.column_config.ProgressColumn(
                "Health Risk Index",
                help="Composite Health Risk Index (0-100)",
                format="%.1f",
                min_value=0,
                max_value=100
            ),
            "SpO2_pct": st.column_config.NumberColumn("SpO₂ (%)", format="%.1f%%"),
            "HR_bpm": st.column_config.NumberColumn("HR (bpm)", format="%d"),
            "NEWS2": st.column_config.NumberColumn("NEWS2", format="%d")
        },
        height=320
    )

    csv_data = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Clinical Records to CSV",
        data=csv_data,
        file_name="vital_health_alert_records.csv",
        mime="text/csv",
        key="export_csv_btn"
    )