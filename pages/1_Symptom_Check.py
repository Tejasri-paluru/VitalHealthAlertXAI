"""
1_Symptom_Check.py — Clinical Symptom Triage & Rapid Urgency Evaluation
Vital Health Alert XAI

Categorized symptom assessment, red flag emergency detection,
and direct bridge to the multi-vital AI Deterioration Predictor.
"""

import streamlit as st
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Theme import inject_base_css, wave_divider, render_auth_header

st.set_page_config(
    page_title="Symptom Check — Vital Health Alert XAI",
    page_icon="🚑",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_base_css()
render_auth_header()

st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(6, 44, 56, 0.85) 0%, rgba(4, 27, 36, 0.95) 100%); border: 1px solid rgba(2, 195, 154, 0.25); border-radius: 20px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="section-label">Quick Clinical Triage</div>
        <h1 style="margin: 0.2rem 0 0.5rem 0; font-size:2.2rem;">🚑 Symptom Checker & Urgency Assessment</h1>
        <p style="color:rgba(245,250,248,0.72); margin:0; font-size:0.95rem;">
            A structured, rule-based clinical safety triage for evaluating urgent warning signs and determining
            whether you should seek immediate emergency care or proceed with multi-vital AI scoring.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "💡 **How to use:** Select all symptoms that describe your current state. "
    "If you have measuring devices (smartwatch, blood pressure cuff, oximeter, thermometer), "
    "you can proceed to the **Vitals Predictor** after checking your symptoms for a full AI risk assessment."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='color:#FFFFFF;'>🚨 Emergency Red Flags</h4>", unsafe_allow_html=True)
    st.caption("Immediate life-threatening signs:")
    rf_chest = st.checkbox("Severe crushing chest pain / pressure", key="rf_chest")
    rf_sob = st.checkbox("Gasping for air / unable to speak full sentences", key="rf_sob")
    rf_lips = st.checkbox("Cyanosis (blue/grey lips or fingertips)", key="rf_lips")
    rf_neuro = st.checkbox("Sudden confusion, slurred speech, or facial drooping", key="rf_neuro")
    rf_faint = st.checkbox("Loss of consciousness / fainting episode", key="rf_faint")
    rf_bleed = st.checkbox("Severe uncontrolled bleeding", key="rf_bleed")

with col2:
    st.markdown("<h4 style='color:#FFFFFF;'>🫁 Respiratory & Cardiac Signs</h4>", unsafe_allow_html=True)
    st.caption("Cardiopulmonary symptoms:")
    card_palp = st.checkbox("Rapid, pounding, or racing heartbeat (palpitations)", key="c_palp")
    card_dysp = st.checkbox("Shortness of breath on mild exertion or resting", key="c_dysp")
    card_cough = st.checkbox("Persistent wheezing or severe cough", key="c_cough")
    card_tight = st.checkbox("Mild chest tightness or heaviness", key="c_tight")
    card_swelling = st.checkbox("Swelling in feet, ankles, or legs", key="c_swelling")

with col3:
    st.markdown("<h4 style='color:#FFFFFF;'>🌡 Systemic & General Symptoms</h4>", unsafe_allow_html=True)
    st.caption("Infection and generalized signs:")
    sys_fever = st.checkbox("High fever with shivering, chills, or sweating", key="s_fever")
    sys_fatigue = st.checkbox("Profound weakness / severe fatigue", key="s_fatigue")
    sys_dizzy = st.checkbox("Lightheadedness, dizziness, or unsteadiness", key="s_dizzy")
    sys_nausea = st.checkbox("Nausea, vomiting, or persistent abdominal pain", key="s_nausea")
    sys_headache = st.checkbox("Throbbing headache or body aches", key="s_headache")

st.write("")
check_btn = st.button("🔍 Evaluate My Symptoms", type="primary", use_container_width=True, key="eval_symptoms_btn")

if check_btn:
    red_flags = [rf_chest, rf_sob, rf_lips, rf_neuro, rf_faint, rf_bleed]
    cardio_resp = [card_palp, card_dysp, card_cough, card_tight, card_swelling]
    systemic = [sys_fever, sys_fatigue, sys_dizzy, sys_nausea, sys_headache]

    has_red_flags = any(red_flags)
    n_cardio = sum(cardio_resp)
    n_systemic = sum(systemic)
    total_selected = sum(red_flags) + n_cardio + n_systemic

    wave_divider()
    st.markdown("### 📋 Triage Assessment Result")

    if total_selected == 0:
        st.warning("Please select at least one symptom, or head directly to the Vitals Predictor if you are feeling well.")
    elif has_red_flags:
        st.markdown(
            """
            <div style="background:rgba(232, 84, 62, 0.22); border:2px solid #E8543E; border-radius:16px; padding:1.5rem; margin-bottom:1rem;">
                <h3 style="color:#E8543E; margin-top:0; font-size:1.3rem;">🚨 Critical Emergency Warning — Seek Immediate Medical Care</h3>
                <p style="color:#FFFFFF; font-size:0.95rem; line-height:1.6; margin-bottom:0.8rem;">
                    You have selected one or more <b>Emergency Red Flags</b> (such as severe chest pain, extreme breathlessness, or neurological distress).
                    These signs can indicate acute myocardial infarction, pulmonary embolism, stroke, or severe sepsis.
                </p>
                <div style="background:rgba(0,0,0,0.4); border-radius:10px; padding:0.9rem 1.1rem; color:#FFFFFF; font-weight:700;">
                    👉 <b>Action:</b> Call your local emergency services immediately (e.g. <b>112</b> in India, <b>911</b> in the US, <b>999</b> in the UK)
                    or proceed to the nearest Emergency Department. Do NOT wait.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    elif n_cardio >= 2 or (sys_fever and (card_dysp or card_palp)):
        st.markdown(
            """
            <div style="background:rgba(242, 166, 90, 0.22); border:2px solid #F2A65A; border-radius:16px; padding:1.5rem; margin-bottom:1rem;">
                <h3 style="color:#F2A65A; margin-top:0; font-size:1.3rem;">🟠 Urgent Medical Review Recommended</h3>
                <p style="color:#FFFFFF; font-size:0.95rem; line-height:1.6; margin-bottom:0.8rem;">
                    Your cluster of symptoms indicates notable physiological distress (such as fever combined with rapid heart rate or shortness of breath).
                    A prompt clinical examination is recommended within 12–24 hours.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<h4 style='color:#FFFFFF;'>🔬 Proceed to Vitals Predictor</h4>", unsafe_allow_html=True)
        st.caption("If you have access to a smartwatch, blood pressure cuff, or thermometer, run the full AI Predictor:")

        if st.button("🚀 Transfer to Vitals Predictor →", type="primary", key="bridge_to_pred"):
            st.switch_page("pages/2_Predict.py")
    else:
        st.markdown(
            """
            <div style="background:rgba(2, 195, 154, 0.18); border:2px solid #02C39A; border-radius:16px; padding:1.5rem; margin-bottom:1rem;">
                <h3 style="color:#02C39A; margin-top:0; font-size:1.3rem;">🟢 Mild Symptoms — Rest and Observation</h3>
                <p style="color:#FFFFFF; font-size:0.95rem; line-height:1.6; margin-bottom:0.8rem;">
                    Your selected symptoms appear mild and do not suggest acute decompensation. Ensure adequate hydration, rest, and monitor your symptoms.
                    If your condition worsens or new red flags develop, seek professional medical guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("📊 Run Full Vitals Check Anyway →", key="mild_bridge"):
            st.switch_page("pages/2_Predict.py")

wave_divider()

st.caption(
    "⚠️ **Disclaimer:** This tool provides clinical decision support guidance only and is not a medical diagnosis. "
    "Always consult a qualified healthcare professional in case of doubt or worsening health."
)