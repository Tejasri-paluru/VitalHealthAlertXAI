"""
4_Account.py — Patient & Clinician Portal
Vital Health Alert XAI

Full Authentication & Dashboard Portal:
- User Login & Registration
- 3-Stage OTP & Token Password Reset
- Live Dashboard with Health Risk Index & Gauge
- New Vitals Entry & AI Prediction
- Patient History & Trend Line
"""

import streamlit as st
import streamlit.components.v1 as components
import time
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from auth_utils import (
    get_db, hash_password, create_user, user_exists, verify_login,
    update_user_password, generate_otp, create_password_reset, verify_otp,
    verify_reset_token, clear_password_reset, save_vitals, get_patient_history,
    get_latest_vitals, compute_risk_score, get_app_base_url, send_email,
    registration_email_html, otp_email_html, password_changed_email_html
)

st.set_page_config(
    page_title="Portal — Vital Health Alert XAI",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- RISK CONSTANTS ----------------
RISK_COLORS = {
    "Healthy": "#2DD4BF",
    "Moderate Risk": "#FBBF24",
    "High Risk": "#FB923C",
    "Critical": "#FF6B6B",
}

RISK_GUIDANCE = {
    "Healthy": "Vitals look stable. Continue routine monitoring.",
    "Moderate Risk": "Some readings are outside the normal range. Recheck in a few hours and note any symptoms.",
    "High Risk": "Multiple vitals show concerning deviation. We recommend contacting your doctor soon.",
    "Critical": "Vitals indicate a potential emergency. Please seek immediate medical attention.",
}

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root{
    --navy:#0A2540;
    --brand:#0B3D91;
    --teal:#00B4D8;
    --mint:#2DD4BF;
    --coral:#FF6B6B;
    --bg:#EEF3F8;
    --ink:#1B2430;
    --muted:#5B6B7C;
    --deep:#041B24;
    --mid:#064E5B;
    --light:#02C39A;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 15%, rgba(0, 168, 150, 0.18) 0%, transparent 40%),
                radial-gradient(circle at 80% 85%, rgba(2, 195, 154, 0.15) 0%, transparent 45%),
                linear-gradient(160deg, #031822 0%, #063A48 50%, #03202C 100%) fixed !important;
    color: #FFFFFF !important;
}

[data-testid="stHeader"]{ background: rgba(0,0,0,0); }

/* High contrast inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.95) !important;
    color: #1B2430 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}

/* Glass Card */
.glass-card {
    background: rgba(6, 44, 56, 0.88);
    padding: 30px;
    border-radius: 22px;
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.4);
    border: 1px solid rgba(2, 195, 154, 0.35);
    animation: fadeInUp 0.5s ease-out;
    margin-bottom: 20px;
}

.card-icon-badge{
    width: 64px; height: 64px;
    margin: 0 auto 14px auto;
    border-radius: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    background: linear-gradient(135deg, var(--mid), var(--light));
    box-shadow: 0 10px 24px rgba(2,195,154,0.35);
}

.or-divider{
    display: flex; align-items: center; gap: 12px;
    color: rgba(255,255,255,0.6); font-size: 12.5px; font-weight: 700;
    letter-spacing: .5px;
    margin: 18px 0;
}
.or-divider span{ flex: 1; height: 1px; background: rgba(255,255,255,0.2); }

@keyframes fadeInUp{
    from{opacity:0; transform:translateY(18px);}
    to{opacity:1; transform:translateY(0);}
}

.brand-mark{
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
    margin-bottom: 12px;
}
.brand-mark .dot{
    width: 10px; height: 10px; border-radius: 50%;
    background: #02C39A;
    box-shadow: 0 0 0 4px rgba(2,195,154,0.25);
}
.brand-mark span{
    font-weight: 700; color: #FFFFFF; font-size: 20px;
}

.google-link-btn{
    display: flex; align-items: center; justify-content: center; gap: 10px;
    width: 100%; height: 50px; border-radius: 12px;
    background: #ffffff; color: #1B2430; font-weight: 700; font-size: 15px;
    text-decoration: none; border: 1.5px solid rgba(255,255,255,0.4);
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    margin-bottom: 6px;
}
.google-link-btn:hover{
    background: #f0f4f8; transform: translateY(-1px);
}

.app-alert{
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; border-radius: 12px; font-size: 14px; font-weight: 600;
    margin: 10px 0 6px 0;
}
.app-alert.error{
    background: rgba(255,59,48,0.18); border: 1.5px solid #FF6B6B; color: #FFD2D2;
}
.app-alert.success{
    background: rgba(2,195,154,0.18); border: 1.5px solid #02C39A; color: #D4FFF4;
}

/* Vital cards */
.vital-card{
    background: rgba(6, 44, 56, 0.9);
    border: 1px solid rgba(2, 195, 154, 0.3);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}
.vital-card .v-icon{ font-size: 26px; }
.vital-card .v-value{ font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #F0F3BD; margin-top: 4px; }
.vital-card .v-label{ color: #02C39A; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .3px; }

/* XAI rows */
.xai-row{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.xai-label{ width: 140px; font-size: 14px; color: #FFFFFF; font-weight: 600; flex-shrink: 0; }
.xai-bar-track{ flex: 1; height: 12px; background: rgba(255,255,255,0.12); border-radius: 999px; overflow: hidden; }
.xai-bar-fill{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, #00A896, #02C39A); }
.xai-pct{ width: 50px; text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #F0F3BD; font-weight: 700; flex-shrink: 0; }

/* History table */
.history-table{ width: 100%; border-collapse: collapse; font-size: 14px; font-family: 'JetBrains Mono', monospace; }
.history-table th{
    text-align: left; padding: 10px 12px; color: #02C39A; font-size: 12px;
    text-transform: uppercase; letter-spacing: .3px; border-bottom: 2px solid #02C39A;
}
.history-table td{ padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.1); color: #FFFFFF; }
.risk-tag{ padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; white-space: nowrap; }
.risk-Healthy{ background: rgba(45,212,191,0.2); color: #2DD4BF; border: 1px solid #2DD4BF; }
.risk-ModerateRisk{ background: rgba(251,191,36,0.2); color: #FBBF24; border: 1px solid #FBBF24; }
.risk-HighRisk{ background: rgba(251,146,60,0.2); color: #FB923C; border: 1px solid #FB923C; }
.risk-Critical{ background: rgba(255,107,107,0.25); color: #FF6B6B; border: 1px solid #FF6B6B; }
</style>
""", unsafe_allow_html=True)


def brand_header():
    st.markdown("""
    <div class="brand-mark"><div class="dot"></div><span>Vital Health Alert XAI</span></div>
    """, unsafe_allow_html=True)


def show_alert(message, kind="error"):
    icon = "✅" if kind == "success" else "⚠️"
    st.markdown(f"""
    <div class="app-alert {kind}">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


def google_login_link():
    st.markdown("""
    <a href="https://accounts.google.com/signin" target="_blank" rel="noopener noreferrer" class="google-link-btn">
        <span style="font-weight:800; font-size:18px; color:#4285F4;">G</span> Continue with Google
    </a>
    """, unsafe_allow_html=True)


def app_nav(active_page):
    items = [
        ("dashboard", "🏠 Dashboard"),
        ("vitals_entry", "➕ New Vitals Check"),
        ("history", "📈 History Timeline"),
    ]
    cols = st.columns(len(items) + 1)
    for i, (key, label) in enumerate(items):
        with cols[i]:
            shown_label = f"● {label}" if key == active_page else label
            if st.button(shown_label, key=f"nav_{key}", use_container_width=True):
                st.session_state.account_page = key
                st.rerun()
    with cols[-1]:
        if st.button("🚪 Logout", key="nav_logout", use_container_width=True):
            st.session_state.current_user_email = ""
            st.session_state.last_prediction = None
            st.session_state.account_page = "user_login"
            st.rerun()
    st.write("")


def render_risk_gauge_portal(score, category):
    color = RISK_COLORS.get(category, "#2DD4BF")
    circumference = 251.2
    offset = circumference - (circumference * min(score, 100) / 100)

    st.markdown(f"""
    <div class="glass-card" style="text-align:center;">
        <div style="text-align:center; padding:10px 0 4px 0;">
            <svg viewBox="0 0 200 120" style="width:100%; max-width:280px;">
                <path d="M20,110 A80,80 0 0,1 180,110" fill="none"
                    stroke="rgba(255,255,255,0.12)" stroke-width="14" stroke-linecap="round"/>
                <path d="M20,110 A80,80 0 0,1 180,110" fill="none"
                    stroke="{color}" stroke-width="14" stroke-linecap="round"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"/>
            </svg>
            <div style="font-family:'JetBrains Mono',monospace; font-size:44px; font-weight:800; color:{color}; margin-top:-46px;">{score}</div>
            <div style="color:rgba(255,255,255,0.7); font-size:13px; font-weight:600; text-transform:uppercase; margin-bottom:10px;">Health Risk Index</div>
            <div style="display:inline-block; padding:6px 18px; border-radius:999px; font-weight:700; font-size:13.5px; background:{color}22; color:{color}; border:1.5px solid {color}88;">{category}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_explainability(contributions):
    rows_html = ""
    for name, pct in sorted(contributions.items(), key=lambda x: -x[1]):
        rows_html += f"""
        <div class="xai-row">
            <div class="xai-label">{name}</div>
            <div class="xai-bar-track"><div class="xai-bar-fill" style="width:{pct}%;"></div></div>
            <div class="xai-pct">{pct}%</div>
        </div>
        """
    st.markdown(f"""
    <div class="glass-card" style="margin-top:20px; text-align:left;">
        <h3 style="color:#FFFFFF; margin-bottom:2px;">🧠 Why this prediction?</h3>
        <p style="color:rgba(255,255,255,0.7); font-size:13.5px; margin-bottom:18px;">Feature contribution to your risk score</p>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)


def render_recommendation(category):
    color = RISK_COLORS.get(category, "#2DD4BF")
    tip = RISK_GUIDANCE.get(category, "")
    st.markdown(f"""
    <div class="glass-card" style="margin-top:20px; text-align:left; border-left:5px solid {color};">
        <h3 style="color:#FFFFFF;">📋 Clinical Guidance</h3>
        <p style="color:#F5FAF8; font-size:15px;">{tip}</p>
        <p style="color:rgba(255,255,255,0.6); font-size:12px; margin-top:12px;">
            This is decision-support guidance, not a medical diagnosis. Always consult a healthcare professional.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_trend_line(scores):
    if len(scores) < 2:
        scores = [scores[0], scores[0]] if scores else [0, 0]
    width, height = 600, 140
    n = len(scores)
    points = []
    for i, s in enumerate(scores):
        x = (i / (n - 1)) * (width - 40) + 20
        y = height - 20 - (min(s, 100) / 100) * (height - 40)
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    st.markdown(f"""
    <div class="glass-card" style="text-align:left;">
        <h3 style="color:#FFFFFF; margin-bottom:10px;">📈 Risk Score Trend</h3>
        <svg viewBox="0 0 {width} {height}" style="width:100%; height:auto;">
            <polyline points="{polyline}" fill="none" stroke="#02C39A" stroke-width="3"
                stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)


# Initialize Account Session State
if "account_page" not in st.session_state:
    if st.session_state.get("current_user_email"):
        st.session_state.account_page = "dashboard"
    else:
        st.session_state.account_page = "user_login"

if "reset_stage" not in st.session_state:
    st.session_state.reset_stage = "request"

if "reset_email" not in st.session_state:
    st.session_state.reset_email = ""


# Handle Reset Token URL param
if "reset_token" in st.query_params and not st.session_state.get("reset_link_handled"):
    _token = st.query_params["reset_token"]
    _email = verify_reset_token(_token)
    if _email:
        st.session_state.reset_email = _email
        st.session_state.reset_stage = "newpass"
        st.session_state.account_page = "reset_password"
    else:
        st.session_state.reset_stage = "request"
        st.session_state.account_page = "reset_password"
        show_alert("This reset link is invalid or has expired.", "error")
    st.session_state.reset_link_handled = True
    st.query_params.clear()


# ============================================================
# PAGE ROUTING
# ============================================================

current_email = st.session_state.get("current_user_email", "")

# 1. USER LOGIN
if st.session_state.account_page == "user_login":
    brand_header()
    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon-badge">➕</div>
            <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Welcome Back!</h2>
            <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:22px;">
                Sign in to view your health reports, records, and AI predictions
            </p>
        """, unsafe_allow_html=True)

        username = st.text_input("Email or Phone Number", key="login_usr_inp")
        password = st.text_input("Password", type="password", key="login_pw_inp")

        f_c1, f_c2 = st.columns([3, 2])
        with f_c2:
            if st.button("Forgot Password?", key="forgot_pw_btn", use_container_width=True):
                st.session_state.reset_stage = "request"
                st.session_state.account_page = "reset_password"
                st.rerun()

        if st.button("Sign In", type="primary", use_container_width=True, key="login_submit_btn"):
            if not username or not password:
                show_alert("Please enter email and password.", "error")
            elif verify_login(username, password):
                st.session_state.current_user_email = username.strip().lower()
                st.session_state.account_page = "dashboard"
                st.rerun()
            else:
                show_alert("Invalid email or password. Please try again.", "error")

        st.markdown('<div class="or-divider"><span></span>OR<span></span></div>', unsafe_allow_html=True)
        google_login_link()

        st.write("")
        c_sw1, c_sw2 = st.columns([3, 2])
        with c_sw1:
            st.markdown("<p style='text-align:right; color:rgba(255,255,255,0.7); font-size:14px; margin-top:8px;'>Don't have an account?</p>", unsafe_allow_html=True)
        with c_sw2:
            if st.button("Sign Up", key="to_signup_btn", use_container_width=True):
                st.session_state.account_page = "user_signup"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #02C39A; height:100%;">
            <h3 style="color:#02C39A; margin-top:0;">Secure Health Portal</h3>
            <p style="font-size:0.95rem; line-height:1.7; color:#FFFFFF;">
                • <b>Track Deterioration Timelines:</b> View progressive vitals across all your checkups.<br><br>
                • <b>SHAP Explanation Audit:</b> Revisit why previous risk scores were computed.<br><br>
                • <b>Encrypted Local Storage:</b> Passwords secured with SHA-256 and local SQLite storage.
            </p>
        </div>
        """, unsafe_allow_html=True)


# 2. USER SIGNUP
elif st.session_state.account_page == "user_signup":
    brand_header()
    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        st.markdown("""
        <div class="glass-card">
            <div class="card-icon-badge">🆕</div>
            <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Create Your Account</h2>
            <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:20px;">
                Join Vital Health Alert XAI to track your health securely
            </p>
        """, unsafe_allow_html=True)

        full_name = st.text_input("Full Name", placeholder="e.g. Dr. Alex Morgan or Elena Rostova", key="su_fullname")
        email = st.text_input("Email Address", placeholder="name@example.com", key="su_email")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210", key="su_phone")
        password = st.text_input("Password", type="password", placeholder="••••••••", key="su_password")

        if st.button("Create Account", type="primary", use_container_width=True, key="su_submit_btn"):
            if not full_name or not email or not phone or not password:
                show_alert("Please enter all details correctly.", "error")
            elif user_exists(email):
                show_alert("An account with this email already exists.", "error")
            else:
                create_user(email, full_name, password, phone)
                send_email(
                    email,
                    "Welcome to Vital Health Alert XAI 🎉",
                    registration_email_html(full_name)
                )
                st.session_state.current_user_email = email.strip().lower()
                st.session_state.account_page = "dashboard"
                st.rerun()

        st.markdown('<div class="or-divider"><span></span>OR<span></span></div>', unsafe_allow_html=True)
        google_login_link()

        st.write("")
        c_sw1, c_sw2 = st.columns([3, 2])
        with c_sw1:
            st.markdown("<p style='text-align:right; color:rgba(255,255,255,0.7); font-size:14px; margin-top:8px;'>Already have an account?</p>", unsafe_allow_html=True)
        with c_sw2:
            if st.button("Login", key="to_login_btn", use_container_width=True):
                st.session_state.account_page = "user_login"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class="glass-card" style="border-left:4px solid #F0F3BD; height:100%;">
            <h3 style="color:#F0F3BD; margin-top:0;">Privacy & Patient Safety</h3>
            <p style="font-size:0.95rem; line-height:1.7; color:#FFFFFF;">
                • <b>Confidential Medical Data:</b> Encrypted and stored locally on your device.<br><br>
                • <b>Instant Alert Access:</b> Access rapid early deterioration assessments.<br><br>
                • <b>Multi-Device Support:</b> Connect smartwatch readings and home monitors.
            </p>
        </div>
        """, unsafe_allow_html=True)


# 3. RESET PASSWORD
elif st.session_state.account_page == "reset_password":
    brand_header()
    stage = st.session_state.get("reset_stage", "request")

    st.markdown("""<div class="glass-card" style="max-width:560px; margin:0 auto;">""", unsafe_allow_html=True)

    if stage == "request":
        st.markdown("""
        <div class="card-icon-badge">🔑</div>
        <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Forgot Password</h2>
        <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:22px;">
            Enter your registered email — we'll send you a 6-digit OTP
        </p>
        """, unsafe_allow_html=True)

        reg_email = st.text_input("Registered Email", placeholder="name@example.com", key="pw_req_email")

        if st.button("Send OTP Code", type="primary", use_container_width=True, key="pw_send_otp_btn"):
            if not reg_email:
                show_alert("Please enter your email.", "error")
            elif not user_exists(reg_email):
                show_alert("No account found with this email.", "error")
            else:
                otp, token = create_password_reset(reg_email)
                reset_link = f"{get_app_base_url()}/?reset_token={token}"
                send_email(
                    reg_email,
                    "Your Password Reset OTP — Vital Health Alert XAI",
                    otp_email_html(otp, reset_link)
                )
                st.session_state.reset_email = reg_email.strip().lower()
                st.session_state.reset_stage = "otp"
                st.session_state["demo_otp_display"] = otp
                st.rerun()

    elif stage == "otp":
        demo_code = st.session_state.get("demo_otp_display", "")
        st.markdown(f"""
        <div class="card-icon-badge">📩</div>
        <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Enter OTP</h2>
        <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:12px;">
            We sent a 6-digit code to <b>{st.session_state.get('reset_email','')}</b>
        </p>
        """, unsafe_allow_html=True)
        if demo_code:
            st.info(f"💡 Offline / Demo Code: **{demo_code}**")

        entered_otp = st.text_input("6-Digit OTP", placeholder="123456", key="pw_otp_val")

        if st.button("Verify OTP", type="primary", use_container_width=True, key="pw_verify_otp_btn"):
            if verify_otp(st.session_state.get("reset_email", ""), entered_otp):
                st.session_state.reset_stage = "newpass"
                st.rerun()
            else:
                show_alert("Invalid or expired OTP. Please try again.", "error")

    elif stage == "newpass":
        st.markdown(f"""
        <div class="card-icon-badge">🔒</div>
        <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Set New Password</h2>
        <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:22px;">
            Resetting password for <b>{st.session_state.get('reset_email','')}</b>
        </p>
        """, unsafe_allow_html=True)

        new_pw = st.text_input("New Password", type="password", placeholder="••••••••", key="pw_new_1")
        confirm_pw = st.text_input("Confirm Password", type="password", placeholder="••••••••", key="pw_new_2")

        if st.button("Reset Password Now", type="primary", use_container_width=True, key="pw_update_sub_btn"):
            if not new_pw or not confirm_pw:
                show_alert("Please fill in both fields.", "error")
            elif new_pw != confirm_pw:
                show_alert("Passwords do not match.", "error")
            else:
                em = st.session_state.get("reset_email", "")
                update_user_password(em, new_pw)
                clear_password_reset(em)
                send_email(
                    em,
                    "Your Password Has Been Changed — Vital Health Alert XAI",
                    password_changed_email_html()
                )
                st.session_state.reset_stage = "request"
                st.session_state.current_user_email = em
                st.session_state.account_page = "dashboard"
                st.rerun()

    st.write("")
    if st.button("← Back to Sign In", key="pw_back_login_btn", use_container_width=True):
        st.session_state.reset_stage = "request"
        st.session_state.account_page = "user_login"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# 4. DASHBOARD
elif st.session_state.account_page == "dashboard":
    app_nav("dashboard")

    st.markdown(f"""
    <div style="text-align:center; margin-bottom:22px;">
        <div style="color:#02C39A; font-size:14px; font-family:'JetBrains Mono',monospace; letter-spacing:1px; font-weight:800;">PORTAL DASHBOARD</div>
        <div style="color:#FFFFFF; font-size:26px; font-weight:800;">Welcome, {current_email} 👋</div>
    </div>
    """, unsafe_allow_html=True)

    latest = get_latest_vitals(current_email)

    if not latest:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:40px;">🩺</div>
            <h3 style="color:#FFFFFF; margin-top:6px;">No vitals recorded yet</h3>
            <p style="color:rgba(255,255,255,0.7); font-size:14.5px;">
                Enter your first vital signs to see your AI-powered Health Risk Index.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➕ Enter Vitals Now", type="primary", use_container_width=True, key="dash_cta"):
            st.session_state.account_page = "vitals_entry"
            st.rerun()
    else:
        recorded_at, hr, bp_sys, bp_dia, spo2, temp, resp_rate, risk_score, risk_category = latest

        render_risk_gauge_portal(risk_score, risk_category)

        st.write("")
        st.markdown("<h4 style='color:#fff; text-align:center;'>Latest Recorded Vitals</h4>", unsafe_allow_html=True)
        v1, v2, v3 = st.columns(3)
        with v1:
            st.markdown(f"""<div class="vital-card"><div class="v-icon">❤️</div>
                <div class="v-value">{int(hr)}</div><div class="v-label">Heart Rate (bpm)</div></div>""", unsafe_allow_html=True)
        with v2:
            st.markdown(f"""<div class="vital-card"><div class="v-icon">🩸</div>
                <div class="v-value">{int(bp_sys)}/{int(bp_dia)}</div><div class="v-label">Blood Pressure</div></div>""", unsafe_allow_html=True)
        with v3:
            st.markdown(f"""<div class="vital-card"><div class="v-icon">🫁</div>
                <div class="v-value">{spo2}%</div><div class="v-label">SpO2 Level</div></div>""", unsafe_allow_html=True)

        st.write("")
        if st.button("➕ Enter New Vitals", type="primary", use_container_width=True, key="dash_new_vitals"):
            st.session_state.account_page = "vitals_entry"
            st.rerun()


# 5. VITALS ENTRY
elif st.session_state.account_page == "vitals_entry":
    app_nav("vitals_entry")

    st.markdown("""
    <div class="glass-card" style="max-width:650px; margin:0 auto;">
        <div class="card-icon-badge">🩺</div>
        <h2 style="text-align:center; color:#FFFFFF; margin-bottom:4px;">Enter Vital Signs</h2>
        <p style="text-align:center; color:rgba(255,255,255,0.7); font-size:14.5px; margin-bottom:20px;">
            The AI model will analyze these readings instantly
        </p>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hr = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=75, key="ve_hr")
        bp_sys = st.number_input("BP Systolic (mmHg)", min_value=60, max_value=250, value=118, key="ve_sbp")
        spo2 = st.number_input("Oxygen Saturation (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.5, key="ve_spo2")
    with c2:
        bp_dia = st.number_input("BP Diastolic (mmHg)", min_value=40, max_value=150, value=78, key="ve_dbp")
        temp = st.number_input("Temperature (°C)", min_value=33.0, max_value=43.0, value=36.8, step=0.1, key="ve_temp")
        resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=15, key="ve_rr")

    if st.button("🔮 Run AI Prediction", type="primary", use_container_width=True, key="predict_entry_btn"):
        # Convert C to F for compute_risk_score if needed or standard evaluation
        temp_f = (temp * 9/5) + 32
        score, category, contributions = compute_risk_score(hr, bp_sys, bp_dia, spo2, temp_f, resp_rate)
        save_vitals(current_email, hr, bp_sys, bp_dia, spo2, temp, resp_rate, score, category)
        st.session_state.last_prediction = (score, category, contributions)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("last_prediction"):
        score, category, contributions = st.session_state.last_prediction
        st.write("")
        render_risk_gauge_portal(score, category)
        render_explainability(contributions)
        render_recommendation(category)


# 6. HISTORY / TIMELINE
elif st.session_state.account_page == "history":
    app_nav("history")

    st.markdown("""
    <h2 style="text-align:center; color:#FFFFFF; margin-bottom:20px;">
        Your Personal Health Timeline
    </h2>
    """, unsafe_allow_html=True)

    rows = get_patient_history(current_email)

    if not rows:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <p style="color:rgba(255,255,255,0.7);">No history recorded yet. Enter your first vitals to start tracking.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        recent_oldest_first = list(reversed(rows[:10]))
        scores = [r[7] for r in recent_oldest_first]
        render_trend_line(scores)

        table_rows = ""
        for r in rows:
            recorded_at, hr, bp_sys, bp_dia, spo2, temp, resp_rate, risk_score, risk_category = r
            date_part, time_part = (recorded_at.split("T") + [""])[:2]
            tag_class = "risk-" + risk_category.replace(" ", "")
            table_rows += f"""
            <tr>
                <td>{date_part} {time_part[:5]}</td>
                <td>{int(hr)}</td>
                <td>{int(bp_sys)}/{int(bp_dia)}</td>
                <td>{spo2}%</td>
                <td>{temp}°C</td>
                <td>{int(resp_rate)}</td>
                <td><b>{risk_score}</b></td>
                <td><span class="risk-tag {tag_class}">{risk_category}</span></td>
            </tr>
            """

        st.markdown(f"""
        <div class="glass-card" style="margin-top:20px; overflow-x:auto; text-align:left;">
            <table class="history-table">
                <tr><th>Date</th><th>HR</th><th>BP</th><th>SpO2</th><th>Temp</th><th>RR</th><th>Score</th><th>Status</th></tr>
                {table_rows}
            </table>
        </div>
        """, unsafe_allow_html=True)
