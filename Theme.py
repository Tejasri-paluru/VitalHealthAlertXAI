"""
Theme.py — Shared High-Contrast Visual Identity for Vital Health Alert XAI
Crystal-clear typography, crisp white headings, high-contrast parameter labels,
glowing badges, and integrated top auth header widget.
"""

import streamlit as st

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg-main: #041B24;
    --bg-grad-start: #031c28;
    --bg-grad-end: #073542;
    --card-bg: rgba(6, 44, 56, 0.85);
    --card-hover: rgba(8, 58, 74, 0.95);
    --card-border: rgba(2, 195, 154, 0.35);
    --card-border-glow: rgba(2, 195, 154, 0.7);
    
    --primary: #02C39A;
    --primary-glow: rgba(2, 195, 154, 0.5);
    --accent: #00A896;
    --cream: #F0F3BD;
    --text: #FFFFFF;
    --text-muted: rgba(245, 250, 248, 0.85);
    
    --healthy: #02C39A;
    --moderate: #F0F3BD;
    --high: #F2A65A;
    --critical: #E8543E;
}

/* Base Viewport & Body */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 15%, rgba(0, 168, 150, 0.16) 0%, transparent 40%),
                radial-gradient(circle at 80% 85%, rgba(2, 195, 154, 0.12) 0%, transparent 45%),
                linear-gradient(180deg, var(--bg-grad-start) 0%, #042533 50%, var(--bg-grad-end) 100%) fixed !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1260px !important;
}

#MainMenu, footer { visibility: hidden !important; }

/* HIGH-CONTRAST HEADINGS — 100% PURE WHITE */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.4) !important;
}

/* HIGH-CONTRAST FORM LABELS — Parameter Names (Heart Rate, BP, Age, Sex, etc.) */
label, 
label p, 
label span, 
[data-testid="stWidgetLabel"] p, 
[data-testid="stWidgetLabel"] span,
.stSelectbox label, 
.stNumberInput label, 
.stTextInput label, 
.stSlider label, 
.stRadio label p,
.stCheckbox label p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.96rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.6) !important;
}

/* Number inputs, text inputs, select boxes text */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    color: #FFFFFF !important;
    background-color: rgba(6, 44, 56, 0.95) !important;
    border: 1px solid rgba(2, 195, 154, 0.4) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}

.mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Section Labels & Headings */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--primary);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 800;
    margin-bottom: 0.35rem;
    display: inline-block;
    background: rgba(2, 195, 154, 0.12);
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    border: 1px solid rgba(2, 195, 154, 0.3);
}

.section-title {
    font-size: 2rem;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 1.2rem;
    line-height: 1.25;
}

/* Glassmorphism Cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.5rem 1.6rem;
    backdrop-filter: blur(14px);
    transition: all 0.25s ease;
    margin-bottom: 1.2rem;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
}

.glass-card:hover {
    background: var(--card-hover);
    border-color: var(--card-border-glow);
    box-shadow: 0 16px 40px rgba(2, 195, 154, 0.2);
    transform: translateY(-2px);
}

/* Glowing Risk Pills */
.risk-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1.1rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(6, 44, 56, 0.9);
    border: 1.5px solid var(--card-border);
    color: #FFFFFF;
}

.dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.35); opacity: 0.7; }
}

/* Streamlit Buttons */
div.stButton > button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: all 0.2s ease !important;
    border: 1.5px solid var(--card-border) !important;
    background: rgba(6, 44, 56, 0.85) !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
}

div.stButton > button:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 18px var(--primary-glow) !important;
    transform: translateY(-2px) !important;
    color: #F0F3BD !important;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #00A896 0%, #02C39A 100%) !important;
    color: #04241F !important;
    border: none !important;
    box-shadow: 0 0 22px rgba(2, 195, 154, 0.45) !important;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 32px rgba(2, 195, 154, 0.75) !important;
    transform: translateY(-2px) !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--cream) !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

[data-testid="stMetricLabel"] p {
    color: var(--primary) !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    color: rgba(255, 255, 255, 0.75) !important;
    background: transparent !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 0.8rem 1.4rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #02C39A !important;
    border-bottom-color: #02C39A !important;
    background: rgba(2, 195, 154, 0.08) !important;
}

/* Wave divider */
.wave-divider {
    width: 100%;
    height: 28px;
    margin: 2.2rem 0;
    opacity: 0.7;
}
.wave-divider svg {
    width: 100%;
    height: 100%;
    filter: drop-shadow(0 0 6px rgba(2, 195, 154, 0.45));
}
</style>
"""


def inject_base_css():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def wave_divider():
    st.markdown(
        """
        <div class="wave-divider">
            <svg viewBox="0 0 1200 28" preserveAspectRatio="none">
                <path d="M0,14 Q60,0 120,14 T240,14 T360,14 T480,14 T600,14 T720,14 T840,14 T960,14 T1080,14 T1200,14"
                      fill="none" stroke="url(#g1)" stroke-width="2.4"/>
                <defs>
                    <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0">
                        <stop offset="0%" stop-color="#00A896"/>
                        <stop offset="100%" stop-color="#02C39A"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_auth_header():
    """
    Renders top status bar across all pages showing auth state, user profile, and quick actions.
    """
    user_email = st.session_state.get("current_user_email", "")

    header_col1, header_col2 = st.columns([3, 1.2])

    with header_col1:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; padding: 0.2rem 0;">
                <span style="font-size:1.4rem;">🫀</span>
                <span style="font-family:'Space Grotesk'; font-weight:800; font-size:1.15rem; color:#FFFFFF; letter-spacing:-0.02em;">
                    Vital Health Alert <span style="color:#02C39A;">XAI</span>
                </span>
                <span style="font-family:'JetBrains Mono'; font-size:0.72rem; background:rgba(2,195,154,0.15); border:1px solid rgba(2,195,154,0.35); color:#02C39A; padding:2px 8px; border-radius:999px; font-weight:700;">
                    CLINICAL DECISION SUPPORT
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with header_col2:
        if user_email:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:flex-end; gap:10px; font-family:'Inter'; font-size:0.85rem;">
                    <span style="color:#02C39A;">👤 <b>{user_email.split('@')[0]}</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )
            h_c1, h_c2 = st.columns(2)
            with h_c1:
                if st.button("👤 Profile", key="hdr_profile_btn", use_container_width=True):
                    st.switch_page("pages/4_Account.py")
            with h_c2:
                if st.button("🚪 Logout", key="hdr_logout_btn", use_container_width=True):
                    st.session_state.current_user_email = ""
                    st.session_state.last_prediction = None
                    st.rerun()
        else:
            if st.button("🔐 Login / Sign Up", key="hdr_auth_btn", use_container_width=True):
                st.switch_page("pages/4_Account.py")

    st.write("")