"""
3_AI_Assistant.py — Real-World Medical AI Agent & Clinical Copilot
Vital Health Alert XAI

Features:
- ChatGPT-Style Conversational Clinical AI Copilot
- Live LLM Integration (OpenAI, Gemini, Groq) or Advanced Offline Medical NLP Brain
- High-contrast glowing chat UI with clear visibility
- One-click clinical inquiries & prompt chips
"""

import streamlit as st
import time
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Theme import inject_base_css, wave_divider, render_auth_header
from ai_assistant_utils import get_llm_response

st.set_page_config(
    page_title="AI Clinical Copilot — Vital Health Alert XAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_base_css()
render_auth_header()

# Custom High-Contrast Chat Styling
st.markdown(
    """
    <style>
    /* Chat Message Bubbles */
    .user-chat-bubble {
        background: linear-gradient(135deg, rgba(0, 168, 150, 0.35) 0%, rgba(2, 195, 154, 0.2) 100%);
        border: 1.5px solid rgba(2, 195, 154, 0.5);
        border-radius: 18px 18px 4px 18px;
        padding: 1.2rem 1.4rem;
        color: #FFFFFF !important;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .ai-chat-bubble {
        background: rgba(6, 44, 56, 0.92);
        border: 1.5px solid rgba(2, 195, 154, 0.4);
        border-left: 5px solid #02C39A;
        border-radius: 18px 18px 18px 4px;
        padding: 1.3rem 1.5rem;
        color: #FFFFFF !important;
        font-size: 0.98rem;
        line-height: 1.65;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    .ai-chat-bubble p, .ai-chat-bubble li, .ai-chat-bubble span, .ai-chat-bubble strong {
        color: #FFFFFF !important;
    }

    .ai-chat-bubble table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
        font-size: 0.9rem;
    }
    .ai-chat-bubble th {
        color: #02C39A;
        border-bottom: 2px solid #02C39A;
        padding: 6px;
        text-align: left;
    }
    .ai-chat-bubble td {
        padding: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        color: #FFFFFF;
    }

    /* Chat input styling */
    div[data-testid="stChatInput"] textarea {
        color: #FFFFFF !important;
        background-color: rgba(6, 44, 56, 0.95) !important;
        border: 1.5px solid rgba(2, 195, 154, 0.6) !important;
        border-radius: 14px !important;
        font-size: 1rem !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Banner
st.markdown(
    """
    <div style="background: linear-gradient(135deg, rgba(6, 44, 56, 0.85) 0%, rgba(4, 27, 36, 0.95) 100%); border: 1px solid rgba(2, 195, 154, 0.25); border-radius: 20px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;">
        <div class="section-label">Intelligent Health Copilot</div>
        <h1 style="margin: 0.2rem 0 0.5rem 0; font-size:2.2rem;">🤖 Real-World Clinical AI Medical Agent</h1>
        <p style="color:rgba(245,250,248,0.85); margin:0; font-size:0.96rem;">
            A ChatGPT-style Medical AI Copilot powered by clinical guidelines (AHA, NHS NEWS2, WHO) and Explainable AI.
            Ask any question about vital signs, disease symptoms, medication precautions, or SHAP risk attribution.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ AI Engine Settings")
    ai_provider = st.selectbox(
        "AI Engine Mode:",
        ["🧠 Internal Medical NLP Brain (Offline / Fast)", "⚡ Live Google Gemini", "🌐 Live OpenAI (ChatGPT)", "🚀 Live Groq (Llama-3)"],
        index=0
    )

    custom_key = ""
    if "Live" in ai_provider:
        st.caption("Enter your API key below or set it in `.streamlit/secrets.toml`:")
        custom_key = st.text_input("API Key (Optional)", type="password", placeholder="sk-... / AIzaSy...")

    provider_code = "auto"
    if "Gemini" in ai_provider: provider_code = "gemini"
    elif "OpenAI" in ai_provider: provider_code = "openai"
    elif "Groq" in ai_provider: provider_code = "groq"
    elif "Internal" in ai_provider: provider_code = "internal"

    st.write("---")
    st.markdown("### 💡 Quick Prompt Topics")
    if st.button("🫁 Low SpO₂ (<90%) Alert", use_container_width=True, key="sb_q1"):
        st.session_state["chat_query_trigger"] = "What does SpO2 below 90% mean and why is it considered a clinical emergency?"
    if st.button("🩸 High BP (160/100) Review", use_container_width=True, key="sb_q2"):
        st.session_state["chat_query_trigger"] = "My blood pressure is 160/100 mmHg and heart rate is 105 bpm. What does this mean?"
    if st.button("🚨 Sepsis Early Signs", use_container_width=True, key="sb_q3"):
        st.session_state["chat_query_trigger"] = "What are the earliest vital sign red flags for sepsis in a hospital or home patient?"
    if st.button("🧠 How SHAP Explains Risk", use_container_width=True, key="sb_q4"):
        st.session_state["chat_query_trigger"] = "How does SHAP (Explainable AI) calculate risk contributions for each vital sign?"
    if st.button("📊 NHS NEWS2 System", use_container_width=True, key="sb_q5"):
        st.session_state["chat_query_trigger"] = "Explain the NHS NEWS2 deterioration score and clinical escalation thresholds."

    st.write("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True, key="sb_clear_btn"):
        st.session_state.ai_agent_messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! I am your **Clinical AI Medical Copilot**. Ask me any question about your vital signs, health symptoms, explainable AI predictions, or medical guidelines. How can I assist you?"
            }
        ]
        st.rerun()

# Initialize Chat Memory
if "ai_agent_messages" not in st.session_state:
    st.session_state.ai_agent_messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am your **Clinical AI Medical Copilot**. Ask me any question about your vital signs, health symptoms, explainable AI predictions, or medical guidelines. How can I assist you today?"
        }
    ]

# Render Chat History
chat_container = st.container()
with chat_container:
    for msg in st.session_state.ai_agent_messages:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; margin-bottom:0.8rem;">
                    <div class="user-chat-bubble" style="max-width:80%;">
                        <div style="font-size:0.8rem; font-weight:800; color:#02C39A; margin-bottom:4px; font-family:'JetBrains Mono';">👤 YOU</div>
                        <div>{msg['content']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-start; margin-bottom:1rem;">
                    <div class="ai-chat-bubble" style="max-width:90%;">
                        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                            <span style="font-size:1.2rem;">🩺</span>
                            <span style="font-size:0.85rem; font-weight:800; color:#02C39A; font-family:'JetBrains Mono';">CLINICAL AI COPILOT</span>
                        </div>
                        <div>{msg['content']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# Process Prompt Trigger or Chat Input
user_chat_text = st.chat_input("Ask a medical question, enter vital signs (e.g. 150/95 BP, 110 HR), or ask how SHAP works...")

prompt_to_send = None
if "chat_query_trigger" in st.session_state:
    prompt_to_send = st.session_state.pop("chat_query_trigger")
elif user_chat_text:
    prompt_to_send = user_chat_text

if prompt_to_send:
    # Append User Message
    st.session_state.ai_agent_messages.append({"role": "user", "content": prompt_to_send})

    # Generate Response
    with st.spinner("🤖 Clinical AI Agent is reasoning and synthesizing guidance..."):
        ai_resp = get_llm_response(
            prompt_to_send,
            chat_history=st.session_state.ai_agent_messages,
            api_key=custom_key,
            provider=provider_code
        )

    st.session_state.ai_agent_messages.append({"role": "assistant", "content": ai_resp})
    st.rerun()

wave_divider()
st.caption("🛡️ **Clinical Disclaimer:** Vital Health Alert XAI provides evidence-based medical decision support. It does not replace emergency clinical diagnosis. In life-threatening situations, contact emergency services (112 / 911 / 999) immediately.")
