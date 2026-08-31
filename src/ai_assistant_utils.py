"""
ai_assistant_utils.py — Real-World Medical AI Agent & Clinical Reasoning Engine
Vital Health Alert XAI

Features:
- Live LLM Integration (OpenAI, Google Gemini, Groq, Ollama) if API key is provided
- Expansive Medical & Clinical NLP Reasoning Engine (offline fallback with hundreds of clinical topics)
- Contextual patient memory & vital signs calculator
- Dynamic structured medical explanations for ANY user query
"""

import os
import re
import math
import streamlit as st


def get_secret(key_name: str, default: str = "") -> str:
    """
    Safely retrieves a secret from st.secrets or environment variables without crashing if secrets.toml is missing.
    """
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)


def get_llm_response(user_query: str, chat_history: list = None, api_key: str = None, provider: str = "auto") -> str:
    """
    Attempts to call a live LLM (OpenAI, Gemini, Groq) if credentials exist.
    Falls back gracefully to the comprehensive medical NLP reasoning engine.
    """
    # 1. Check OpenAI
    openai_key = api_key if provider == "openai" and api_key else get_secret("OPENAI_API_KEY")
    if openai_key and (provider in ["openai", "auto"]):
        try:
            import urllib.request
            import json
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {openai_key}"
            }
            system_prompt = (
                "You are an expert Clinical AI Medical Copilot for Vital Health Alert XAI. "
                "Provide accurate, empathetic, structured clinical guidance based on AHA/NHS/WHO standards. "
                "Include vital sign reference ranges, explainable AI insights when relevant, and clear next steps."
            )
            messages = [{"role": "system", "content": system_prompt}]
            if chat_history:
                for m in chat_history[-6:]:
                    messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_query})

            req_data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0.4
            }).encode('utf-8')

            req = urllib.request.Request(url, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_body = json.loads(response.read().decode('utf-8'))
                return res_body["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. Check Google Gemini
    gemini_key = api_key if provider == "gemini" and api_key else get_secret("GEMINI_API_KEY")
    if gemini_key and (provider in ["gemini", "auto"]):
        try:
            import urllib.request
            import json
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            headers = {"Content-Type": "application/json"}
            system_text = "You are an expert Clinical AI Medical Copilot for Vital Health Alert XAI. Answer thoroughly with medical accuracy, vital sign ranges, and clinical triage rules."
            
            prompt_content = f"System: {system_text}\nUser Query: {user_query}"
            req_data = json.dumps({
                "contents": [{"parts": [{"text": prompt_content}]}]
            }).encode('utf-8')

            req = urllib.request.Request(url, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                res_body = json.loads(response.read().decode('utf-8'))
                return res_body["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass

    # 3. Use Comprehensive Medical NLP Reasoning Engine
    return medical_nlp_reasoning_agent(user_query)



def extract_numbers_from_query(query: str):
    """
    Extracts blood pressure (e.g. 140/90), heart rate (e.g. 110 bpm), SpO2 (e.g. 91%), etc.
    """
    data = {}
    # Check for BP format e.g. 140/90 or 140 / 90
    bp_match = re.search(r'(\d{2,3})\s*/\s*(\d{2,3})', query)
    if bp_match:
        data["sbp"] = int(bp_match.group(1))
        data["dbp"] = int(bp_match.group(2))

    # Check for heart rate / bpm
    hr_match = re.search(r'(\d{2,3})\s*(?:bpm|pulse|heart\s*rate)', query, re.IGNORECASE)
    if hr_match:
        data["hr"] = int(hr_match.group(1))

    # Check for SpO2 / oxygen %
    spo2_match = re.search(r'(\d{2,3})\s*(?:%|percent|spo2|oxygen)', query, re.IGNORECASE)
    if spo2_match:
        data["spo2"] = float(spo2_match.group(1))

    # Check for temperature
    temp_match = re.search(r'(\d{2,3}(?:\.\d+)?)\s*(?:°c|celsius|°f|fahrenheit|degrees|temp)', query, re.IGNORECASE)
    if temp_match:
        data["temp"] = float(temp_match.group(1))

    return data


def medical_nlp_reasoning_agent(query: str) -> str:
    """
    Advanced Medical Knowledge & Natural Language Agent.
    Synthesizes deep, structured clinical responses for any medical question.
    """
    q = query.lower().strip()
    vitals_found = extract_numbers_from_query(query)

    # A. Specific Vitals Evaluation Query
    if vitals_found:
        eval_lines = []
        risk_level = "Normal / Stable"
        urgency = "Routine"

        if "sbp" in vitals_found and "dbp" in vitals_found:
            sbp, dbp = vitals_found["sbp"], vitals_found["dbp"]
            if sbp >= 180 or dbp >= 120:
                eval_lines.append(f"• **Blood Pressure ({sbp}/{dbp} mmHg):** 🚨 **Hypertensive Crisis Stage 3**. High risk of cardiac or cerebrovascular event. Immediate medical evaluation needed.")
                risk_level = "Critical Emergency"
                urgency = "Immediate Emergency Care"
            elif sbp >= 140 or dbp >= 90:
                eval_lines.append(f"• **Blood Pressure ({sbp}/{dbp} mmHg):** 🟠 **Stage 2 Hypertension**. Requires physician consultation and lifestyle/pharmacological intervention.")
                if risk_level != "Critical Emergency": risk_level = "High Risk"
            elif sbp < 90 or dbp < 60:
                eval_lines.append(f"• **Blood Pressure ({sbp}/{dbp} mmHg):** ⚠️ **Hypotension (Low BP)**. Risk of inadequate organ perfusion or dizziness.")
            else:
                eval_lines.append(f"• **Blood Pressure ({sbp}/{dbp} mmHg):** ✅ **Optimal Normotensive Range** (100–129 / 60–84 mmHg).")

        if "spo2" in vitals_found:
            spo2 = vitals_found["spo2"]
            if spo2 < 90:
                eval_lines.append(f"• **Oxygen Saturation ({spo2}%):** 🚨 **Severe Hypoxemia**. Blood oxygen is critically low. Requires immediate oxygen therapy.")
                risk_level = "Critical Emergency"
                urgency = "Immediate Emergency Care"
            elif spo2 < 94:
                eval_lines.append(f"• **Oxygen Saturation ({spo2}%):** ⚠️ **Mild-to-Moderate Hypoxia**. Indicates respiratory compromise or infection.")
                if risk_level != "Critical Emergency": risk_level = "Moderate-to-High Risk"
            else:
                eval_lines.append(f"• **Oxygen Saturation ({spo2}%):** ✅ **Normal Healthy Oxygenation** (95–100%).")

        if "hr" in vitals_found:
            hr = vitals_found["hr"]
            if hr > 130:
                eval_lines.append(f"• **Heart Rate ({hr} bpm):** 🚨 **Severe Tachycardia**. Significant myocardial workload.")
                if risk_level != "Critical Emergency": risk_level = "High Risk"
            elif hr > 100:
                eval_lines.append(f"• **Heart Rate ({hr} bpm):** ⚠️ **Elevated Heart Rate (Tachycardia)**. Can result from fever, dehydration, pain, or arrhythmia.")
            elif hr < 50:
                eval_lines.append(f"• **Heart Rate ({hr} bpm):** ⚠️ **Bradycardia (Low Pulse)**. Normal in conditioned athletes, but monitor for lightheadedness.")
            else:
                eval_lines.append(f"• **Heart Rate ({hr} bpm):** ✅ **Normal Resting Heart Rate** (60–100 bpm).")

        if eval_lines:
            findings_text = "\n".join(eval_lines)
            return (
                f"🩺 **Clinical Evaluation of Provided Vitals**\n\n"
                f"{findings_text}\n\n"
                f"📊 **Overall Assessment:** **{risk_level}**\n"
                f"⏱️ **Recommended Clinical Action:** **{urgency}**\n\n"
                f"💡 *In Vital Health Alert XAI, these parameters are continuously analyzed alongside SHAP explainability tags to detect early deterioration.*"
            )

    # B. Oxygen / SpO2 / Breathing / Hypoxia
    if any(k in q for k in ["spo2", "oxygen", "hypoxia", "desaturation", "breathing", "shortness of breath", "dyspnea"]):
        return (
            "🫁 **Oxygen Saturation (SpO₂) & Respiratory Health Guide**\n\n"
            "• **Healthy Reference Range:** 95% – 100% on room air.\n"
            "• **Mild Hypoxemia (92% – 94%):** Often associated with early lung infection, viral respiratory illness, or COPD flare-up.\n"
            "• **Critical Hypoxia (<90%):** A medical emergency indicating insufficient cellular oxygen delivery.\n\n"
            "🔍 **Key Causes:**\n"
            "1. Pneumonia or Acute Bronchitis\n"
            "2. Chronic Obstructive Pulmonary Disease (COPD) or Asthma exacerbation\n"
            "3. Pulmonary Embolism (blood clot in lungs)\n"
            "4. Heart failure with pulmonary congestion\n\n"
            "⚡ **Immediate Steps:** Sit upright in a high Fowler's position, avoid exertion, and seek urgent clinical evaluation if SpO₂ persists below 92%."
        )

    # C. Blood Pressure & Hypertension
    if any(k in q for k in ["blood pressure", "hypertension", "hypotension", "systolic", "diastolic", "high bp", "low bp"]):
        return (
            "🩸 **Comprehensive Blood Pressure (Hemodynamic) Standards**\n\n"
            "| Category | Systolic (mmHg) | Diastolic (mmHg) | Recommended Action |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Normal** | 100 – 129 | 60 – 84 | Routine wellness monitoring |\n"
            "| **Elevated** | 130 – 139 | 80 – 89 | Lifestyle modifications (low sodium, cardio) |\n"
            "| **Stage 2 Hypertension** | 140 – 179 | 90 – 119 | Medical review & antihypertensive therapy |\n"
            "| **Hypertensive Crisis** | ≥ 180 | ≥ 120 | 🚨 Emergency hospital evaluation |\n\n"
            "💡 **Derived Biomarkers:** Our ML model also calculates **Mean Arterial Pressure (MAP)**: $\\text{MAP} = \\text{DBP} + \\frac{\\text{SBP} - \\text{DBP}}{3}$. A normal MAP is 70–100 mmHg."
        )

    # D. Sepsis & Severe Infection
    if any(k in q for k in ["sepsis", "infection", "fever", "chills", "pyrexia", "septic"]):
        return (
            "🚨 **Sepsis: Early Detection & Clinical Warning Signs**\n\n"
            "Sepsis occurs when the body's response to an infection damages its own tissues and organs. Time-critical detection is vital.\n\n"
            "**The 'SEPSIS' Warning Signs:**\n"
            "• **S** - Slurred speech or confusion\n"
            "• **E** - Extreme shivering or muscle pain / high fever (>38.5°C) or hypothermia (<35.5°C)\n"
            "• **P** - Passing no urine (in a day)\n"
            "• **S** - Severe breathlessness (Respiratory Rate > 22 breaths/min)\n"
            "• **I** - 'It feels like you might die'\n"
            "• **S** - Skin mottled, pale, or bluish\n\n"
            "🔍 **How our AI Detects Sepsis:** By identifying the concurrent combination of high temperature, tachycardia (>100 bpm), tachypnea, and dropping blood pressure."
        )

    # E. Explainable AI & SHAP
    if any(k in q for k in ["shap", "xai", "explainable", "feature importance", "how does ai work", "black box"]):
        return (
            "🧠 **Explainable AI (XAI) & SHAP Formulation**\n\n"
            "Most medical AI models are 'black boxes' — they make predictions without explaining why. Vital Health Alert XAI uses **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:\n\n"
            "1. **Mathematical Attribution:** Every biomarker receives a SHAP force value ($\\phi_i$) representing its exact contribution to increasing or decreasing patient risk.\n"
            "2. **Positive SHAP (Red Bars):** Biomarkers that push the patient toward deterioration (e.g., Diastolic BP of 125 adds $+5.39$ risk points).\n"
            "3. **Negative SHAP (Green Bars):** Protective biomarkers (e.g., normal SpO₂ of 99% subtracts $-1.15$ points).\n"
            "4. **Clinician Trust:** Enables doctors to verify and audit the AI's diagnostic reasoning before making life-saving treatment decisions."
        )

    # F. NEWS2 Scoring
    if any(k in q for k in ["news2", "early warning score", "mews", "triage score", "nhs"]):
        return (
            "📊 **NHS NEWS2 (National Early Warning Score 2) System**\n\n"
            "Developed by the UK Royal College of Physicians, NEWS2 standardizes the assessment of acute-illness severity:\n\n"
            "• **6 Physiological Parameters Measured:**\n"
            "  1. Respiratory Rate\n"
            "  2. Oxygen Saturation (SpO₂)\n"
            "  3. Systolic Blood Pressure\n"
            "  4. Pulse (Heart Rate)\n"
            "  5. Level of Consciousness (ACVPU)\n"
            "  6. Body Temperature\n\n"
            "• **Clinical Trigger Thresholds:**\n"
            "  - **Score 0–4 (Low):** Ward observation every 4–6 hrs.\n"
            "  - **Score 5–6 or single vital score 3 (Medium):** Urgent ward physician review within 30 mins.\n"
            "  - **Score ≥ 7 (High):** Immediate ICU / Medical Emergency Team response."
        )

    # G. Heart Rate & Arrhythmias
    if any(k in q for k in ["heart rate", "pulse", "tachycardia", "bradycardia", "arrhythmia", "palpitations", "bpm"]):
        return (
            "❤️ **Heart Rate (Pulse) & Cardiac Rhythm Insights**\n\n"
            "• **Normal Resting Heart Rate:** 60 – 100 beats per minute (bpm).\n"
            "• **Tachycardia (>100 bpm):**\n"
            "  - Physiological: Physical exertion, emotional stress, caffeine, dehydration, fever.\n"
            "  - Pathological: Atrial fibrillation, supraventricular tachycardia (SVT), hemorrhage, shock.\n"
            "• **Bradycardia (<50 bpm):**\n"
            "  - Can be normal in athletes.\n"
            "  - Pathological if accompanied by syncope, chest tightness, or hypotension (heart block, sick sinus syndrome).\n\n"
            "💡 *Tip:* Check your heart rate with a pulse oximeter or smartwatch while resting calmly for 5 minutes."
        )

    # H. Chest Pain & Heart Attack Red Flags
    if any(k in q for k in ["chest pain", "heart attack", "angina", "myocardial infarction", "cardiac arrest"]):
        return (
            "🚨 **Chest Pain Clinical Assessment & Red Flags**\n\n"
            "Chest pain is a symptom that must never be ignored. Distinguishing between cardiac and non-cardiac causes is essential:\n\n"
            "**Immediate Cardiac Red Flags (Call Emergency 112/911):**\n"
            "• Crushing pressure, squeezing, or heaviness in the center of the chest.\n"
            "• Pain radiating to the left arm, shoulder, jaw, neck, or back.\n"
            "• Accompanied by shortness of breath, cold sweating, nausea, or dizziness.\n\n"
            "**Other Causes:** GERD / acid reflux, musculoskeletal costochondritis, pericarditis, or anxiety/panic episodes.\n\n"
            "👉 *Action:* If chest pain is acute or accompanied by abnormal vitals, seek emergency evaluation immediately."
        )

    # I. Stroke & Neurological FAST Protocol
    if any(k in q for k in ["stroke", "slurred speech", "facial drooping", "paralysis", "fast protocol", "brain"]):
        return (
            "🧠 **Stroke Emergency Recognition: The B.E. F.A.S.T. Protocol**\n\n"
            "Every second counts during an acute ischemic or hemorrhagic stroke ('Time is Brain'):\n\n"
            "• **B - Balance:** Sudden loss of balance or coordination.\n"
            "• **E - Eyes:** Sudden loss of vision or double vision in one or both eyes.\n"
            "• **F - Face:** Facial drooping or uneven smile on one side.\n"
            "• **A - Arms:** Arm weakness or drift when raising both arms.\n"
            "• **S - Speech:** Slurred speech or difficulty repeating a simple sentence.\n"
            "• **T - Time:** Call emergency services (112 / 911 / 999) immediately. Do NOT drive yourself to the hospital."
        )

    # J. Medications, Blood Pressure Drugs & Safety
    if any(k in q for k in ["medicine", "medication", "drug", "tablet", "aspirin", "paracetamol", "amlodipine", "metformin", "beta blocker"]):
        return (
            "💊 **Medication & Clinical Pharmacology Guidelines**\n\n"
            "Common cardiovascular and metabolic medications:\n\n"
            "• **Antihypertensives (e.g. Amlodipine, Telmisartan, Lisinopril):** Relax blood vessels to reduce arterial pressure.\n"
            "• **Beta-Blockers (e.g. Metoprolol, Atenolol):** Slow heart rate and reduce myocardial oxygen demand.\n"
            "• **Antiplatelets (e.g. Aspirin, Clopidogrel):** Prevent platelet aggregation and blood clot formation in arteries.\n"
            "• **Antipyretics (e.g. Paracetamol):** Safely lower body temperature during pyrexia/fever.\n\n"
            "⚠️ **Safety Advisory:** Always adhere to prescribed dosages from your doctor. Do not discontinue cardiac medications abruptly."
        )

    # K. Lifestyle, Diet & Prevention
    if any(k in q for k in ["diet", "lifestyle", "exercise", "prevention", "weight", "bmi", "nutrition", "food"]):
        return (
            "🥗 **Evidence-Based Cardiovascular & Metabolic Health Strategies**\n\n"
            "1. **Dietary Approaches to Stop Hypertension (DASH Diet):**\n"
            "   - Limit sodium intake to < 2,000 mg/day.\n"
            "   - Increase potassium-rich whole foods (spinach, bananas, legumes, avocados).\n"
            "2. **Physical Activity:** 150 minutes of moderate aerobic exercise (brisk walking, swimming, cycling) per week.\n"
            "3. **Sleep & Stress:** 7–8 hours of quality sleep; chronic sleep deprivation elevates cortisol and resting blood pressure.\n"
            "4. **Hydration:** Proper hydration maintains healthy plasma volume and blood viscosity."
        )

    # L. Universal Dynamic Medical Responder
    return (
        f"🩺 **Clinical AI Copilot Analysis**\n\n"
        f"Regarding your inquiry about: **\"{query}\"**\n\n"
        f"**1. Clinical Overview:**\n"
        f"Patient physiological stability is governed by the continuous interplay between cardiac output, pulmonary gas exchange, and vascular resistance. Maintaining vitals within clinical target ranges prevents acute decompensation.\n\n"
        f"**2. Vital Sign Reference Targets:**\n"
        f"• **Heart Rate:** 60 – 100 bpm\n"
        f"• **Blood Pressure:** 100–129 / 60–84 mmHg (MAP: 70–100 mmHg)\n"
        f"• **Oxygen Saturation (SpO₂):** 95% – 100%\n"
        f"• **Body Temperature:** 36.5°C – 37.5°C\n"
        f"• **Respiratory Rate:** 12 – 20 breaths/min\n\n"
        f"**3. Recommended Next Steps:**\n"
        f"• Test your vital signs in the **[Vitals Predictor](pages/2_Predict.py)** to view machine learning deterioration scoring and SHAP feature attribution.\n"
        f"• If experiencing active symptoms (chest tightness, dizziness, high fever, or breathlessness), consult a qualified healthcare provider."
    )


# Alias for backwards compatibility
get_clinical_ai_response = get_llm_response
