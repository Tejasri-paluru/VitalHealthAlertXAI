"""
predict_utils.py — Clinical Intelligence & Explainable AI Utility Engine
Vital Health Alert XAI

Integrates:
- LightGBM & Random Forest ML Classifiers
- SHAP TreeExplainer for feature attribution
- NHS NEWS2 (National Early Warning Score 2) clinical rules
- Isolation Forest Anomaly Detection
- Plain-English AI Clinical Consultation Summaries
- Symptom-to-Vitals Physiological Estimation
- Database management & Benchmark Patient Trajectory Library
"""

import os
import sqlite3
import joblib
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "database", "vital_health.db")

# Ensure database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Load Models
try:
    lgb_model = joblib.load(os.path.join(MODELS_DIR, "lgb_model.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    iso_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    try:
        rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    except Exception:
        rf_model = None
except Exception:
    lgb_model = None
    feature_names = [
        'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Gender',
        'Weight (kg)', 'Height (m)', 'Derived_HRV', 'Derived_Pulse_Pressure',
        'Derived_BMI', 'Derived_MAP'
    ]
    iso_forest = None
    rf_model = None

# Initialize SHAP
import shap
if lgb_model is not None:
    try:
        explainer = shap.TreeExplainer(lgb_model)
    except Exception:
        explainer = None
else:
    explainer = None


def compute_news2(hr, rr, spo2, sbp, temp):
    """
    Standard NHS NEWS2 calculation.
    Returns: total_score (int), breakdown (dict), risk_level (str)
    """
    breakdown = {}

    # RR
    if rr <= 8:
        breakdown['Respiratory Rate'] = 3
    elif 9 <= rr <= 11:
        breakdown['Respiratory Rate'] = 1
    elif 12 <= rr <= 20:
        breakdown['Respiratory Rate'] = 0
    elif 21 <= rr <= 24:
        breakdown['Respiratory Rate'] = 2
    else:
        breakdown['Respiratory Rate'] = 3

    # SpO2
    if spo2 >= 96:
        breakdown['Oxygen Saturation'] = 0
    elif 94 <= spo2 <= 95:
        breakdown['Oxygen Saturation'] = 1
    elif 92 <= spo2 <= 93:
        breakdown['Oxygen Saturation'] = 2
    else:
        breakdown['Oxygen Saturation'] = 3

    # SBP
    if sbp <= 90:
        breakdown['Systolic BP'] = 3
    elif 91 <= sbp <= 100:
        breakdown['Systolic BP'] = 2
    elif 101 <= sbp <= 110:
        breakdown['Systolic BP'] = 1
    elif 111 <= sbp <= 159:
        breakdown['Systolic BP'] = 0
    elif 160 <= sbp <= 179:
        breakdown['Systolic BP'] = 1
    elif 180 <= sbp <= 219:
        breakdown['Systolic BP'] = 2
    else:
        breakdown['Systolic BP'] = 3

    # HR
    if hr <= 40:
        breakdown['Heart Rate'] = 3
    elif 41 <= hr <= 50:
        breakdown['Heart Rate'] = 1
    elif 51 <= hr <= 90:
        breakdown['Heart Rate'] = 0
    elif 91 <= hr <= 110:
        breakdown['Heart Rate'] = 1
    elif 111 <= hr <= 130:
        breakdown['Heart Rate'] = 2
    else:
        breakdown['Heart Rate'] = 3

    # Temp
    if temp <= 35.0:
        breakdown['Body Temp'] = 3
    elif 35.1 <= temp <= 36.0:
        breakdown['Body Temp'] = 1
    elif 36.1 <= temp <= 38.0:
        breakdown['Body Temp'] = 0
    elif 38.1 <= temp <= 39.0:
        breakdown['Body Temp'] = 1
    else:
        breakdown['Body Temp'] = 2

    total_news2 = sum(breakdown.values())

    # NEWS2 Clinical Risk Category
    if total_news2 == 0:
        level = "Low (Routine Monitoring)"
    elif 1 <= total_news2 <= 4:
        level = "Low (Ward/Clinic Review)"
    elif 5 <= total_news2 <= 6 or any(v == 3 for v in breakdown.values()):
        level = "Medium (Urgent Clinical Review)"
    else:
        level = "High (Emergency / Critical Care Response)"

    return total_news2, breakdown, level


def estimate_vitals_from_symptoms(base_hr, base_spo2, base_sbp, base_dbp, base_temp, base_rr, symptoms_dict):
    """
    Clinically estimates realistic physiological vital shifts when a patient reports
    acute symptoms without specialized hospital monitors.
    """
    adj_hr = float(base_hr)
    adj_spo2 = float(base_spo2)
    adj_sbp = float(base_sbp)
    adj_dbp = float(base_dbp)
    adj_temp = float(base_temp)
    adj_rr = float(base_rr)

    # 1. Shortness of breath
    if symptoms_dict.get('sob'):
        adj_rr = max(adj_rr, 26.0)
        adj_spo2 = min(adj_spo2, 91.5)
        adj_hr = max(adj_hr, 102.0)

    # 2. Chest pain / tightness
    if symptoms_dict.get('chest'):
        adj_sbp = max(adj_sbp, 168.0)
        adj_dbp = max(adj_dbp, 102.0)
        adj_hr = max(adj_hr, 114.0)

    # 3. Dizziness / Lightheadedness / Weakness
    if symptoms_dict.get('dizzy'):
        if not symptoms_dict.get('chest'):
            adj_sbp = min(adj_sbp, 88.0)  # Hypotension
            adj_dbp = min(adj_dbp, 54.0)
        adj_hr = max(adj_hr, 108.0)

    # 4. Fever / Chills
    if symptoms_dict.get('fever'):
        adj_temp = max(adj_temp, 39.2)
        adj_hr = max(adj_hr, 116.0)
        adj_rr = max(adj_rr, 22.0)

    # 5. Cyanosis / Blue lips
    if symptoms_dict.get('cyanosis'):
        adj_spo2 = min(adj_spo2, 84.0)
        adj_rr = max(adj_rr, 30.0)

    # 6. Confusion / Slurred speech
    if symptoms_dict.get('confusion'):
        adj_sbp = max(adj_sbp, 195.0)
        adj_dbp = max(adj_dbp, 118.0)

    return adj_hr, adj_spo2, adj_sbp, adj_dbp, adj_temp, adj_rr


def calculate_health_risk_index(probability, news2_score, input_dict):
    """
    Computes a composite, calibrated 0-100 Health Risk Index combining ML and physiological markers.
    """
    ml_score = probability * 100.0
    news2_score_component = min(news2_score * 12.0, 100.0)
    composite = (0.65 * ml_score) + (0.35 * news2_score_component)

    # Emergency vital overrides
    spo2 = input_dict.get('Oxygen Saturation', 98)
    sbp = input_dict.get('Systolic Blood Pressure', 120)
    dbp = input_dict.get('Diastolic Blood Pressure', 80)
    temp = input_dict.get('Body Temperature', 37.0)
    hr = input_dict.get('Heart Rate', 75)

    if spo2 < 88 or sbp > 190 or dbp > 120 or temp > 40.0 or hr > 140 or sbp < 80:
        composite = max(composite, 82.0)
    elif spo2 < 93 or sbp > 165 or dbp > 105 or temp > 38.8 or hr > 115 or sbp < 95:
        composite = max(composite, 62.0)
    elif spo2 >= 96 and 105 <= sbp <= 135 and 65 <= dbp <= 85 and 36.2 <= temp <= 37.4 and 55 <= hr <= 85:
        composite = min(composite, 25.0)

    risk_score = round(float(np.clip(composite, 0.0, 100.0)), 1)

    if risk_score <= 30.0:
        category = "Healthy / Low Risk"
        color = "#02C39A"
    elif risk_score <= 60.0:
        category = "Moderate Risk"
        color = "#F0F3BD"
    elif risk_score <= 80.0:
        category = "High Risk"
        color = "#F2A65A"
    else:
        category = "Critical Emergency"
        color = "#E8543E"

    return risk_score, category, color


def get_clinical_explanation_tag(feature, value, contribution):
    """
    Translates SHAP feature values into clear, clinician-friendly pathology notes.
    """
    direction = "Increases Risk" if contribution > 0 else "Decreases Risk"

    note = ""
    if feature == "Oxygen Saturation":
        if value < 90:
            note = "Severe Hypoxemia (<90% SpO2)"
        elif value < 94:
            note = "Mild-to-Moderate Hypoxia"
        else:
            note = "Optimal Oxygenation (Normal)"
    elif feature == "Systolic Blood Pressure":
        if value >= 180:
            note = "Hypertensive Crisis Stage 3"
        elif value >= 140:
            note = "Stage 2 Hypertension"
        elif value < 90:
            note = "Hypotension / Shock danger"
        else:
            note = "Normotensive Range"
    elif feature == "Diastolic Blood Pressure":
        if value >= 120:
            note = "Severe Diastolic Hypertension"
        elif value < 60:
            note = "Low Diastolic Pressure"
        else:
            note = "Normal Diastolic"
    elif feature == "Heart Rate":
        if value > 130:
            note = "Severe Tachycardia (>130 bpm)"
        elif value > 100:
            note = "Elevated Heart Rate"
        elif value < 50:
            note = "Bradycardia (<50 bpm)"
        else:
            note = "Normal Resting Heart Rate"
    elif feature == "Body Temperature":
        if value >= 39.5:
            note = "High Grade Pyrexia / Severe Fever"
        elif value >= 38.0:
            note = "Fever / Systemic Inflammatory Response"
        elif value < 35.5:
            note = "Hypothermia Risk"
        else:
            note = "Afebrile (Normal Body Temp)"
    elif feature == "Respiratory Rate":
        if value >= 25:
            note = "Severe Tachypnea / Respiratory Distress"
        elif value >= 21:
            note = "Elevated Breathing Rate"
        elif value < 10:
            note = "Bradypnea / Hypoventilation"
        else:
            note = "Normal Breathing Rate"
    elif feature == "Derived_BMI":
        if value >= 35:
            note = "Class II/III Obesity"
        elif value < 18.5:
            note = "Underweight"
        else:
            note = "Healthy Weight Range"
    elif feature == "Derived_MAP":
        if value > 110:
            note = "Elevated Mean Arterial Pressure"
        elif value < 65:
            note = "Low Organ Perfusion Pressure"
        else:
            note = "Normal Perfusion Pressure"
    else:
        note = "Physiological Indicator"

    return direction, note


def generate_ai_clinical_summary(input_dict, risk_score, risk_category, news2_score, news2_level, is_anomaly, top_factors):
    """
    Generates a structured, professional AI Clinical Decision Support summary.
    """
    hr = input_dict.get('Heart Rate', 75)
    spo2 = input_dict.get('Oxygen Saturation', 98)
    sbp = input_dict.get('Systolic Blood Pressure', 120)
    dbp = input_dict.get('Diastolic Blood Pressure', 80)
    temp = input_dict.get('Body Temperature', 37.0)
    rr = input_dict.get('Respiratory Rate', 16)

    abnormalities = []
    if spo2 < 94:
        abnormalities.append(f"SpO₂ is depressed at {spo2}% (Normal: 95-100%)")
    if sbp >= 140 or dbp >= 90:
        abnormalities.append(f"Blood pressure is elevated at {sbp}/{dbp} mmHg")
    elif sbp < 90:
        abnormalities.append(f"Blood pressure is dangerously low at {sbp}/{dbp} mmHg")
    if hr > 100:
        abnormalities.append(f"Tachycardia detected ({hr} bpm)")
    elif hr < 50:
        abnormalities.append(f"Bradycardia detected ({hr} bpm)")
    if temp >= 38.0:
        abnormalities.append(f"Pyrexia/fever recorded ({temp}°C)")
    elif temp < 35.5:
        abnormalities.append(f"Hypothermia alert ({temp}°C)")
    if rr >= 22:
        abnormalities.append(f"Tachypnea noted ({rr} breaths/min)")

    if not abnormalities:
        findings_str = "All primary vital signs (SpO₂, Blood Pressure, Heart Rate, Temperature, Respiratory Rate) reside within standard healthy physiological reference ranges."
    else:
        findings_str = "; ".join(abnormalities) + "."

    if risk_score > 80:
        action = "🚨 **EMERGENCY ACTION**: Immediate medical evaluation and rapid response team escalation required. Oxygen therapy, continuous telemetry, and urgent physician triage recommended."
    elif risk_score > 60:
        action = "⚠️ **URGENT CLINICAL REVIEW**: Patient demonstrates significant physiological instability. Escalation to ward physician, frequent vitals monitoring (every 30-60 mins), and diagnostic workup advised."
    elif risk_score > 30:
        action = "🔍 **MONITOR & RE-EVALUATE**: Patient exhibits mild vital variances. Routine vitals monitoring every 4-6 hours, hydration review, and symptom monitoring indicated."
    else:
        action = "✅ **ROUTINE STATUS**: Patient is clinically stable. Continue routine wellness observation or standard periodic health checkup."

    summary = {
        "status_headline": f"{risk_category} (Score: {risk_score}/100 | NEWS2: {news2_score})",
        "findings": findings_str,
        "action": action,
        "anomaly_note": "Isolation Forest flagged atypical multi-parameter variance." if is_anomaly == "Yes" else "Vitals conform to standard statistical cohort distributions.",
        "primary_driver": top_factors[0][0] if top_factors else "Vital Signs"
    }

    return summary


def predict_patient(input_dict):
    """
    Main prediction entry point.
    Returns complete clinical package: prediction, risk_score, risk_category,
    color, news2_score, news2_breakdown, news2_level, is_anomaly, top_factors,
    ai_summary, probability.
    """
    patient_df = pd.DataFrame([input_dict])[feature_names]

    if lgb_model is not None:
        probability = float(lgb_model.predict_proba(patient_df)[0][1])
    else:
        probability = 0.5

    news2_score, news2_breakdown, news2_level = compute_news2(
        hr=input_dict.get('Heart Rate', 75),
        rr=input_dict.get('Respiratory Rate', 16),
        spo2=input_dict.get('Oxygen Saturation', 98),
        sbp=input_dict.get('Systolic Blood Pressure', 120),
        temp=input_dict.get('Body Temperature', 37.0)
    )

    risk_score, risk_category, color = calculate_health_risk_index(probability, news2_score, input_dict)

    if iso_forest is not None:
        anomaly_pred = iso_forest.predict(patient_df)[0]
        is_anomaly = "Yes" if anomaly_pred == -1 else "No"
    else:
        is_anomaly = "No"

    if explainer is not None:
        shap_raw = explainer.shap_values(patient_df)
        values = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]
    else:
        values = np.zeros(len(feature_names))

    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': patient_df.values[0],
        'Contribution': values
    }).sort_values(by='Contribution', key=abs, ascending=False)

    top_factors = []
    for row in explanation_df.head(5).itertuples(index=False):
        feat_name = row.Feature
        feat_val = row.Value
        feat_contrib = row.Contribution
        direction, note = get_clinical_explanation_tag(feat_name, feat_val, feat_contrib)
        top_factors.append((feat_name, feat_val, feat_contrib, direction, note))

    result_label = "High Risk / Deteriorating" if risk_score > 50 else "Low Risk / Stable"

    ai_summary = generate_ai_clinical_summary(
        input_dict, risk_score, risk_category, news2_score, news2_level, is_anomaly, top_factors
    )

    return {
        "prediction": result_label,
        "probability": probability,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_color": color,
        "news2_score": news2_score,
        "news2_breakdown": news2_breakdown,
        "news2_level": news2_level,
        "is_anomaly": is_anomaly,
        "top_factors": top_factors,
        "explanation_df": explanation_df,
        "ai_summary": ai_summary
    }


# ============================================================
# DATABASE & BENCHMARK PATIENT SEEDING
# ============================================================

def init_db():
    """
    Initializes database tables and seeds benchmark historical trajectories if empty.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        cohort_tag TEXT DEFAULT 'General',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute("PRAGMA table_info(patients)")
    columns = [col[1] for col in cursor.fetchall()]
    if "cohort_tag" not in columns:
        cursor.execute("ALTER TABLE patients ADD COLUMN cohort_tag TEXT DEFAULT 'General'")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        heart_rate REAL,
        respiratory_rate REAL,
        body_temperature REAL,
        oxygen_saturation REAL,
        systolic_bp REAL,
        diastolic_bp REAL,
        age REAL,
        gender REAL,
        weight REAL,
        height REAL,
        hrv REAL,
        pulse_pressure REAL,
        bmi REAL,
        map_value REAL,
        prediction TEXT,
        risk_score REAL,
        risk_category TEXT,
        news2_score INTEGER DEFAULT 0,
        is_anomaly TEXT,
        top_factors TEXT,
        clinical_summary TEXT,
        predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
    )
    ''')

    cursor.execute("PRAGMA table_info(predictions)")
    pred_columns = [col[1] for col in cursor.fetchall()]
    if "news2_score" not in pred_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN news2_score INTEGER DEFAULT 0")
    if "clinical_summary" not in pred_columns:
        cursor.execute("ALTER TABLE predictions ADD COLUMN clinical_summary TEXT DEFAULT ''")

    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if count < 15:
        seed_benchmark_patients(cursor, conn)

    conn.close()


def seed_benchmark_patients(cursor, conn):
    """
    Seeds 5 distinct clinical patient trajectories showing progressive health timelines.
    """
    benchmarks = [
        # Patient 1: ICU Sepsis Trajectory
        {
            "name": "David Miller (ICU Sepsis Case)",
            "age": 64,
            "gender": "Male",
            "tag": "Sepsis Deterioration",
            "history": [
                {"hours_ago": 18, "hr": 78, "rr": 16, "temp": 37.2, "spo2": 97.0, "sbp": 122, "dbp": 78, "w": 78.0, "h": 1.76},
                {"hours_ago": 12, "hr": 96, "rr": 20, "temp": 38.4, "spo2": 95.0, "sbp": 110, "dbp": 70, "w": 78.0, "h": 1.76},
                {"hours_ago": 6,  "hr": 118, "rr": 24, "temp": 39.3, "spo2": 92.5, "sbp": 95,  "dbp": 60, "w": 78.0, "h": 1.76},
                {"hours_ago": 1,  "hr": 138, "rr": 29, "temp": 40.2, "spo2": 89.0, "sbp": 82,  "dbp": 48, "w": 78.0, "h": 1.76},
            ]
        },
        # Patient 2: Hypertensive Crisis & Recovery
        {
            "name": "Sarah Jenkins (Hypertensive Crisis)",
            "age": 58,
            "gender": "Female",
            "tag": "Cardiovascular",
            "history": [
                {"hours_ago": 24, "hr": 92, "rr": 18, "temp": 36.8, "spo2": 97.0, "sbp": 205, "dbp": 125, "w": 82.0, "h": 1.65},
                {"hours_ago": 16, "hr": 88, "rr": 17, "temp": 36.7, "spo2": 97.5, "sbp": 185, "dbp": 110, "w": 82.0, "h": 1.65},
                {"hours_ago": 8,  "hr": 82, "rr": 16, "temp": 36.8, "spo2": 98.0, "sbp": 160, "dbp": 95,  "w": 82.0, "h": 1.65},
                {"hours_ago": 2,  "hr": 76, "rr": 15, "temp": 36.7, "spo2": 98.5, "sbp": 134, "dbp": 84,  "w": 82.0, "h": 1.65},
            ]
        },
        # Patient 3: Stable Healthy Baseline
        {
            "name": "Elena Rostova (Wellness Baseline)",
            "age": 34,
            "gender": "Female",
            "tag": "Healthy Baseline",
            "history": [
                {"hours_ago": 72, "hr": 68, "rr": 14, "temp": 36.7, "spo2": 99.0, "sbp": 116, "dbp": 74, "w": 62.0, "h": 1.68},
                {"hours_ago": 48, "hr": 70, "rr": 15, "temp": 36.8, "spo2": 98.5, "sbp": 118, "dbp": 76, "w": 62.0, "h": 1.68},
                {"hours_ago": 24, "hr": 72, "rr": 14, "temp": 36.6, "spo2": 99.0, "sbp": 115, "dbp": 75, "w": 62.0, "h": 1.68},
                {"hours_ago": 4,  "hr": 69, "rr": 15, "temp": 36.8, "spo2": 98.8, "sbp": 117, "dbp": 76, "w": 62.0, "h": 1.68},
            ]
        },
        # Patient 4: COPD / Hypoxia Exacerbation
        {
            "name": "Robert Chen (COPD Flare-up)",
            "age": 71,
            "gender": "Male",
            "tag": "Respiratory Failure",
            "history": [
                {"hours_ago": 30, "hr": 84, "rr": 18, "temp": 36.9, "spo2": 94.0, "sbp": 130, "dbp": 82, "w": 70.0, "h": 1.72},
                {"hours_ago": 20, "hr": 98, "rr": 22, "temp": 37.4, "spo2": 91.0, "sbp": 136, "dbp": 84, "w": 70.0, "h": 1.72},
                {"hours_ago": 10, "hr": 112, "rr": 26, "temp": 37.8, "spo2": 86.5, "sbp": 142, "dbp": 88, "w": 70.0, "h": 1.72},
                {"hours_ago": 1,  "hr": 122, "rr": 32, "temp": 38.1, "spo2": 81.5, "sbp": 145, "dbp": 90, "w": 70.0, "h": 1.72},
            ]
        },
        # Patient 5: Post-Op Cardiac Instability
        {
            "name": "Maria Gonzalez (Post-Op Cardiac)",
            "age": 60,
            "gender": "Female",
            "tag": "Post-Op Monitoring",
            "history": [
                {"hours_ago": 36, "hr": 76, "rr": 16, "temp": 37.0, "spo2": 97.5, "sbp": 120, "dbp": 80, "w": 68.0, "h": 1.64},
                {"hours_ago": 24, "hr": 95, "rr": 18, "temp": 37.3, "spo2": 96.0, "sbp": 135, "dbp": 86, "w": 68.0, "h": 1.64},
                {"hours_ago": 12, "hr": 128, "rr": 22, "temp": 37.6, "spo2": 94.0, "sbp": 148, "dbp": 92, "w": 68.0, "h": 1.64},
                {"hours_ago": 3,  "hr": 142, "rr": 25, "temp": 37.9, "spo2": 92.0, "sbp": 155, "dbp": 96, "w": 68.0, "h": 1.64},
            ]
        }
    ]

    base_time = datetime.now()

    for p in benchmarks:
        cursor.execute(
            "INSERT INTO patients (name, age, gender, cohort_tag) VALUES (?, ?, ?, ?)",
            (p["name"], p["age"], p["gender"], p["tag"])
        )
        patient_id = cursor.lastrowid

        for h in p["history"]:
            ts = (base_time - timedelta(hours=h["hours_ago"])).strftime('%Y-%m-%d %H:%M:%S')
            pp = h["sbp"] - h["dbp"]
            bmi = round(h["w"] / (h["h"] ** 2), 2)
            map_val = round(h["dbp"] + (pp / 3.0), 2)
            hrv = 0.05 if h["hr"] > 110 else 0.11

            input_dict = {
                'Heart Rate': h["hr"],
                'Respiratory Rate': h["rr"],
                'Body Temperature': h["temp"],
                'Oxygen Saturation': h["spo2"],
                'Systolic Blood Pressure': h["sbp"],
                'Diastolic Blood Pressure': h["dbp"],
                'Age': p["age"],
                'Gender': 1 if p["gender"] == "Male" else 0,
                'Weight (kg)': h["w"],
                'Height (m)': h["h"],
                'Derived_HRV': hrv,
                'Derived_Pulse_Pressure': pp,
                'Derived_BMI': bmi,
                'Derived_MAP': map_val
            }

            res = predict_patient(input_dict)
            top_factors_str = "; ".join([f"{f[0]}: {f[2]:.2f}" for f in res['top_factors']])

            cursor.execute('''
                INSERT INTO predictions (
                    patient_id, heart_rate, respiratory_rate, body_temperature,
                    oxygen_saturation, systolic_bp, diastolic_bp, age, gender,
                    weight, height, hrv, pulse_pressure, bmi, map_value,
                    prediction, risk_score, risk_category, news2_score,
                    is_anomaly, top_factors, clinical_summary, predicted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id, h["hr"], h["rr"], h["temp"], h["spo2"],
                h["sbp"], h["dbp"], p["age"], 1 if p["gender"] == "Male" else 0,
                h["w"], h["h"], hrv, pp, bmi, map_val,
                res['prediction'], res['risk_score'], res['risk_category'],
                res['news2_score'], res['is_anomaly'], top_factors_str,
                res['ai_summary']['findings'], ts
            ))

    conn.commit()


def save_prediction(input_dict, result, patient_name="Anonymous"):
    """
    Saves a live user prediction to the database.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO patients (name, age, gender, cohort_tag) VALUES (?, ?, ?, ?)",
        (patient_name, input_dict['Age'], 'Male' if input_dict['Gender'] == 1 else 'Female', 'Live Session')
    )
    patient_id = cursor.lastrowid

    top_factors_str = "; ".join([f"{f[0]}: {f[2]:.2f}" for f in result['top_factors']])

    cursor.execute('''
        INSERT INTO predictions (
            patient_id, heart_rate, respiratory_rate, body_temperature,
            oxygen_saturation, systolic_bp, diastolic_bp, age, gender,
            weight, height, hrv, pulse_pressure, bmi, map_value,
            prediction, risk_score, risk_category, news2_score,
            is_anomaly, top_factors, clinical_summary, predicted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id, input_dict['Heart Rate'], input_dict['Respiratory Rate'],
        input_dict['Body Temperature'], input_dict['Oxygen Saturation'],
        input_dict['Systolic Blood Pressure'], input_dict['Diastolic Blood Pressure'],
        input_dict['Age'], input_dict['Gender'], input_dict['Weight (kg)'],
        input_dict['Height (m)'], input_dict['Derived_HRV'],
        input_dict['Derived_Pulse_Pressure'], input_dict['Derived_BMI'],
        input_dict['Derived_MAP'], result['prediction'], result['risk_score'],
        result['risk_category'], result['news2_score'], result['is_anomaly'],
        top_factors_str, result['ai_summary']['findings'],
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))

    conn.commit()
    conn.close()
    return patient_id


def get_prediction_history():
    """
    Retrieves full prediction history joined with patient info.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        p.prediction_id,
        pt.name as Patient_Name,
        pt.cohort_tag as Case_Type,
        p.predicted_at as Timestamp,
        p.risk_score as Risk_Score,
        p.risk_category as Category,
        p.news2_score as NEWS2,
        p.heart_rate as HR_bpm,
        p.oxygen_saturation as SpO2_pct,
        p.systolic_bp as SBP,
        p.diastolic_bp as DBP,
        p.body_temperature as Temp_C,
        p.respiratory_rate as RR,
        p.is_anomaly as Anomaly,
        p.top_factors as SHAP_Top_Factors
    FROM predictions p
    LEFT JOIN patients pt ON p.patient_id = pt.patient_id
    ORDER BY p.predicted_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_patient_timeline(patient_name_or_id):
    """
    Retrieves timeline progression for a specific patient.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        p.predicted_at,
        p.risk_score,
        p.news2_score,
        p.heart_rate,
        p.oxygen_saturation,
        p.systolic_bp,
        p.diastolic_bp,
        p.body_temperature,
        p.respiratory_rate,
        p.risk_category
    FROM predictions p
    LEFT JOIN patients pt ON p.patient_id = pt.patient_id
    WHERE pt.name = ? OR pt.patient_id = ?
    ORDER BY p.predicted_at ASC
    """
    df = pd.read_sql_query(query, conn, params=(patient_name_or_id, patient_name_or_id))
    conn.close()
    return df


# Run init on import
init_db()