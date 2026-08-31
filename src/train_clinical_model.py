"""
Clinical Deterioration Model Training Pipeline
Vital Health Alert XAI

Incorporates physiological clinical standards (NEWS2, MEWS, AHA guidelines)
to train LightGBM, Random Forest, Isolation Forest, and SHAP explainers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def calculate_news2_score(hr, rr, spo2, sbp, temp):
    """
    Computes standard NHS NEWS2 (National Early Warning Score) subscores.
    """
    score = 0

    # Respiration Rate
    if rr <= 8:
        score += 3
    elif 9 <= rr <= 11:
        score += 1
    elif 12 <= rr <= 20:
        score += 0
    elif 21 <= rr <= 24:
        score += 2
    else:  # >= 25
        score += 3

    # Oxygen Saturation (SpO2)
    if spo2 >= 96:
        score += 0
    elif 94 <= spo2 <= 95:
        score += 1
    elif 92 <= spo2 <= 93:
        score += 2
    else:  # <= 91
        score += 3

    # Systolic Blood Pressure
    if sbp <= 90:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1
    elif 111 <= sbp <= 159:
        score += 0
    elif 160 <= sbp <= 179:
        score += 1
    elif 180 <= sbp <= 219:
        score += 2
    else:  # >= 220
        score += 3

    # Heart Rate (Pulse)
    if hr <= 40:
        score += 3
    elif 41 <= hr <= 50:
        score += 1
    elif 51 <= hr <= 90:
        score += 0
    elif 91 <= hr <= 110:
        score += 1
    elif 111 <= hr <= 130:
        score += 2
    else:  # >= 131
        score += 3

    # Body Temperature
    if temp <= 35.0:
        score += 3
    elif 35.1 <= temp <= 36.0:
        score += 1
    elif 36.1 <= temp <= 38.0:
        score += 0
    elif 38.1 <= temp <= 39.0:
        score += 1
    else:  # >= 39.1
        score += 2

    return score


def generate_clinical_dataset(n_samples=60000, random_seed=42):
    """
    Generates a realistic clinical dataset covering healthy individuals,
    mild ailments, chronic conditions, and acute life-threatening deteriorations.
    """
    np.random.seed(random_seed)

    records = []

    # Distribution of patient clinical states:
    # 40% Healthy / Normal baseline
    # 25% Mildly unwell (early infection, mild hypertension)
    # 20% Moderately deteriorated (acute illness, high fever, COPD flare)
    # 15% Critically ill (sepsis, hypertensive emergency, severe hypoxia, shock)

    n_healthy = int(n_samples * 0.40)
    n_mild = int(n_samples * 0.25)
    n_mod = int(n_samples * 0.20)
    n_crit = n_samples - (n_healthy + n_mild + n_mod)

    # 1. Healthy Cohort
    for _ in range(n_healthy):
        age = int(np.random.uniform(18, 75))
        gender = np.random.choice([0, 1])  # 0=Female, 1=Male
        height = np.random.normal(1.70, 0.10)
        height = np.clip(height, 1.45, 2.05)
        bmi = np.random.normal(23.5, 3.0)
        bmi = np.clip(bmi, 18.5, 32.0)
        weight = bmi * (height ** 2)

        hr = int(np.random.normal(72, 8))
        hr = np.clip(hr, 55, 88)
        rr = int(np.random.normal(15, 2))
        rr = np.clip(rr, 12, 18)
        spo2 = round(float(np.random.normal(98.2, 1.0)), 1)
        spo2 = np.clip(spo2, 96.0, 100.0)
        sbp = int(np.random.normal(118, 8))
        sbp = np.clip(sbp, 105, 130)
        dbp = int(sbp * 0.65 + np.random.normal(0, 4))
        dbp = np.clip(dbp, 65, 84)
        temp = round(float(np.random.normal(36.8, 0.3)), 2)
        temp = np.clip(temp, 36.2, 37.4)
        hrv = round(float(np.random.normal(0.12, 0.02)), 4)
        hrv = np.clip(hrv, 0.08, 0.18)

        records.append({
            'Heart Rate': hr,
            'Respiratory Rate': rr,
            'Body Temperature': temp,
            'Oxygen Saturation': spo2,
            'Systolic Blood Pressure': sbp,
            'Diastolic Blood Pressure': dbp,
            'Age': age,
            'Gender': gender,
            'Weight (kg)': round(float(weight), 2),
            'Height (m)': round(float(height), 2),
            'Derived_HRV': hrv,
            'Cohort': 'Healthy'
        })

    # 2. Mildly Unwell Cohort
    for _ in range(n_mild):
        age = int(np.random.uniform(20, 80))
        gender = np.random.choice([0, 1])
        height = np.clip(np.random.normal(1.70, 0.10), 1.45, 2.05)
        bmi = np.clip(np.random.normal(26.0, 4.0), 18.0, 36.0)
        weight = bmi * (height ** 2)

        condition_type = np.random.choice(['mild_fever', 'mild_htn', 'mild_tachy', 'mild_resp'])
        hr = int(np.random.normal(84, 10))
        rr = int(np.random.normal(17, 2))
        spo2 = round(float(np.random.normal(95.5, 1.2)), 1)
        sbp = int(np.random.normal(136, 10))
        temp = round(float(np.random.normal(37.3, 0.5)), 2)

        if condition_type == 'mild_fever':
            temp = round(float(np.random.uniform(37.8, 38.6)), 2)
            hr = int(np.random.uniform(85, 105))
        elif condition_type == 'mild_htn':
            sbp = int(np.random.uniform(140, 158))
        elif condition_type == 'mild_tachy':
            hr = int(np.random.uniform(96, 115))
        elif condition_type == 'mild_resp':
            rr = int(np.random.uniform(20, 23))
            spo2 = round(float(np.random.uniform(93.5, 95.0)), 1)

        hr = np.clip(hr, 50, 120)
        rr = np.clip(rr, 10, 24)
        spo2 = np.clip(spo2, 93.0, 99.0)
        sbp = np.clip(sbp, 95, 165)
        dbp = int(sbp * 0.65 + np.random.normal(0, 5))
        dbp = np.clip(dbp, 60, 95)
        temp = np.clip(temp, 35.8, 38.8)
        hrv = round(float(np.random.normal(0.09, 0.02)), 4)
        hrv = np.clip(hrv, 0.05, 0.14)

        records.append({
            'Heart Rate': hr,
            'Respiratory Rate': rr,
            'Body Temperature': temp,
            'Oxygen Saturation': spo2,
            'Systolic Blood Pressure': sbp,
            'Diastolic Blood Pressure': dbp,
            'Age': age,
            'Gender': gender,
            'Weight (kg)': round(float(weight), 2),
            'Height (m)': round(float(height), 2),
            'Derived_HRV': hrv,
            'Cohort': 'Mild'
        })

    # 3. Moderately Deteriorated Cohort
    for _ in range(n_mod):
        age = int(np.random.uniform(25, 88))
        gender = np.choice([0, 1]) if hasattr(np, 'choice') else np.random.choice([0, 1])
        height = np.clip(np.random.normal(1.70, 0.10), 1.45, 2.05)
        bmi = np.clip(np.random.normal(27.5, 5.0), 16.0, 42.0)
        weight = bmi * (height ** 2)

        condition_type = np.random.choice(['high_fever_infection', 'stage2_htn', 'hypoxia_moderate', 'tachy_arrhythmia'])
        hr = int(np.random.normal(105, 12))
        rr = int(np.random.normal(22, 3))
        spo2 = round(float(np.random.normal(92.5, 1.5)), 1)
        sbp = int(np.random.normal(155, 15))
        temp = round(float(np.random.normal(38.4, 0.6)), 2)

        if condition_type == 'high_fever_infection':
            temp = round(float(np.random.uniform(38.8, 39.8)), 2)
            hr = int(np.random.uniform(105, 125))
            rr = int(np.random.uniform(22, 26))
        elif condition_type == 'stage2_htn':
            sbp = int(np.random.uniform(165, 185))
            dbp = int(np.random.uniform(100, 115))
        elif condition_type == 'hypoxia_moderate':
            spo2 = round(float(np.random.uniform(90.0, 93.0)), 1)
            rr = int(np.random.uniform(22, 27))
        elif condition_type == 'tachy_arrhythmia':
            hr = int(np.random.uniform(118, 140))

        hr = np.clip(hr, 42, 145)
        rr = np.clip(rr, 9, 28)
        spo2 = np.clip(spo2, 88.0, 96.0)
        sbp = np.clip(sbp, 88, 190)
        dbp = int(sbp * 0.64 + np.random.normal(0, 6))
        dbp = np.clip(dbp, 55, 115)
        temp = np.clip(temp, 35.2, 40.0)
        hrv = round(float(np.random.normal(0.065, 0.015)), 4)
        hrv = np.clip(hrv, 0.03, 0.10)

        records.append({
            'Heart Rate': hr,
            'Respiratory Rate': rr,
            'Body Temperature': temp,
            'Oxygen Saturation': spo2,
            'Systolic Blood Pressure': sbp,
            'Diastolic Blood Pressure': dbp,
            'Age': age,
            'Gender': gender,
            'Weight (kg)': round(float(weight), 2),
            'Height (m)': round(float(height), 2),
            'Derived_HRV': hrv,
            'Cohort': 'Moderate'
        })

    # 4. Critically Deteriorated Cohort (Emergency / ICU / Sepsis / Shock)
    for _ in range(n_crit):
        age = int(np.random.uniform(30, 92))
        gender = np.random.choice([0, 1])
        height = np.clip(np.random.normal(1.70, 0.10), 1.45, 2.05)
        bmi = np.clip(np.random.normal(28.0, 6.0), 15.0, 48.0)
        weight = bmi * (height ** 2)

        crit_type = np.random.choice(['sepsis_shock', 'severe_hypoxia_ards', 'hypertensive_crisis', 'severe_brady_shock', 'hyperpyrexia'])
        if crit_type == 'sepsis_shock':
            temp = round(float(np.random.uniform(39.4, 40.8)), 2) if np.random.rand() > 0.15 else round(float(np.random.uniform(34.2, 35.2)), 2)
            hr = int(np.random.uniform(125, 160))
            rr = int(np.random.uniform(26, 36))
            sbp = int(np.random.uniform(70, 88))  # Septic hypotension
            dbp = int(np.random.uniform(40, 55))
            spo2 = round(float(np.random.uniform(86.0, 92.0)), 1)
        elif crit_type == 'severe_hypoxia_ards':
            spo2 = round(float(np.random.uniform(75.0, 87.0)), 1)
            rr = int(np.random.uniform(28, 40))
            hr = int(np.random.uniform(115, 150))
            sbp = int(np.random.uniform(110, 160))
            dbp = int(np.random.uniform(70, 100))
            temp = round(float(np.random.normal(37.5, 1.0)), 2)
        elif crit_type == 'hypertensive_crisis':
            sbp = int(np.random.uniform(190, 240))
            dbp = int(np.random.uniform(120, 145))
            hr = int(np.random.uniform(95, 130))
            rr = int(np.random.uniform(20, 28))
            spo2 = round(float(np.random.uniform(91.0, 97.0)), 1)
            temp = round(float(np.random.normal(37.0, 0.6)), 2)
        elif crit_type == 'severe_brady_shock':
            hr = int(np.random.uniform(30, 42))  # Severe bradycardia
            sbp = int(np.random.uniform(70, 88))
            dbp = int(np.random.uniform(40, 55))
            rr = int(np.random.uniform(10, 18))
            spo2 = round(float(np.random.uniform(88.0, 94.0)), 1)
            temp = round(float(np.random.uniform(35.0, 36.2)), 2)
        else:  # Hyperpyrexia
            temp = round(float(np.random.uniform(40.2, 41.5)), 2)
            hr = int(np.random.uniform(130, 165))
            rr = int(np.random.uniform(28, 38))
            sbp = int(np.random.uniform(95, 140))
            dbp = int(np.random.uniform(55, 85))
            spo2 = round(float(np.random.uniform(88.0, 93.0)), 1)

        hr = np.clip(hr, 30, 180)
        rr = np.clip(rr, 6, 42)
        spo2 = np.clip(spo2, 70.0, 96.0)
        sbp = np.clip(sbp, 65, 250)
        dbp = np.clip(dbp, 35, 150)
        temp = np.clip(temp, 34.0, 41.8)
        hrv = round(float(np.random.normal(0.04, 0.015)), 4)
        hrv = np.clip(hrv, 0.015, 0.07)

        records.append({
            'Heart Rate': hr,
            'Respiratory Rate': rr,
            'Body Temperature': temp,
            'Oxygen Saturation': spo2,
            'Systolic Blood Pressure': sbp,
            'Diastolic Blood Pressure': dbp,
            'Age': age,
            'Gender': gender,
            'Weight (kg)': round(float(weight), 2),
            'Height (m)': round(float(height), 2),
            'Derived_HRV': hrv,
            'Cohort': 'Critical'
        })

    df = pd.DataFrame(records)

    # Compute Derived Features exactly matching existing schema
    df['Derived_Pulse_Pressure'] = df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']
    df['Derived_BMI'] = (df['Weight (kg)'] / (df['Height (m)'] ** 2)).round(2)
    df['Derived_MAP'] = (df['Diastolic Blood Pressure'] + (df['Derived_Pulse_Pressure'] / 3.0)).round(2)

    # Compute NEWS2 and Clinical Ground Truth
    news2_scores = []
    risk_labels = []
    risk_indices = []

    for idx, row in df.iterrows():
        n2 = calculate_news2_score(
            row['Heart Rate'],
            row['Respiratory Rate'],
            row['Oxygen Saturation'],
            row['Systolic Blood Pressure'],
            row['Body Temperature']
        )
        news2_scores.append(n2)

        # Composite Continuous Risk Index (0-100)
        base_index = n2 * 9.5

        # Extreme vital penalties
        if row['Oxygen Saturation'] < 90:
            base_index += (90 - row['Oxygen Saturation']) * 3.0
        if row['Systolic Blood Pressure'] > 180:
            base_index += (row['Systolic Blood Pressure'] - 180) * 0.45
        elif row['Systolic Blood Pressure'] < 90:
            base_index += (90 - row['Systolic Blood Pressure']) * 0.8
        if row['Body Temperature'] > 39.0:
            base_index += (row['Body Temperature'] - 39.0) * 12.0
        elif row['Body Temperature'] < 35.5:
            base_index += (35.5 - row['Body Temperature']) * 14.0
        if row['Heart Rate'] > 120:
            base_index += (row['Heart Rate'] - 120) * 0.4
        elif row['Heart Rate'] < 45:
            base_index += (45 - row['Heart Rate']) * 0.8
        if row['Derived_BMI'] > 35 or row['Derived_BMI'] < 16:
            base_index += 4.0
        if row['Age'] > 70:
            base_index += (row['Age'] - 70) * 0.2

        risk_idx = np.clip(base_index + np.random.normal(0, 2.0), 0.0, 100.0)
        risk_indices.append(round(float(risk_idx), 1))

        # Binary label for classifier (High Risk vs Low Risk):
        if n2 >= 5 or risk_idx >= 45.0 or row['Cohort'] in ['Moderate', 'Critical']:
            risk_labels.append(1)  # High Risk
        else:
            risk_labels.append(0)  # Low Risk / Healthy

    df['NEWS2_Score'] = news2_scores
    df['Continuous_Risk_Index'] = risk_indices
    df['Target_Risk'] = risk_labels

    return df


def train_and_save_models():
    print("==================================================")
    print("Generating Clinically Grounded Deterioration Data...")
    print("==================================================")
    df = generate_clinical_dataset(n_samples=60000, random_seed=42)

    csv_path = os.path.join(DATA_DIR, "clinical_vitals_dataset_validated.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved validated clinical dataset to {csv_path} ({len(df)} records)")

    feature_cols = [
        'Heart Rate', 'Respiratory Rate', 'Body Temperature', 'Oxygen Saturation',
        'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Age', 'Gender',
        'Weight (kg)', 'Height (m)', 'Derived_HRV', 'Derived_Pulse_Pressure',
        'Derived_BMI', 'Derived_MAP'
    ]

    X = df[feature_cols]
    y = df['Target_Risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining LightGBM on {len(X_train)} samples...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.06,
        max_depth=6,
        num_leaves=31,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)

    lgb_preds = lgb_model.predict(X_test)
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

    print("\n--- LightGBM Test Results ---")
    print(f"Accuracy:  {accuracy_score(y_test, lgb_preds):.4f}")
    print(f"Precision: {precision_score(y_test, lgb_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, lgb_preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, lgb_preds):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, lgb_probs):.4f}")

    print("\nTraining Random Forest Ensemble...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=6,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

    print("\nTraining Isolation Forest (Anomaly Detector)...")
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.06,
        max_samples=0.8,
        random_state=42,
        n_jobs=-1
    )
    healthy_X = X[df['Cohort'] == 'Healthy']
    iso_forest.fit(healthy_X)

    print("\nInitializing SHAP TreeExplainer on LightGBM...")
    explainer = shap.TreeExplainer(lgb_model)

    print("\nSaving models to 'models/' directory...")
    joblib.dump(lgb_model, os.path.join(MODELS_DIR, "lgb_model.pkl"))
    joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_model.pkl"))
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_names.pkl"))
    print("All models successfully saved!")

    print("\n==================================================")
    print("CLINICAL TEST VERIFICATIONS")
    print("==================================================")

    test_cases = [
        {
            "name": "Normal Healthy Adult",
            "vitals": {
                'Heart Rate': 72, 'Respiratory Rate': 15, 'Body Temperature': 36.8,
                'Oxygen Saturation': 98.5, 'Systolic Blood Pressure': 118, 'Diastolic Blood Pressure': 76,
                'Age': 32, 'Gender': 1, 'Weight (kg)': 70.0, 'Height (m)': 1.75,
                'Derived_HRV': 0.12, 'Derived_Pulse_Pressure': 42, 'Derived_BMI': 22.86, 'Derived_MAP': 90.0
            },
            "expected": "Low Risk / Healthy (Score <= 25)"
        },
        {
            "name": "Hypertensive Emergency (BP 205/125)",
            "vitals": {
                'Heart Rate': 92, 'Respiratory Rate': 18, 'Body Temperature': 36.9,
                'Oxygen Saturation': 97.0, 'Systolic Blood Pressure': 205, 'Diastolic Blood Pressure': 125,
                'Age': 58, 'Gender': 1, 'Weight (kg)': 85.0, 'Height (m)': 1.72,
                'Derived_HRV': 0.07, 'Derived_Pulse_Pressure': 80, 'Derived_BMI': 28.73, 'Derived_MAP': 151.67
            },
            "expected": "High / Critical Risk (Score >= 70)"
        },
        {
            "name": "Severe Hypoxia / COPD (SpO2 82%)",
            "vitals": {
                'Heart Rate': 115, 'Respiratory Rate': 30, 'Body Temperature': 37.2,
                'Oxygen Saturation': 82.0, 'Systolic Blood Pressure': 130, 'Diastolic Blood Pressure': 82,
                'Age': 68, 'Gender': 0, 'Weight (kg)': 62.0, 'Height (m)': 1.62,
                'Derived_HRV': 0.05, 'Derived_Pulse_Pressure': 48, 'Derived_BMI': 23.62, 'Derived_MAP': 98.0
            },
            "expected": "Critical Risk (Score >= 85)"
        },
        {
            "name": "Sepsis Alert (Temp 40.2C, HR 135, RR 28, BP 82/50)",
            "vitals": {
                'Heart Rate': 135, 'Respiratory Rate': 28, 'Body Temperature': 40.2,
                'Oxygen Saturation': 91.0, 'Systolic Blood Pressure': 82, 'Diastolic Blood Pressure': 50,
                'Age': 52, 'Gender': 1, 'Weight (kg)': 74.0, 'Height (m)': 1.78,
                'Derived_HRV': 0.03, 'Derived_Pulse_Pressure': 32, 'Derived_BMI': 23.36, 'Derived_MAP': 60.67
            },
            "expected": "Critical Risk (Score >= 90)"
        }
    ]

    for tc in test_cases:
        df_tc = pd.DataFrame([tc["vitals"]])[feature_cols]
        prob = lgb_model.predict_proba(df_tc)[0][1]
        pred = lgb_model.predict(df_tc)[0]
        shap_raw = explainer.shap_values(df_tc)
        shap_vals = shap_raw[1][0] if isinstance(shap_raw, list) else shap_raw[0]

        top_feat_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        top_factors = [(feature_cols[i], df_tc.values[0][i], round(float(shap_vals[i]), 3)) for i in top_feat_idx]

        print(f"\nTest Case: {tc['name']}")
        print(f"  -> Predicted Probability: {prob * 100:.1f}% | Prediction: {'High Risk' if pred == 1 else 'Low Risk'}")
        print(f"  -> Expected: {tc['expected']}")
        print(f"  -> Top SHAP Drivers: {top_factors}")


if __name__ == "__main__":
    train_and_save_models()
