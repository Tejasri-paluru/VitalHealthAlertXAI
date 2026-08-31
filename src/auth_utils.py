"""
auth_utils.py — User Authentication, Password Hashing, OTP Reset, & Profile Management
Vital Health Alert XAI

Integrates:
- SQLite user management (users, password_resets, vitals tables)
- SHA-256 password hashing
- OTP & reset token generation and verification
- SMTP email notifications
- Patient history & vitals persistence
"""

import os
import sqlite3
import hashlib
import secrets as pysecrets
import random
import string
import smtplib
import ssl
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "vital_health.db")

# Ensure database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_db():
    """
    Returns an SQLite connection with WAL mode and thread safety.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            full_name TEXT,
            phone TEXT DEFAULT '',
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'Patient',
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS password_resets (
            email TEXT PRIMARY KEY,
            otp TEXT,
            otp_expiry TEXT,
            reset_token TEXT,
            token_expiry TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vitals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            recorded_at TEXT,
            heart_rate REAL,
            bp_systolic REAL,
            bp_diastolic REAL,
            spo2 REAL,
            temperature REAL,
            resp_rate REAL,
            risk_score REAL,
            risk_category TEXT
        )
    """)
    conn.commit()
    return conn


def hash_password(password: str) -> str:
    """
    SHA-256 cryptographic password hashing.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def user_exists(email: str) -> bool:
    """
    Checks if a user already exists in the database.
    """
    conn = get_db()
    row = conn.execute("SELECT email FROM users WHERE LOWER(email) = ?", ((email or "").strip().lower(),)).fetchone()
    conn.close()
    return row is not None


def create_user(email: str, full_name: str, password: str, phone: str = "") -> bool:
    """
    Creates a new user account with hashed password.
    """
    try:
        conn = get_db()
        conn.execute(
            """INSERT OR REPLACE INTO users (email, full_name, phone, password_hash, role, created_at)
               VALUES (?, ?, ?, ?, 'Patient', ?)""",
            ((email or "").strip().lower(), full_name.strip(), (phone or "").strip(), hash_password(password), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def verify_login(email: str, password: str) -> bool:
    """
    Verifies user credentials against stored SHA-256 hash.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT password_hash FROM users WHERE LOWER(email) = ?", ((email or "").strip().lower(),)
    ).fetchone()
    conn.close()
    if not row:
        return False
    return row[0] == hash_password(password)


def get_user_profile(email: str):
    """
    Retrieves user profile details.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT email, full_name, phone, role, created_at FROM users WHERE LOWER(email) = ?",
        ((email or "").strip().lower(),)
    ).fetchone()
    conn.close()
    if row:
        return {
            "email": row[0],
            "full_name": row[1],
            "phone": row[2],
            "role": row[3],
            "created_at": row[4]
        }
    return None


def update_user_password(email: str, new_password: str) -> bool:
    """
    Updates user password.
    """
    try:
        conn = get_db()
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE LOWER(email) = ?",
            (hash_password(new_password), (email or "").strip().lower())
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def generate_otp() -> str:
    """
    Generates a 6-digit numeric OTP.
    """
    return "".join(random.choices(string.digits, k=6))


def create_password_reset(email: str):
    """
    Generates a 10-minute OTP and 30-minute reset token for password recovery.
    """
    email_clean = (email or "").strip().lower()
    otp = generate_otp()
    token = pysecrets.token_urlsafe(24)
    now = datetime.now()
    otp_expiry = (now + timedelta(minutes=10)).isoformat()
    token_expiry = (now + timedelta(minutes=30)).isoformat()

    conn = get_db()
    conn.execute(
        """INSERT OR REPLACE INTO password_resets (email, otp, otp_expiry, reset_token, token_expiry)
           VALUES (?, ?, ?, ?, ?)""",
        (email_clean, otp, otp_expiry, token, token_expiry)
    )
    conn.commit()
    conn.close()
    return otp, token


def verify_otp(email: str, entered_otp: str) -> bool:
    """
    Verifies the entered OTP and checks expiration.
    """
    email_clean = (email or "").strip().lower()
    conn = get_db()
    row = conn.execute(
        "SELECT otp, otp_expiry FROM password_resets WHERE LOWER(email) = ?", (email_clean,)
    ).fetchone()
    conn.close()
    if not row:
        return False
    stored_otp, expiry = row
    if datetime.now() > datetime.fromisoformat(expiry):
        return False
    return (entered_otp or "").strip() == stored_otp


def verify_reset_token(token: str):
    """
    Verifies reset token from URL links.
    """
    conn = get_db()
    row = conn.execute(
        "SELECT email, token_expiry FROM password_resets WHERE reset_token = ?", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    email, expiry = row
    if datetime.now() > datetime.fromisoformat(expiry):
        return None
    return email


def clear_password_reset(email: str):
    """
    Clears reset records after successful reset.
    """
    conn = get_db()
    conn.execute("DELETE FROM password_resets WHERE LOWER(email) = ?", ((email or "").strip().lower(),))
    conn.commit()
    conn.close()


def save_vitals(email: str, hr: float, bp_sys: float, bp_dia: float, spo2: float, temp: float, resp_rate: float, risk_score: float, risk_category: str):
    """
    Records a user's vital check into the vitals database table.
    """
    conn = get_db()
    conn.execute(
        """INSERT INTO vitals
           (email, recorded_at, heart_rate, bp_systolic, bp_diastolic, spo2, temperature, resp_rate, risk_score, risk_category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ((email or "").strip().lower(), datetime.now().isoformat(), hr, bp_sys, bp_dia, spo2, temp, resp_rate, risk_score, risk_category)
    )
    conn.commit()
    conn.close()


save_user_vitals = save_vitals  # Alias for compatibility


def get_patient_history(email: str):
    """
    Retrieves vital sign records for a specific logged-in user.
    """
    conn = get_db()
    rows = conn.execute(
        """SELECT recorded_at, heart_rate, bp_systolic, bp_diastolic, spo2, temperature, resp_rate, risk_score, risk_category
           FROM vitals WHERE LOWER(email) = ? ORDER BY recorded_at DESC""",
        ((email or "").strip().lower(),)
    ).fetchall()
    conn.close()
    return rows


get_user_vitals_history = get_patient_history  # Alias for compatibility


def get_latest_vitals(email: str):
    history = get_patient_history(email)
    return history[0] if history else None


def compute_risk_score(hr, bp_sys, bp_dia, spo2, temp, resp_rate):
    """
    Computes rule-based explainability breakdown.
    """
    def deviation(value, low, high, hard_low, hard_high):
        if low <= value <= high:
            return 0.0
        if value < low:
            span = max(low - hard_low, 1)
            return min(100.0, (low - value) / span * 100)
        span = max(hard_high - high, 1)
        return min(100.0, (value - high) / span * 100)

    raw = {
        "Heart Rate": deviation(hr, 60, 100, 30, 180),
        "Blood Pressure": deviation(bp_sys, 90, 120, 60, 200),
        "Oxygen Saturation": deviation(spo2, 95, 100, 70, 100),
        "Temperature": deviation(temp, 97, 99, 93, 106),
        "Respiratory Rate": deviation(resp_rate, 12, 20, 6, 40),
    }
    weights = {
        "Heart Rate": 0.20,
        "Blood Pressure": 0.20,
        "Oxygen Saturation": 0.30,
        "Temperature": 0.15,
        "Respiratory Rate": 0.15,
    }

    total = round(min(100.0, sum(raw[k] * weights[k] for k in raw)), 1)

    if total <= 30:
        category = "Healthy"
    elif total <= 60:
        category = "Moderate Risk"
    elif total <= 80:
        category = "High Risk"
    else:
        category = "Critical"

    raw_sum = sum(raw.values()) or 1.0
    contributions = {k: round((v / raw_sum) * 100, 1) for k, v in raw.items()}

    return total, category, contributions


# ============================================================
# SMTP EMAIL NOTIFICATIONS
# ============================================================

def get_app_base_url():
    try:
        return st.secrets["smtp"].get("app_base_url", "http://localhost:8501")
    except Exception:
        return "http://localhost:8501"


def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """
    Sends an SMTP email. Gracefully returns False if secrets.toml is not configured.
    """
    try:
        smtp_email = st.secrets["smtp"]["email"]
        smtp_password = st.secrets["smtp"]["app_password"]
        smtp_server = st.secrets["smtp"].get("server", "smtp.gmail.com")
        smtp_port = int(st.secrets["smtp"].get("port", 587))
    except Exception:
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"Vital Health Alert XAI <{smtp_email}>"
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception:
        return False


def registration_email_html(full_name: str) -> str:
    base_url = get_app_base_url()
    return f"""
    <div style="font-family:Arial, sans-serif; max-width:500px; margin:0 auto; padding:30px; background:#F5F8FB; border-radius:16px; border:1px solid #02C39A;">
        <h2 style="color:#0B3D91; margin-top:0;">Welcome to Vital Health Alert XAI 🎉</h2>
        <p>Hi <b>{full_name or 'there'}</b>,</p>
        <p>Your registration was <b>successful</b>! Your account is ready — you can now log in, track your vital signs, and receive explainable AI health predictions.</p>
        <a href="{base_url}" style="display:inline-block; margin-top:16px; padding:12px 24px; background:#0B3D91; color:#ffffff; text-decoration:none; font-weight:700; border-radius:8px;">Go to Login</a>
        <p style="margin-top:24px; color:#888; font-size:12px;">Academic Research Prototype • Final Year Project</p>
    </div>
    """


def otp_email_html(otp: str, reset_link: str) -> str:
    return f"""
    <div style="font-family:Arial, sans-serif; max-width:500px; margin:0 auto; padding:30px; background:#F5F8FB; border-radius:16px; border:1px solid #02C39A;">
        <h2 style="color:#0B3D91; margin-top:0;">🔑 Password Reset Verification</h2>
        <p>Use the 6-digit OTP below to reset your Vital Health Alert XAI account password:</p>
        <div style="font-size:32px; font-weight:800; letter-spacing:6px; color:#0B3D91; margin:16px 0; background:rgba(0,0,0,0.05); padding:12px; border-radius:8px; text-align:center;">{otp}</div>
        <p style="font-size:13px; color:#666;">This code expires in 10 minutes.</p>
        <p>Or click below to reset your password directly:</p>
        <a href="{reset_link}" style="display:inline-block; margin-top:10px; padding:12px 24px; background:#00B4D8; color:#ffffff; text-decoration:none; font-weight:700; border-radius:8px;">Reset Password</a>
        <p style="margin-top:24px; color:#888; font-size:12px;">If you did not request this, you can safely ignore this email.</p>
    </div>
    """


def password_changed_email_html() -> str:
    base_url = get_app_base_url()
    return f"""
    <div style="font-family:Arial, sans-serif; max-width:500px; margin:0 auto; padding:30px; background:#F5F8FB; border-radius:16px; border:1px solid #02C39A;">
        <h2 style="color:#0B3D91; margin-top:0;">✅ Your Password Has Been Changed</h2>
        <p>This confirms your Vital Health Alert XAI account password was just updated.</p>
        <a href="{base_url}" style="display:inline-block; margin-top:16px; padding:12px 24px; background:#0B3D91; color:#ffffff; text-decoration:none; font-weight:700; border-radius:8px;">Login Now</a>
        <p style="margin-top:24px; color:#888; font-size:12px;">If you didn't make this change, please contact support immediately.</p>
    </div>
    """


# Initialize database tables on module load
get_db()
