import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load saved model, scaler, and feature names ─────────────────────────────
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mental Health Risk Screener", page_icon="🧠", layout="centered")

st.title("Mental Health Risk Screener")
st.write(
    "This tool estimates depression risk based on lifestyle and demographic factors "
    "using a machine learning model trained on survey data. "
    "**This is not a clinical diagnosis.** If you are struggling, please speak to a "
    "qualified mental health professional."
)
st.divider()

# ── Encoding maps (must match the LabelEncoder mapping used in training) ────
# NOTE: these mappings are alphabetical, matching sklearn's LabelEncoder default behaviour
gender_map = {"Female": 0, "Male": 1}
working_status_map = {"Student": 0, "Working Professional": 1}
sleep_map = {"5-6 hours": 0, "7-8 hours": 1, "Less than 5 hours": 2, "More than 8 hours": 3}
diet_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
suicidal_thoughts_map = {"No": 0, "Yes": 1}
family_history_map = {"No": 0, "Yes": 1}

# ── User input form ──────────────────────────────────────────────────────────
st.subheader("Tell us about yourself")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", list(gender_map.keys()))
    age = st.slider("Age", 15, 80, 25)
    working_status = st.selectbox("Are you a student or working professional?", list(working_status_map.keys()))
    sleep_duration = st.selectbox("Sleep duration", list(sleep_map.keys()))
    dietary_habits = st.selectbox("Dietary habits", list(diet_map.keys()))
    work_study_hours = st.slider("Work/Study hours per day", 0, 16, 6)

with col2:
    academic_pressure = st.slider("Academic pressure (0 = none, 5 = very high)", 0, 5, 0)
    work_pressure = st.slider("Work pressure (0 = none, 5 = very high)", 0, 5, 0)
    study_satisfaction = st.slider("Study satisfaction (0 = low, 5 = high)", 0, 5, 0)
    job_satisfaction = st.slider("Job satisfaction (0 = low, 5 = high)", 0, 5, 0)
    financial_stress = st.slider("Financial stress (0 = none, 5 = very high)", 0, 5, 2)
    cgpa = st.slider("CGPA (if student, else leave at 0)", 0.0, 10.0, 0.0, step=0.01)

suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", list(suicidal_thoughts_map.keys()), horizontal=True)
family_history = st.radio("Family history of mental illness?", list(family_history_map.keys()), horizontal=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Check my risk", type="primary"):

    # Build input row in the same column order as training data
    input_dict = {
        "Gender": gender_map[gender],
        "Age": age,
        "Working Professional or Student": working_status_map[working_status],
        "Profession": 0,  # simplified — not collected in this form
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_map[sleep_duration],
        "Dietary Habits": diet_map[dietary_habits],
        "Degree": 0,  # simplified — not collected in this form
        "Have you ever had suicidal thoughts ?": suicidal_thoughts_map[suicidal_thoughts],
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history_map[family_history],
    }

    # Ensure correct column order matching training
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"**Higher risk indicated** — estimated probability: {probability*100:.1f}%")
        st.write(
            "This result suggests some patterns associated with depression risk in the model's training data. "
            "This is **not a diagnosis**. Please consider speaking with a doctor, counsellor, or mental health "
            "professional, or reach out to a mental health helpline in your country."
        )
    else:
        st.success(f"**Lower risk indicated** — estimated probability: {probability*100:.1f}%")
        st.write(
            "This result suggests fewer patterns associated with depression risk in the model's training data. "
            "This tool is not a substitute for professional advice — if you are ever concerned about your "
            "mental health, please reach out to a qualified professional."
        )

st.divider()
st.caption(
    "Disclaimer: This application is an academic project (COM 763 — Advanced Machine Learning) and is "
    "**not a certified medical or diagnostic tool**. It is intended for educational and demonstrative purposes only."
)