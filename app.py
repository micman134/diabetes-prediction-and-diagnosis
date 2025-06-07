import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model and scaler ---
@st.cache_resource
def load_model_scaler():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Risk Prediction")

# Define numeric ranges for inputs (based on dataset documentation or domain knowledge)
BMI_MIN, BMI_MAX = 10, 60  # approximate BMI realistic range
MENTHLTH_MIN, MENTHLTH_MAX = 0, 30  # days poor mental health last month
PHYSHLTH_MIN, PHYSHLTH_MAX = 0, 30  # days poor physical health last month
AGE_MIN, AGE_MAX = 18, 100
EDUCATION_MIN, EDUCATION_MAX = 1, 6
INCOME_MIN, INCOME_MAX = 1, 8

# Helper to convert yes/no dropdown to int
def yes_no_to_int(val):
    return 1 if val == "Yes" else 0

# Layout with two columns per row
def two_cols(label1, widget1, label2, widget2):
    col1, col2 = st.columns(2)
    with col1:
        st.write(label1)
        widget1()
    with col2:
        st.write(label2)
        widget2()

st.markdown("### Please fill in the following health indicators:")

# Collect inputs
# Use st.session_state to hold temporary variables
inputs = {}

with st.form("input_form"):
    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        inputs['HighBP'] = yes_no_to_int(st.selectbox(
            "High Blood Pressure (HighBP)?",
            options=["Yes", "No"],
            help="Have you ever been told you have high blood pressure?",
            key="HighBP"))
    with col2:
        inputs['HighChol'] = yes_no_to_int(st.selectbox(
            "High Cholesterol (HighChol)?",
            options=["Yes", "No"],
            help="Have you ever been told you have high cholesterol?",
            key="HighChol"))
    
    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        inputs['CholCheck'] = yes_no_to_int(st.selectbox(
            "Cholesterol Check in last 5 years?",
            options=["Yes", "No"],
            help="Have you had your cholesterol checked in the last 5 years?",
            key="CholCheck"))
    with col2:
        inputs['BMI'] = st.number_input(
            "BMI (Body Mass Index)",
            min_value=BMI_MIN,
            max_value=BMI_MAX,
            value=25.0,
            help="BMI = weight in kg / (height in meters)^2. Typical adult range 10 to 60.",
            format="%.1f",
            key="BMI")
    
    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        inputs['Smoker'] = yes_no_to_int(st.selectbox(
            "Are you a smoker?",
            options=["Yes", "No"],
            help="Do you currently smoke?",
            key="Smoker"))
    with col2:
        inputs['Stroke'] = yes_no_to_int(st.selectbox(
            "Have you ever had a stroke?",
            options=["Yes", "No"],
            help="Have you ever been told you had a stroke?",
            key="Stroke"))
    
    # Row 4
    col1, col2 = st.columns(2)
    with col1:
        inputs['HeartDiseaseorAttack'] = yes_no_to_int(st.selectbox(
            "Heart Disease or Attack?",
            options=["Yes", "No"],
            help="Have you ever been told you have coronary heart disease or myocardial infarction (heart attack)?",
            key="HeartDiseaseorAttack"))
    with col2:
        inputs['PhysActivity'] = yes_no_to_int(st.selectbox(
            "Physically Active?",
            options=["Yes", "No"],
            help="Have you done any physical activity or exercise in the past 30 days other than your regular job?",
            key="PhysActivity"))
    
    # Row 5
    col1, col2 = st.columns(2)
    with col1:
        inputs['Fruits'] = yes_no_to_int(st.selectbox(
            "Eat Fruits?",
            options=["Yes", "No"],
            help="Do you consume fruit at least once per day?",
            key="Fruits"))
    with col2:
        inputs['Veggies'] = yes_no_to_int(st.selectbox(
            "Eat Vegetables?",
            options=["Yes", "No"],
            help="Do you consume vegetables at least once per day?",
            key="Veggies"))
    
    # Row 6
    col1, col2 = st.columns(2)
    with col1:
        inputs['HvyAlcoholConsump'] = yes_no_to_int(st.selectbox(
            "Heavy Alcohol Consumption?",
            options=["Yes", "No"],
            help="Do you have heavy alcohol consumption? (more than 14 drinks/week for men, 7 for women)",
            key="HvyAlcoholConsump"))
    with col2:
        inputs['AnyHealthcare'] = yes_no_to_int(st.selectbox(
            "Has any healthcare coverage?",
            options=["Yes", "No"],
            help="Do you have any kind of healthcare coverage?",
            key="AnyHealthcare"))
    
    # Row 7
    col1, col2 = st.columns(2)
    with col1:
        inputs['NoDocbcCost'] = yes_no_to_int(st.selectbox(
            "Could not see doctor because of cost?",
            options=["Yes", "No"],
            help="During the past 12 months, was there a time you needed to see a doctor but could not because of cost?",
            key="NoDocbcCost"))
    with col2:
        inputs['GenHlth'] = st.slider(
            "General Health (1=Excellent to 5=Poor)",
            min_value=1,
            max_value=5,
            value=3,
            help="Self-rated general health status from 1 (Excellent) to 5 (Poor).",
            key="GenHlth")
    
    # Row 8
    col1, col2 = st.columns(2)
    with col1:
        inputs['MentHlth'] = st.number_input(
            "Mental Health (days poor in last month)",
            min_value=MENTHLTH_MIN,
            max_value=MENTHLTH_MAX,
            value=0,
            help="Number of days in the past 30 days your mental health was not good.",
            key="MentHlth")
    with col2:
        inputs['PhysHlth'] = st.number_input(
            "Physical Health (days poor in last month)",
            min_value=PHYSHLTH_MIN,
            max_value=PHYSHLTH_MAX,
            value=0,
            help="Number of days in the past 30 days your physical health was not good.",
            key="PhysHlth")
    
    # Row 9
    col1, col2 = st.columns(2)
    with col1:
        inputs['DiffWalk'] = yes_no_to_int(st.selectbox(
            "Do you have serious difficulty walking or climbing stairs?",
            options=["Yes", "No"],
            help="Do you have serious difficulty walking or climbing stairs?",
            key="DiffWalk"))
    with col2:
        inputs['Sex'] = st.selectbox(
            "Sex",
            options=["Male", "Female"],
            help="Select your sex.",
            key="Sex")
        inputs['Sex'] = 1 if inputs['Sex'] == "Male" else 0
    
    # Row 10
    col1, col2 = st.columns(2)
    with col1:
        inputs['Age'] = st.number_input(
            "Age",
            min_value=AGE_MIN,
            max_value=AGE_MAX,
            value=30,
            help="Age in years",
            key="Age")
    with col2:
        inputs['Education'] = st.slider(
            "Education Level (1=Never attended to 6=College graduate)",
            min_value=EDUCATION_MIN,
            max_value=EDUCATION_MAX,
            value=3,
            help="Education levels from 1 (Never attended school) to 6 (College graduate).",
            key="Education")
    
    # Row 11
    col1, col2 = st.columns(2)
    with col1:
        inputs['Income'] = st.slider(
            "Income Level (1=Less than $10,000 to 8=$75,000 or more)",
            min_value=INCOME_MIN,
            max_value=INCOME_MAX,
            value=4,
            help="Income levels from 1 (less than $10,000) to 8 ($75,000 or more).",
            key="Income")
        # Empty second column for balance
        col2.write("")
    
    submitted = st.form_submit_button("Predict Diabetes Risk")

if submitted:
    # Create DataFrame from inputs
    input_df = pd.DataFrame([inputs])

    # Reorder columns exactly as model expects
    feature_order = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
                     'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
                     'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
                     'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
    input_df = input_df[feature_order]

    # Scale numeric columns
    numeric_cols = ['BMI', 'MentHlth', 'PhysHlth']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"High risk of Diabetes detected! Probability: {pred_prob:.2%}" if pred_prob else "High risk of Diabetes detected!")
    else:
        st.success(f"Low risk of Diabetes detected! Probability: {pred_prob:.2%}" if pred_prob else "Low risk of Diabetes detected!")
