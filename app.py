import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Prediction App", layout="wide")
st.title("Diabetes Prediction App")
st.markdown("Fill in the health indicators below to predict diabetes risk.")

@st.cache_resource
def load_model_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

def yes_no_to_num(val):
    return 1 if val == "Yes" else 0

with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", ["Male", "Female"], help="Select your biological sex")
        sex_num = 1 if sex == "Male" else 0
        
        highbp = st.selectbox("High Blood Pressure (HighBP)", ["Yes", "No"], help="Do you have high blood pressure?")
        highbp_num = yes_no_to_num(highbp)
        
        cholcheck = st.selectbox("Cholesterol Check (CholCheck)", ["Yes", "No"], help="Had cholesterol checked in past 5 years?")
        cholcheck_num = yes_no_to_num(cholcheck)
        
        smoker = st.selectbox("Smoker", ["Yes", "No"], help="Do you currently smoke?")
        smoker_num = yes_no_to_num(smoker)
        
        heart_disease = st.selectbox("Heart Disease or Attack", ["Yes", "No"], help="Ever had heart disease or attack?")
        heart_disease_num = yes_no_to_num(heart_disease)
        
        fruits = st.selectbox("Consume Fruits", ["Yes", "No"], help="Eat fruits daily?")
        fruits_num = yes_no_to_num(fruits)
        
        hvy_alcohol = st.selectbox("Heavy Alcohol Consumption", ["Yes", "No"], help="Heavy drinker?")
        hvy_alcohol_num = yes_no_to_num(hvy_alcohol)
        
        nodocbcost = st.selectbox("No Doctor Because of Cost", ["Yes", "No"], help="Avoided doctor due to cost?")
        nodocbcost_num = yes_no_to_num(nodocbcost)
        
        diffwalk = st.selectbox("Difficulty Walking", ["Yes", "No"], help="Difficulty walking or climbing stairs?")
        diffwalk_num = yes_no_to_num(diffwalk)

    with col2:
        highchol = st.selectbox("High Cholesterol (HighChol)", ["Yes", "No"], help="Have high cholesterol?")
        highchol_num = yes_no_to_num(highchol)
        
        stroke = st.selectbox("Stroke", ["Yes", "No"], help="Ever had stroke?")
        stroke_num = yes_no_to_num(stroke)
        
        phys_activity = st.selectbox("Physical Activity", ["Yes", "No"], help="Physically active past month?")
        phys_activity_num = yes_no_to_num(phys_activity)
        
        veggies = st.selectbox("Consume Vegetables", ["Yes", "No"], help="Eat vegetables daily?")
        veggies_num = yes_no_to_num(veggies)
        
        any_healthcare = st.selectbox("Any Healthcare Access", ["Yes", "No"], help="Have healthcare coverage?")
        any_healthcare_num = yes_no_to_num(any_healthcare)
        
        genhlth = st.slider("General Health (GenHlth)", 1, 5, 3, help="1=Excellent to 5=Poor")
        
        menthlth = st.number_input("Mental Health (MentHlth) [days]", min_value=0, max_value=30, value=0, step=1, help="Days mental health not good")
        
        physhlth = st.number_input("Physical Health (PhysHlth) [days]", min_value=0, max_value=30, value=0, step=1, help="Days physical health not good")
        
        bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=70.0, value=25.0, step=0.1, format="%.1f", help="BMI between 12 and 70")
        
        age = st.slider("Age Category", 1, 13, 5, help="1=18-24 ... 13=80+")
        
        education = st.slider("Education Level", 1, 6, 3, help="1=No school ... 6=College grad")
        
        income = st.slider("Income Level", 1, 8, 5, help="1= <$10k ... 8= >$75k")
    
    submit = st.form_submit_button("Predict Diabetes Risk")

if submit:
    input_dict = {
        'HighBP': highbp_num,
        'HighChol': highchol_num,
        'CholCheck': cholcheck_num,
        'BMI': bmi,
        'Smoker': smoker_num,
        'Stroke': stroke_num,
        'HeartDiseaseorAttack': heart_disease_num,
        'PhysActivity': phys_activity_num,
        'Fruits': fruits_num,
        'Veggies': veggies_num,
        'HvyAlcoholConsump': hvy_alcohol_num,
        'AnyHealthcare': any_healthcare_num,
        'NoDocbcCost': nodocbcost_num,
        'GenHlth': genhlth,
        'MentHlth': menthlth,
        'PhysHlth': physhlth,
        'DiffWalk': diffwalk_num,
        'Sex': sex_num,
        'Age': age,
        'Education': education,
        'Income': income
    }
    input_df = pd.DataFrame([input_dict])
    
    # Scale numeric columns only
    numeric_cols = ['BMI', 'MentHlth', 'PhysHlth']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("⚠️ High risk of diabetes detected.")
    else:
        st.success("✅ Low risk of diabetes detected.")
    
    if prediction_proba is not None:
        st.info(f"Prediction confidence: {prediction_proba:.2%}")
