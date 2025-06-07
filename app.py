import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page config - MUST be first Streamlit command
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("Diabetes Prediction App")
st.markdown("Fill in the health indicators below to predict diabetes risk.")

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

# Helper for Yes/No dropdown -> numeric conversion
def yes_no_to_num(val):
    return 1 if val == "Yes" else 0

# Input fields with tooltips and limits
with st.form("input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sex = st.selectbox("Sex", options=["Male", "Female"], help="Select your biological sex")
        sex_num = 1 if sex == "Male" else 0
        
        highbp = st.selectbox("High Blood Pressure (HighBP)", ["Yes", "No"], help="Do you have high blood pressure?")
        highbp_num = yes_no_to_num(highbp)
        
        cholcheck = st.selectbox("Cholesterol Check (CholCheck)", ["Yes", "No"], help="Have you had your cholesterol checked in the past 5 years?")
        cholcheck_num = yes_no_to_num(cholcheck)
        
        smoker = st.selectbox("Smoker", ["Yes", "No"], help="Do you currently smoke?")
        smoker_num = yes_no_to_num(smoker)
        
        heart_disease = st.selectbox("Heart Disease or Attack", ["Yes", "No"], help="Have you ever had heart disease or a heart attack?")
        heart_disease_num = yes_no_to_num(heart_disease)
        
        fruits = st.selectbox("Consume Fruits", ["Yes", "No"], help="Do you eat fruits daily?")
        fruits_num = yes_no_to_num(fruits)
        
        hvy_alcohol = st.selectbox("Heavy Alcohol Consumption (HvyAlcoholConsump)", ["Yes", "No"], help="Do you drink heavily (more than 14 drinks/week for men, 7 for women)?")
        hvy_alcohol_num = yes_no_to_num(hvy_alcohol)
        
        nodocbcost = st.selectbox("No Doctor Because of Cost (NoDocbcCost)", ["Yes", "No"], help="Did you not see a doctor due to cost in the past 12 months?")
        nodocbcost_num = yes_no_to_num(nodocbcost)
        
        diffwalk = st.selectbox("Difficulty Walking (DiffWalk)", ["Yes", "No"], help="Do you have serious difficulty walking or climbing stairs?")
        diffwalk_num = yes_no_to_num(diffwalk)

    with col2:
        highchol = st.selectbox("High Cholesterol (HighChol)", ["Yes", "No"], help="Do you have high cholesterol?")
        highchol_num = yes_no_to_num(highchol)
        
        stroke = st.selectbox("Stroke", ["Yes", "No"], help="Have you ever had a stroke?")
        stroke_num = yes_no_to_num(stroke)
        
        phys_activity = st.selectbox("Physical Activity (PhysActivity)", ["Yes", "No"], help="Do you participate in physical activity/exercise during the past month?")
        phys_activity_num = yes_no_to_num(phys_activity)
        
        veggies = st.selectbox("Consume Vegetables", ["Yes", "No"], help="Do you eat vegetables daily?")
        veggies_num = yes_no_to_num(veggies)
        
        any_healthcare = st.selectbox("Any Healthcare Access", ["Yes", "No"], help="Do you have any kind of healthcare coverage?")
        any_healthcare_num = yes_no_to_num(any_healthcare)
        
        genhlth = st.slider("General Health (GenHlth)", min_value=1, max_value=5, value=3, help="""
        1 = Excellent  
        2 = Very good  
        3 = Good  
        4 = Fair  
        5 = Poor
        """)
        
        menthlth = st.number_input("Mental Health (MentHlth) [0-30 days]", min_value=0, max_value=30, value=0, help="Number of days mental health was not good in past 30 days")
        
        physhlth = st.number_input("Physical Health (PhysHlth) [0-30 days]", min_value=0, max_value=30, value=0, help="Number of days physical health was not good in past 30 days")
        
        bmi = st.number_input("Body Mass Index (BMI) [12.0 - 70.0]", min_value=12.0, max_value=70.0, value=25.0, step=0.1, help="BMI between 12 and 70")
        
        age = st.slider("Age Category (Age)", min_value=1, max_value=13, value=5, help="""
        Age categories:  
        1: 18-24, 2: 25-29, 3: 30-34, 4: 35-39, 5: 40-44, 6: 45-49, 7: 50-54, 8: 55-59, 9: 60-64, 10: 65-69, 11: 70-74, 12: 75-79, 13: 80 or older
        """)
        
        education = st.slider("Education Level", min_value=1, max_value=6, value=3, help="""
        1 = Never attended school or only kindergarten  
        2 = Grades 1 through 8 (Elementary)  
        3 = Grades 9 through 11 (Some high school)  
        4 = Grade 12 or GED (High school graduate)  
        5 = College 1 year to 3 years (Some college or technical school)  
        6 = College 4 years or more (College graduate)
        """)
        
        income = st.slider("Income Level", min_value=1, max_value=8, value=5, help="""
        1 = Less than $10,000  
        2 = $10,000 to $15,000  
        3 = $15,000 to $20,000  
        4 = $20,000 to $25,000  
        5 = $25,000 to $35,000  
        6 = $35,000 to $50,000  
        7 = $50,000 to $75,000  
        8 = $75,000 or more
        """)

    submit = st.form_submit_button("Predict Diabetes Risk")

if submit:
    # Create dataframe from inputs
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
    
    # Scale numeric columns
    numeric_cols = ['BMI', 'MentHlth', 'PhysHlth']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This input indicates a **high risk** of diabetes.")
    else:
        st.success("✅ This input indicates **low risk** of diabetes.")
    
    if prediction_proba is not None:
        st.info(f"Prediction Probability for diabetes: {prediction_proba:.2%}")
