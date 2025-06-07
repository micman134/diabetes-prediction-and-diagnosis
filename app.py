import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

st.title("Diabetes Prediction")

# Define input widgets for each feature except target
# Binary/categorical features: HighBP, HighChol, CholCheck, Smoker, Stroke,
# HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
# AnyHealthcare, NoDocbcCost, DiffWalk, Sex
# Numerical: BMI, MentHlth, PhysHlth, Age, Education, Income, GenHlth (assumed ordinal)

# Helper function for yes/no to int
def yes_no_input(label):
    return 1 if st.radio(label, ("No", "Yes")) == "Yes" else 0

HighBP = yes_no_input("High Blood Pressure (HighBP)?")
HighChol = yes_no_input("High Cholesterol (HighChol)?")
CholCheck = yes_no_input("Cholesterol Check in past 5 years (CholCheck)?")
Smoker = yes_no_input("Current Smoker (Smoker)?")
Stroke = yes_no_input("Had Stroke (Stroke)?")
HeartDiseaseorAttack = yes_no_input("Heart Disease or Attack (HeartDiseaseorAttack)?")
PhysActivity = yes_no_input("Physical Activity in past 30 days (PhysActivity)?")
Fruits = yes_no_input("Eat Fruits (Fruits)?")
Veggies = yes_no_input("Eat Vegetables (Veggies)?")
HvyAlcoholConsump = yes_no_input("Heavy Alcohol Consumption (HvyAlcoholConsump)?")
AnyHealthcare = yes_no_input("Have Any Healthcare Coverage (AnyHealthcare)?")
NoDocbcCost = yes_no_input("Could not see doctor due to cost (NoDocbcCost)?")
DiffWalk = yes_no_input("Have Difficulty Walking (DiffWalk)?")

Sex = st.selectbox("Sex (1=Male, 0=Female)", options=[0, 1], index=1)

# Numeric inputs
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
MentHlth = st.number_input("Mental Health (days not good in past 30)", min_value=0, max_value=30, value=0)
PhysHlth = st.number_input("Physical Health (days not good in past 30)", min_value=0, max_value=30, value=0)
Age = st.slider("Age (in years)", 18, 100, 40)
Education = st.selectbox("Education Level (1=Never attended, ... 6=College Graduate)", options=list(range(1,7)), index=3)
Income = st.selectbox("Income Level (1=Less than $10k ... 8=$75k or more)", options=list(range(1,9)), index=4)
GenHlth = st.selectbox("General Health (1=Excellent ... 5=Poor)", options=list(range(1,6)), index=1)

# Put inputs into a dataframe row in correct column order
input_data = pd.DataFrame({
    'HighBP': [HighBP],
    'HighChol': [HighChol],
    'CholCheck': [CholCheck],
    'BMI': [BMI],
    'Smoker': [Smoker],
    'Stroke': [Stroke],
    'HeartDiseaseorAttack': [HeartDiseaseorAttack],
    'PhysActivity': [PhysActivity],
    'Fruits': [Fruits],
    'Veggies': [Veggies],
    'HvyAlcoholConsump': [HvyAlcoholConsump],
    'AnyHealthcare': [AnyHealthcare],
    'NoDocbcCost': [NoDocbcCost],
    'GenHlth': [GenHlth],
    'MentHlth': [MentHlth],
    'PhysHlth': [PhysHlth],
    'DiffWalk': [DiffWalk],
    'Sex': [Sex],
    'Age': [Age],
    'Education': [Education],
    'Income': [Income]
})

# Scale numeric columns before prediction
num_cols = ['BMI', 'MentHlth', 'PhysHlth']
input_data[num_cols] = scaler.transform(input_data[num_cols])

if st.button("Predict Diabetes"):
    pred = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    st.write(f"**Prediction:** {'Diabetes' if pred == 1 else 'No Diabetes'}")
    if pred_proba is not None:
        st.write(f"**Probability of Diabetes:** {pred_proba:.2%}")
