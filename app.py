import streamlit as st
import pickle
import numpy as np

# Load scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Diabetes Prediction App")

st.write("Enter the patient's health details below:")

# Input fields according to your features after dropping:
# ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
#  'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
#  'HvyAlcoholConsump', 'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk',
#  'Sex', 'Age']

# Note: Sex is assumed encoded as 0 or 1

HighBP = st.selectbox("High Blood Pressure (1 = Yes, 0 = No)", options=[0,1], index=0)
HighChol = st.selectbox("High Cholesterol (1 = Yes, 0 = No)", options=[0,1], index=0)
CholCheck = st.selectbox("Cholesterol Check in Past 5 Years (1 = Yes, 0 = No)", options=[0,1], index=1)
BMI = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
Smoker = st.selectbox("Smoker (1 = Yes, 0 = No)", options=[0,1], index=0)
Stroke = st.selectbox("Stroke (1 = Yes, 0 = No)", options=[0,1], index=0)
HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack (1 = Yes, 0 = No)", options=[0,1], index=0)
PhysActivity = st.selectbox("Physically Active (1 = Yes, 0 = No)", options=[0,1], index=1)
Fruits = st.selectbox("Eat Fruits One or More Times Per Day (1 = Yes, 0 = No)", options=[0,1], index=1)
Veggies = st.selectbox("Eat Vegetables One or More Times Per Day (1 = Yes, 0 = No)", options=[0,1], index=1)
HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption (1 = Yes, 0 = No)", options=[0,1], index=0)
GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", min_value=1, max_value=5, value=3)
MentHlth = st.slider("Days of Poor Mental Health in Past 30 Days", min_value=0, max_value=30, value=0)
PhysHlth = st.slider("Days of Poor Physical Health in Past 30 Days", min_value=0, max_value=30, value=0)
DiffWalk = st.selectbox("Difficulty Walking or Climbing Stairs (1 = Yes, 0 = No)", options=[0,1], index=0)
Sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0,1], index=1)
Age = st.selectbox(
    "Age Category",
    options=[1,2,3,4,5,6,7,8,9],
    format_func=lambda x: {
        1: "18-24",
        2: "25-29",
        3: "30-34",
        4: "35-39",
        5: "40-44",
        6: "45-49",
        7: "50-54",
        8: "55-59",
        9: "60+"
    }[x],
    index=0
)

if st.button("Predict Diabetes Class"):
    # Prepare feature array in correct order
    features = np.array([[
        HighBP, HighChol, CholCheck, BMI, Smoker,
        Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
        HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk,
        Sex, Age
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    pred = model.predict(features_scaled)[0]

    # Map prediction to label if you have class meaning
    class_mapping = {
        0: "No Diabetes",
        1: "Prediabetes",
        2: "Diabetes"
    }

    st.success(f"Prediction: **{class_mapping.get(pred, 'Unknown')}**")
