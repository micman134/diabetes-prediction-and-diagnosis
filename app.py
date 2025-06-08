import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go

# Set page title and icon
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")

# Cache model and scaler loading for efficiency
@st.cache_data
def load_artifacts():
    with open('scalers.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_models.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")

st.markdown("""
Please fill in the following information to predict diabetes risk.
""")

def yes_no_selectbox(label, help_text=""):
    choice = st.selectbox(label, ['No', 'Yes'], help=help_text)
    return 1 if choice == 'Yes' else 0

def sex_selectbox(label):
    choice = st.selectbox(label, ['Female', 'Male'], help="Select your biological sex")
    return 1 if choice == 'Male' else 0

col1, col2 = st.columns(2)

with col1:
    HighBP = yes_no_selectbox("High Blood Pressure", "Do you have high blood pressure?")
with col2:
    HighChol = yes_no_selectbox("High Cholesterol", "Do you have high cholesterol?")

with col1:
    CholCheck = yes_no_selectbox("Cholesterol Check in last 5 years", "Have you had a cholesterol check in the last 5 years?")
with col2:
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                          help="Body Mass Index, typical healthy range: 18.5 to 24.9")

with col1:
    Smoker = yes_no_selectbox("Smoker", "Do you currently smoke?")
with col2:
    Stroke = yes_no_selectbox("Had Stroke", "Have you ever had a stroke?")

with col1:
    HeartDiseaseorAttack = yes_no_selectbox("Heart Disease or Attack", "Have you had a heart disease or heart attack?")
with col2:
    PhysActivity = yes_no_selectbox("Physically Active", "Are you physically active?")

with col1:
    Fruits = yes_no_selectbox("Eat Fruits", "Do you regularly eat fruits?")
with col2:
    Veggies = yes_no_selectbox("Eat Vegetables", "Do you regularly eat vegetables?")

with col1:
    HvyAlcoholConsump = yes_no_selectbox("Heavy Alcohol Consumption", "Do you consume heavy amounts of alcohol?")
with col2:
    GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3,
                        help="Rate your general health from excellent to poor")

with col1:
    MentHlth = st.number_input("Number of days mental health not good (0-30)", 0, 30, 0,
                              help="Number of days your mental health was not good in the past month")
with col2:
    PhysHlth = st.number_input("Number of days physical health not good (0-30)", 0, 30, 0,
                              help="Number of days your physical health was not good in the past month")

with col1:
    DiffWalk = yes_no_selectbox("Difficulty walking", "Do you have difficulty walking or climbing stairs?")
with col2:
    Sex = sex_selectbox("Sex")

with col1:
    Age = st.number_input("Age (years)", min_value=18, max_value=120, value=50,
                          help="Enter your age in years")
with col2:
    st.write("")  # Blank to keep layout consistent

# Simple input validation example
input_valid = True
if BMI < 10 or BMI > 60:
    input_valid = False
    st.warning("Please enter a BMI between 10 and 60.")

if Age < 18 or Age > 120:
    input_valid = False
    st.warning("Please enter an age between 18 and 120.")

if MentHlth < 0 or MentHlth > 30 or PhysHlth < 0 or PhysHlth > 30:
    input_valid = False
    st.warning("Please enter days between 0 and 30 for health inputs.")

if st.button("Predict Diabetes Risk"):
    if not input_valid:
        st.error("Please correct invalid inputs before prediction.")
    else:
        with st.spinner('Predicting...'):
            input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                                    HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                                    HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth,
                                    DiffWalk, Sex, Age]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

        class_names = ["No diabetes", "Pre-diabetes", "Diabetes"]

        st.markdown("### 🩺 Prediction Result")
        st.markdown(f"**Predicted Class:** :blue[{class_names[prediction]}]")

        st.markdown("### 🔍 Model Confidence per Class")
        for i, class_name in enumerate(class_names):
            conf = probabilities[i] * 100
            if i == prediction:
                st.markdown(f"- **{class_name}: {conf:.1f}%** (Predicted)")
            else:
                st.markdown(f"- {class_name}: {conf:.1f}%")

        st.markdown("### 📊 Class Probabilities")

        percentages = probabilities * 100
        prob_dict = {class_names[i]: percentages[i] for i in range(len(class_names))}
        
        fig = go.Figure(go.Bar(
            x=list(prob_dict.values()),
            y=list(prob_dict.keys()),
            orientation='h',
            text=[f"{v:.1f}%" for v in prob_dict.values()],
            textposition='auto',
            marker_color=['#2ca02c', '#ff7f0e', '#d62728']  # Green, Orange, Red
        ))

        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Class",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=100, r=20, t=20, b=20),
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)
