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

def yes_no_selectbox(label, key=None):
    choice = st.selectbox(label, ['No', 'Yes'], key=key)
    return 1 if choice == 'Yes' else 0

def sex_selectbox(label, key=None):
    choice = st.selectbox(label, ['Female', 'Male'], key=key)
    return 1 if choice == 'Male' else 0

col1, col2 = st.columns(2)

with col1:
    HighBP = yes_no_selectbox("High Blood Pressure", key="HighBP")
with col2:
    HighChol = yes_no_selectbox("High Cholesterol", key="HighChol")

with col1:
    CholCheck = yes_no_selectbox("Cholesterol Check in last 5 years", key="CholCheck")
with col2:
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="BMI")

with col1:
    Smoker = yes_no_selectbox("Smoker", key="Smoker")
with col2:
    Stroke = yes_no_selectbox("Had Stroke", key="Stroke")

with col1:
    HeartDiseaseorAttack = yes_no_selectbox("Heart Disease or Attack", key="HeartDiseaseorAttack")
with col2:
    PhysActivity = yes_no_selectbox("Physically Active", key="PhysActivity")

with col1:
    Fruits = yes_no_selectbox("Eat Fruits", key="Fruits")
with col2:
    Veggies = yes_no_selectbox("Eat Vegetables", key="Veggies")

with col1:
    HvyAlcoholConsump = yes_no_selectbox("Heavy Alcohol Consumption", key="HvyAlcoholConsump")
with col2:
    GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3, key="GenHlth")

with col1:
    MentHlth = st.number_input("Number of days mental health not good (0-30)", 0, 30, 0, key="MentHlth")
with col2:
    PhysHlth = st.number_input("Number of days physical health not good (0-30)", 0, 30, 0, key="PhysHlth")

with col1:
    DiffWalk = yes_no_selectbox("Difficulty walking", key="DiffWalk")
with col2:
    Sex = sex_selectbox("Sex", key="Sex")

with col1:
    Age = st.number_input("Age (years)", min_value=18, max_value=120, value=50, key="Age")
with col2:
    st.write("")  # Blank to keep layout consistent

class_names = ["No diabetes", "Pre-diabetes", "Diabetes"]

def predict_and_show(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.markdown("### 🩺 Prediction Result")
    st.markdown(f"**Predicted Class:** :blue[{class_names[prediction]}]")

    st.markdown("### 📊 Class Probabilities")

    # Convert probabilities to percentage
    percentages = probabilities * 100
    prob_dict = {class_names[i]: percentages[i] for i in range(len(class_names))}
    
    # Create a horizontal bar chart using Plotly
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
        yaxis=dict(autorange="reversed"),  # Highest on top
        margin=dict(l=100, r=20, t=20, b=20),
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)

# Main prediction from user input
if st.button("Predict Diabetes Risk"):
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                            HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth,
                            DiffWalk, Sex, Age]])
    with st.spinner("Predicting..."):
        predict_and_show(input_data)

# Button to test a predefined high-risk input
if st.button("Test High-Risk Input"):
    high_risk_input = np.array([[1, 1, 1, 50.0, 1, 1, 1, 0, 0, 0, 1, 5, 30, 30, 1, 1, 70]])
    with st.spinner("Predicting high-risk case..."):
        predict_and_show(high_risk_input)
