import streamlit as st
import numpy as np
import pickle

# --- Load scaler and model ---
@st.cache_resource
def load_artifacts():
    with open('scalers.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('best_models.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

# --- Title ---
st.title("🩺 Diabetes Risk Predictor")
st.markdown("Enter your health indicators below to check your diabetes risk level.")

# --- Input Form ---
with st.form("diabetes_form"):
    st.subheader("Enter your Health Metrics")

    col1, col2 = st.columns(2)

    with col1:
        HighBP = st.selectbox("High Blood Pressure", [0, 1])
        HighChol = st.selectbox("High Cholesterol", [0, 1])
        CholCheck = st.selectbox("Had Cholesterol Check", [0, 1])
        BMI = st.number_input("Body Mass Index", 10.0, 60.0, 25.0)
        Smoker = st.selectbox("Smoker", [0, 1])
        Stroke = st.selectbox("Had Stroke", [0, 1])
        HeartDiseaseorAttack = st.selectbox("Heart Disease/Attack", [0, 1])
        PhysActivity = st.selectbox("Physically Active", [0, 1])
        Fruits = st.selectbox("Consumes Fruits Regularly", [0, 1])
        Veggies = st.selectbox("Consumes Vegetables Regularly", [0, 1])

    with col2:
        HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption", [0, 1])
        GenHlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        MentHlth = st.slider("Mental Health (days, 0-30)", 0, 30, 5)
        PhysHlth = st.slider("Physical Health (days, 0-30)", 0, 30, 5)
        DiffWalk = st.selectbox("Difficulty Walking", [0, 1])
        Sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        Age = st.slider("Age Category (1=18-24, ... 13=80+)", 1, 13, 5)
        PhysicallyHvy = st.selectbox("Heavy Physical Activity", [0, 1])
        GenHlthBinary = st.selectbox("General Health Binary (1=Good/Excellent, 0=Fair/Poor)", [0, 1])

    submitted = st.form_submit_button("🔍 Predict")

# --- Predict ---
if submitted:
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity,
                            Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex,
                            Age, PhysicallyHvy, GenHlthBinary]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    class_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}

    st.success(f"🩺 **Prediction:** {class_map[prediction]}")
    st.info(f"📊 **Probabilities:** No Diabetes: {probabilities[0]:.2f}, "
            f"Prediabetes: {probabilities[1]:.2f}, Diabetes: {probabilities[2]:.2f}")
