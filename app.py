import streamlit as st
import numpy as np
import pickle

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

# Input fields for exactly these 17 features, in order:
HighBP = st.selectbox("High Blood Pressure (1=Yes, 0=No)", [0, 1])
HighChol = st.selectbox("High Cholesterol (1=Yes, 0=No)", [0, 1])
CholCheck = st.selectbox("Cholesterol Check in last 5 years (1=Yes, 0=No)", [0, 1])
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
Smoker = st.selectbox("Smoker (1=Yes, 0=No)", [0, 1])
Stroke = st.selectbox("Had Stroke (1=Yes, 0=No)", [0, 1])
HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack (1=Yes, 0=No)", [0, 1])
PhysActivity = st.selectbox("Physically Active (1=Yes, 0=No)", [0, 1])
Fruits = st.selectbox("Eat Fruits (1=Yes, 0=No)", [0, 1])
Veggies = st.selectbox("Eat Vegetables (1=Yes, 0=No)", [0, 1])
HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption (1=Yes, 0=No)", [0, 1])
GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)
MentHlth = st.number_input("Number of days mental health not good (0-30)", 0, 30, 0)
PhysHlth = st.number_input("Number of days physical health not good (0-30)", 0, 30, 0)
DiffWalk = st.selectbox("Difficulty walking (1=Yes, 0=No)", [0, 1])
Sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
Age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)

if st.button("Predict Diabetes Risk"):
    # Prepare input data as 2D array with 17 features
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                            HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth,
                            DiffWalk, Sex, Age]])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction and get probabilities
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Display results
    st.write(f"### Predicted Diabetes Class: {prediction}")
    st.write(f"### Class Probabilities:")
    st.write(f"Class 0 (No diabetes): {probabilities[0]:.3f}")
    st.write(f"Class 1 (Pre-diabetes): {probabilities[1]:.3f}")
    st.write(f"Class 2 (Diabetes): {probabilities[2]:.3f}")
