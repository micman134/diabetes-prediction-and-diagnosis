import streamlit as st
import numpy as np
import pickle

# Cache model and scaler loading for efficiency
@st.cache_data
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")

st.markdown("""
Please fill in the following information to predict diabetes risk.
""")

def yes_no_selectbox(label):
    choice = st.selectbox(label, ['No', 'Yes'])
    return 1 if choice == 'Yes' else 0

def sex_selectbox(label):
    choice = st.selectbox(label, ['Female', 'Male'])
    return 1 if choice == 'Male' else 0

col1, col2 = st.columns(2)

with col1:
    HighBP = yes_no_selectbox("High Blood Pressure")
with col2:
    HighChol = yes_no_selectbox("High Cholesterol")

with col1:
    CholCheck = yes_no_selectbox("Cholesterol Check in last 5 years")
with col2:
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col1:
    Smoker = yes_no_selectbox("Smoker")
with col2:
    Stroke = yes_no_selectbox("Had Stroke")

with col1:
    HeartDiseaseorAttack = yes_no_selectbox("Heart Disease or Attack")
with col2:
    PhysActivity = yes_no_selectbox("Physically Active")

with col1:
    Fruits = yes_no_selectbox("Eat Fruits")
with col2:
    Veggies = yes_no_selectbox("Eat Vegetables")

with col1:
    HvyAlcoholConsump = yes_no_selectbox("Heavy Alcohol Consumption")
with col2:
    GenHlth = st.slider("General Health (1=Excellent to 5=Poor)", 1, 5, 3)

with col1:
    MentHlth = st.number_input("Number of days mental health not good (0-30)", 0, 30, 0)
with col2:
    PhysHlth = st.number_input("Number of days physical health not good (0-30)", 0, 30, 0)

with col1:
    DiffWalk = yes_no_selectbox("Difficulty walking")
with col2:
    Sex = sex_selectbox("Sex")

with col1:
    Age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)
with col2:
    st.write("")  # Blank to keep layout consistent

if st.button("Predict Diabetes Risk"):
    input_data = np.array([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
                            HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                            HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth,
                            DiffWalk, Sex, Age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.write(f"### Predicted Diabetes Class: {prediction}")
    st.write("### Class Probabilities:")
    st.write(f"Class 0 (No diabetes): {probabilities[0]:.3f}")
    st.write(f"Class 1 (Pre-diabetes): {probabilities[1]:.3f}")
    st.write(f"Class 2 (Diabetes): {probabilities[2]:.3f}")
