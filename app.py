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

# Debug test: Check scaler and model with sample data
def debug_model_and_scaler():
    # Example input: 17 features (make sure this matches your model input exactly)
    sample_input = np.array([[1, 1, 1, 25.0, 1, 0, 0, 1, 1, 1, 0, 3, 0, 0, 0, 1, 45]])
    
    st.write("### Debug Info")
    st.write("Sample raw input:", sample_input)
    
    try:
        sample_scaled = scaler.transform(sample_input)
        st.write("Sample scaled input:", sample_scaled)
    except Exception as e:
        st.error(f"Scaler transform error: {e}")
        return
    
    try:
        pred = model.predict(sample_scaled)
        st.write("Sample prediction:", pred)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return
    
    try:
        probs = model.predict_proba(sample_scaled)
        st.write("Sample prediction probabilities:", probs)
    except AttributeError:
        st.warning("Model does not support predict_proba()")
    except Exception as e:
        st.error(f"Prediction probabilities error: {e}")

# Call debug function (comment out if not needed)
debug_model_and_scaler()

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
    st.write("")  # Blank for layout

if st.button("Predict Diabetes Risk"):
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
