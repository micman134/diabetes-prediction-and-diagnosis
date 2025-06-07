import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model and scaler from file
@st.cache_resource
def load_model_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

# Load model and scaler
model, scaler = load_model_scaler()

# Configure Streamlit layout
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("🧪 Diabetes Risk Prediction App")
st.markdown("""
This app uses your health indicators to predict whether you're at risk of diabetes.  
👉 **Answer Yes/No questions** and **fill numeric inputs**, then click Predict.
""")

# Fields
yes_no_fields = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

numeric_fields = [
    'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Sex',
    'Age', 'Education', 'Income'
]

# User input storage
inputs = {}

# Input form
with st.form("diabetes_form"):
    st.subheader("✅ Yes / No Questions")
    for i in range(0, len(yes_no_fields), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(yes_no_fields):
                field = yes_no_fields[i + j]
                value = cols[j].selectbox(f"{field}:", ["No", "Yes"], key=field)
                inputs[field] = 1 if value == "Yes" else 0

    st.subheader("🔢 Numeric Inputs")
    for i in range(0, len(numeric_fields), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(numeric_fields):
                field = numeric_fields[i + j]
                value = cols[j].text_input(f"{field}:", key=field)
                inputs[field] = value

    submitted = st.form_submit_button("🔍 Predict Diabetes")

# Prediction logic
if submitted:
    try:
        # Build input list
        input_values = []

        # Binary inputs
        for field in yes_no_fields:
            input_values.append(inputs[field])

        # Numeric inputs
        for field in numeric_fields:
            input_values.append(float(inputs[field]))

        # Create DataFrame
        all_fields = yes_no_fields + numeric_fields
        input_df = pd.DataFrame([input_values], columns=all_fields)

        # Scale relevant numeric columns
        num_cols = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Display result
        st.subheader("📈 Prediction Result")
        if prediction == 1:
            st.error("⚠️ You may be at risk of diabetes.")
        else:
            st.success("✅ You are likely not at risk of diabetes.")

        if probability is not None:
            st.info(f"🔢 Risk Probability: **{probability:.2%}**")

    except ValueError:
        st.error("❌ Please make sure all numeric inputs are filled with valid numbers.")
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
