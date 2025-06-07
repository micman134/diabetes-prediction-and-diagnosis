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

# Load once
model, scaler = load_model_scaler()

# Streamlit page setup
st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🧪 Diabetes Prediction App")
st.markdown("""
Enter your health details to predict your risk of diabetes.  
✅ Select "Yes" or "No" for lifestyle/medical questions  
✍️ Enter numeric values like BMI, Age, etc.
""")

# Define yes/no fields and numeric fields
yes_no_fields = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

numeric_fields = [
    'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Sex',
    'Age', 'Education', 'Income'
]

# Dictionary to hold inputs
inputs = {}

# Form for input
with st.form("diabetes_form"):
    st.subheader("🟩 Yes/No Questions")
    for field in yes_no_fields:
        choice = st.selectbox(f"{field}:", ["No", "Yes"], key=field)
        inputs[field] = 1 if choice == "Yes" else 0

    st.subheader("🔢 Numeric Inputs")
    for field in numeric_fields:
        value = st.text_input(f"{field}:", key=field)
        inputs[field] = value

    submitted = st.form_submit_button("🔍 Predict Diabetes")

# Run prediction on submit
if submitted:
    try:
        # Convert inputs to float list
        input_values = []
        for field in yes_no_fields:
            input_values.append(inputs[field])  # already 0 or 1
        for field in numeric_fields:
            input_values.append(float(inputs[field]))  # convert from string

        # Create input DataFrame
        all_fields = yes_no_fields + numeric_fields
        input_df = pd.DataFrame([input_values], columns=all_fields)

        # Scale the required numeric fields
        num_cols = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Display output
        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.success("Prediction: **Diabetes**")
        else:
            st.success("Prediction: **No Diabetes**")

        if proba is not None:
            st.info(f"Probability of having diabetes: **{proba:.2%}**")

    except ValueError:
        st.error("❌ Please enter valid numeric values in all fields.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
