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

# Set wide layout
st.set_page_config(page_title="Diabetes Prediction", layout="wide")
st.title("🧪 Diabetes Prediction App")
st.markdown("""
Enter your health details to predict your risk of diabetes.  
✅ Select "Yes" or "No" for lifestyle/medical questions  
✍️ Enter numeric values like BMI, Age, etc.
""")

# Define input fields
yes_no_fields = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

numeric_fields = [
    'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Sex',
    'Age', 'Education', 'Income'
]

inputs = {}

# Input form
with st.form("diabetes_form"):
    st.subheader("🟩 Yes/No Questions")
    for field in yes_no_fields:
        inputs[field] = 1 if st.selectbox(f"{field}:", ["No", "Yes"], key=field) == "Yes" else 0

    st.subheader("🔢 Numeric Inputs")

    for i in range(0, len(numeric_fields), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(numeric_fields):
                field = numeric_fields[i + j]
                inputs[field] = cols[j].text_input(f"{field}:", key=field)

    submitted = st.form_submit_button("🔍 Predict Diabetes")

# Prediction logic
if submitted:
    try:
        input_values = []
        for field in yes_no_fields:
            input_values.append(inputs[field])
        for field in numeric_fields:
            input_values.append(float(inputs[field]))

        # Create DataFrame
        all_fields = yes_no_fields + numeric_fields
        input_df = pd.DataFrame([input_values], columns=all_fields)

        # Scale necessary fields
        num_cols = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Display results
        st.subheader("📊 Prediction Result")
        st.success("Prediction: **Diabetes**" if prediction == 1 else "Prediction: **No Diabetes**")
        if proba is not None:
            st.info(f"Probability of having diabetes: **{proba:.2%}**")

    except ValueError:
        st.error("❌ Please enter valid numeric values in all fields.")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred: {e}")
