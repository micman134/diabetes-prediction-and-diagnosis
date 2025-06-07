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

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🧪 Diabetes Prediction App")

st.markdown("""
Enter your health indicator values below.  
✅ Use dropdowns for Yes/No questions  
✍️ Enter numeric values (e.g., BMI, Age) in textboxes
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

# Input form
inputs = {}

with st.form("diabetes_form"):
    st.subheader("Yes/No Inputs")
    for field in yes_no_fields:
        user_choice = st.selectbox(f"{field}:", ["No", "Yes"], key=field)
        inputs[field] = 1 if user_choice == "Yes" else 0

    st.subheader("Numeric Inputs")
    for field in numeric_fields:
        value = st.text_input(f"{field}:", key=field)
        inputs[field] = value

    submit = st.form_submit_button("Predict Diabetes")

# Prediction logic
if submit:
    try:
        # Convert inputs to float
        input_values = []
        for field in yes_no_fields:
            input_values.append(inputs[field])
        for field in numeric_fields:
            input_values.append(float(inputs[field]))

        # DataFrame
        all_fields = yes_no_fields + numeric_fields
        input_df = pd.DataFrame([input_values], columns=all_fields)

        # Scale numeric columns
        num_cols = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Output
        st.subheader("📊 Prediction Result")
        st.success("Prediction: **Diabetes**" if prediction == 1 else "Prediction: **No Diabetes**")
        if probability is not None:
            st.info(f"Probability of Diabetes: **{probability:.2%}**")

    except ValueError:
        st.error("🚫 Please enter valid numeric values for all fields.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
