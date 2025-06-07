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
Enter the required health indicator values below.  
All inputs should be **numeric** (e.g., 0 or 1 for Yes/No, or numerical values like BMI).
""")

# All inputs via text input
inputs = {}
feature_list = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
    'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

# Input section
with st.form("diabetes_form"):
    for feature in feature_list:
        inputs[feature] = st.text_input(f"{feature}:", key=feature)

    submit = st.form_submit_button("Predict Diabetes")

# On form submission
if submit:
    try:
        # Convert input strings to float values
        input_values = [float(inputs[feature]) for feature in feature_list]

        # Prepare DataFrame
        input_df = pd.DataFrame([input_values], columns=feature_list)

        # Scale numerical columns
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
