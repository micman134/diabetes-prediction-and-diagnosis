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
✅ Use dropdowns for Yes/No values  
✍️ Enter numbers (e.g., BMI, Age) in textboxes
""")

# Yes/No fields
yes_no_fields = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk'
]

# Numeric-only fields
numeric_fields = [
    'BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Sex',
    'Age', 'Education', 'Income'
]

# Create input form
inputs = {}

with st.form("diabetes_form"):
    st.subheader("Yes/No Inputs (Select 1 for Yes, 0 for No)")
    for field in yes_no_fields:
        inputs[field] = st.selectbox(f"{field}:", ["0", "1"], key=field)

    st.subheader("Numeric Inputs")
    for field in numeric_fields:
        inputs[field] = st.text_input(f"{field}:", key=field)

    submit = st.form_submit_button("Predict Diabetes")

# Handle prediction
if submit:
    try:
        # Combine all inputs into a list
        input_values = [float(inputs[f]) for f in yes_no_fields + numeric_fields]

        # Create DataFrame
        input_df = pd.DataFrame([input_values], columns=yes_no_fields + numeric_fields)

        # Scale numeric columns
        num_cols = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        # Display results
        st.subheader("📊 Prediction Result")
        st.success("Prediction: **Diabetes**" if prediction == 1 else "Prediction: **No Diabetes**")
        if probability is not None:
            st.info(f"Probability of Diabetes: **{probability:.2%}**")

    except ValueError:
        st.error("🚫 Please make sure all fields are filled correctly with numeric values.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
