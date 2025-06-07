import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model and scaler (cached for performance)
@st.cache_resource
def load_model_scaler():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_scaler()

# Set page layout
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

st.title("🧪 Diabetes Risk Prediction")
st.markdown("""
Predict your risk of diabetes based on health indicators.  
Please fill in the form below carefully.
""")

# Binary yes/no fields with tooltips
binary_fields = {
    "HighBP": "High Blood Pressure (Yes/No)",
    "HighChol": "High Cholesterol (Yes/No)",
    "CholCheck": "Cholesterol checked in past 5 years (Yes/No)",
    "Smoker": "Smoked at least 100 cigarettes (Yes/No)",
    "Stroke": "Ever told you had a stroke (Yes/No)",
    "HeartDiseaseorAttack": "Ever had coronary heart disease or heart attack (Yes/No)",
    "PhysActivity": "Physical activity in past 30 days (Yes/No)",
    "Fruits": "Consume fruits (Yes/No)",
    "Veggies": "Consume vegetables (Yes/No)",
    "HvyAlcoholConsump": "Heavy alcohol consumption (Yes/No)",
    "AnyHealthcare": "Have any healthcare coverage (Yes/No)",
    "NoDocbcCost": "Could not see a doctor due to cost (Yes/No)",
    "DiffWalk": "Serious difficulty walking or climbing stairs (Yes/No)"
}

# Numeric or categorical ordinal fields with range tooltips
numeric_fields = {
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "min": 10.0,
        "max": 60.0,
        "tooltip": "Body Mass Index: Normal ~18.5-24.9; Obese >30"
    },
    "GenHlth": {
        "label": "General Health (1=Excellent, 5=Poor)",
        "min": 1,
        "max": 5,
        "tooltip": "1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor"
    },
    "MentHlth": {
        "label": "Days Mental Health Not Good (0-30 days)",
        "min": 0,
        "max": 30,
        "tooltip": "Number of days mental health was not good in past 30 days"
    },
    "PhysHlth": {
        "label": "Days Physical Health Not Good (0-30 days)",
        "min": 0,
        "max": 30,
        "tooltip": "Number of days physical health was not good in past 30 days"
    },
    "Sex": {
        "label": "Sex",
        "options": ["Female", "Male"],
        "tooltip": "Select your sex"
    },
    "Age": {
        "label": "Age Category (1-13)",
        "min": 1,
        "max": 13,
        "tooltip": "1:18-24, 2:25-29, ..., 13:80 or older (Age groups)"
    },
    "Education": {
        "label": "Education Level (1-6)",
        "min": 1,
        "max": 6,
        "tooltip": "1=Never attended, 2=Grades 1-8, ..., 6=College graduate"
    },
    "Income": {
        "label": "Income Level (1-8)",
        "min": 1,
        "max": 8,
        "tooltip": "1=Less than $10K, ..., 8=$75K or more"
    }
}

inputs = {}

with st.form("diabetes_form"):
    st.subheader("✅ Yes / No Questions")

    # Binary inputs - two per row
    keys = list(binary_fields.keys())
    for i in range(0, len(keys), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(keys):
                key = keys[i + j]
                label = binary_fields[key]
                val = cols[j].selectbox(label, options=["No", "Yes"], index=0, key=key)
                inputs[key] = 1 if val == "Yes" else 0

    st.subheader("🔢 Numeric / Categorical Inputs")

    # Numeric inputs - two per row
    num_keys = list(numeric_fields.keys())
    for i in range(0, len(num_keys), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(num_keys):
                key = num_keys[i + j]
                field = numeric_fields[key]
                label = field["label"]
                tooltip = field.get("tooltip", "")
                min_val = field.get("min", None)
                max_val = field.get("max", None)

                if key == "Sex":
                    # Special dropdown for Sex
                    sex_val = cols[j].selectbox(label, options=field["options"], index=0, help=tooltip, key=key)
                    inputs[key] = 1 if sex_val == "Male" else 0
                else:
                    # Numeric input with validation
                    default_val = min_val if min_val is not None else 0
                    val_str = cols[j].text_input(label, value=str(default_val), help=tooltip, key=key)
                    # Validate input and limit range
                    try:
                        val_float = float(val_str)
                        if min_val is not None and val_float < min_val:
                            val_float = min_val
                        if max_val is not None and val_float > max_val:
                            val_float = max_val
                        inputs[key] = val_float
                    except ValueError:
                        # If invalid input, set None to handle later
                        inputs[key] = None

    submitted = st.form_submit_button("🔍 Predict Diabetes")

if submitted:
    # Check if any numeric inputs are None due to invalid entry
    invalid_inputs = [k for k, v in inputs.items() if v is None]
    if invalid_inputs:
        st.error(f"Please enter valid numeric values for: {', '.join(invalid_inputs)}")
    else:
        # Prepare input for model
        # Order input columns as in training
        input_order = list(binary_fields.keys()) + list(numeric_fields.keys())

        # Build list of inputs in correct order
        input_list = [inputs[col] for col in input_order]

        # Create dataframe
        input_df = pd.DataFrame([input_list], columns=input_order)

        # Scale numeric columns: BMI, MentHlth, PhysHlth
        num_cols_to_scale = ['BMI', 'MentHlth', 'PhysHlth']
        input_df[num_cols_to_scale] = scaler.transform(input_df[num_cols_to_scale])

        # Predict
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("📈 Prediction Result")
        if pred == 1:
            st.error("⚠️ You may be at risk of diabetes.")
        else:
            st.success("✅ You are likely not at risk of diabetes.")

        if prob is not None:
            st.info(f"🔢 Risk Probability: **{prob:.2%}**")
