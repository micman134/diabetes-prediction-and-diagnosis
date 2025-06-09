import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import time
from typing import Tuple

# Set page config with improved metadata
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    .prediction-header {
        color: #2b5876;
    }
    .risk-high {
        color: #d62728;
        font-weight: bold;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #666;
        border-top: 1px solid #eee;
        padding-top: 1em;
        margin-top: 2em;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .feature-importance {
        font-size: 0.9em;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model and artifacts loading for efficiency
@st.cache_data(show_spinner=False)
def load_artifacts():
    try:
        # Load the joblib file containing all artifacts
        artifacts = joblib.load('diabetes_artifacts_compressed.joblib')
        
        # Convert numpy arrays to lists if needed
        if hasattr(artifacts['selected_features'], 'tolist'):
            artifacts['selected_features'] = artifacts['selected_features'].tolist()
        if hasattr(artifacts['classes'], 'tolist'):
            artifacts['classes'] = artifacts['classes'].tolist()
        
        return artifacts
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
    menu_option = st.radio("Navigation Menu",
                         ["📊 Prediction", 
                          "🔍 Model Analysis", 
                          "ℹ️ About"],
                         index=0,
                         label_visibility="visible")

# Helper functions
def yes_no_selectbox(label: str, help_text: str = "", key: str = None) -> int:
    """Create a yes/no selectbox and return 1 for Yes, 0 for No."""
    choice = st.selectbox(
        label,
        ['No', 'Yes'],
        help=help_text,
        key=key
    )
    return 1 if choice == 'Yes' else 0

def sex_selectbox(label: str, key: str = None) -> int:
    """Create a sex selectbox and return 1 for Male, 0 for Female."""
    choice = st.selectbox(
        label,
        ['Female', 'Male'],
        help="Select your biological sex",
        key=key
    )
    return 1 if choice == 'Male' else 0

def validate_inputs(bmi: float, age: int, ment_hlth: int, phys_hlth: int) -> Tuple[bool, str]:
    """Validate user inputs and return status and error message."""
    errors = []
    if bmi < 10 or bmi > 60:
        errors.append("BMI must be between 10 and 60")
    if age < 18 or age > 120:
        errors.append("Age must be between 18 and 120")
    if ment_hlth < 0 or ment_hlth > 30:
        errors.append("Mental health days must be between 0 and 30")
    if phys_hlth < 0 or phys_hlth > 30:
        errors.append("Physical health days must be between 0 and 30")
    
    return (len(errors) == 0, "<br>".join(errors))

def get_prevention_tips(prediction: int) -> str:
    """Return prevention tips based on prediction class."""
    tips = {
        0: [
            "✅ Maintain your healthy lifestyle!",
            "✅ Continue regular check-ups with your doctor.",
            "✅ Keep your BMI in the healthy range (18.5-24.9).",
            "✅ Stay physically active with at least 150 minutes of moderate exercise per week."
        ],
        1: [
            "⚠️ Increase physical activity to at least 150 minutes per week",
            "⚠️ Focus on a balanced diet with plenty of vegetables and whole grains",
            "⚠️ Lose 5-7% of body weight if overweight",
            "⚠️ Reduce intake of sugary foods and beverages",
            "⚠️ Get your blood sugar checked annually"
        ],
        2: [
            "❗ Consult with a healthcare provider immediately",
            "❗ Monitor blood sugar levels regularly",
            "❗ Follow a diabetes management plan",
            "❗ Maintain a consistent meal schedule",
            "❗ Check your feet daily for cuts or sores",
            "❗ Keep up with regular eye exams"
        ]
    }
    return "\n\n".join(tips.get(prediction, ["No specific recommendations available."]))

def get_risk_class(prediction: int) -> str:
    """Return risk classification with appropriate styling."""
    classes = {
        0: ("Low Risk", "risk-low"),
        1: ("Medium Risk", "risk-medium"),
        2: ("High Risk", "risk-high")
    }
    text, style = classes.get(prediction, ("Unknown", ""))
    return f'<span class="{style}">{text}</span>'

# Main content based on menu selection
if menu_option == "📊 Prediction":
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("""
    This tool assesses your risk of diabetes based on health and lifestyle factors.  
    Please fill in the following information to get your personalized assessment.
    """)

    with st.form("diabetes_risk_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            Age = st.number_input(
                "Age (years)*",
                min_value=18,
                max_value=120,
                value=45,
                help="Enter your age in years"
            )
            Sex = sex_selectbox("Sex*")
            
        with col2:
            BMI = st.number_input(
                "BMI*",
                min_value=10.0,
                max_value=60.0,
                value=24.0,
                step=0.1,
                help="Body Mass Index. Healthy range: 18.5 to 24.9"
            )
            GenHlth = st.slider(
                "General Health (1=Excellent to 5=Poor)*",
                1, 5, 3,
                help="Rate your general health from excellent to poor"
            )
        
        st.subheader("Health History")
        col1, col2 = st.columns(2)
        
        with col1:
            HighBP = yes_no_selectbox(
                "High Blood Pressure",
                "Do you have high blood pressure?"
            )
            HighChol = yes_no_selectbox(
                "High Cholesterol",
                "Do you have high cholesterol?"
            )
            CholCheck = yes_no_selectbox(
                "Cholesterol Check in last 5 years",
                "Have you had a cholesterol check in the last 5 years?"
            )
            Stroke = yes_no_selectbox(
                "Had Stroke",
                "Have you ever had a stroke?"
            )
            
        with col2:
            HeartDiseaseorAttack = yes_no_selectbox(
                "Heart Disease or Attack",
                "Have you had a heart disease or heart attack?"
            )
            Smoker = yes_no_selectbox(
                "Smoker",
                "Have you smoked at least 100 cigarettes in your life?"
            )
            DiffWalk = yes_no_selectbox(
                "Difficulty walking",
                "Do you have difficulty walking or climbing stairs?"
            )
        
        st.subheader("Lifestyle Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            PhysActivity = yes_no_selectbox(
                "Physically Active",
                "Physical activity in past 30 days (not counting job)"
            )
            Fruits = yes_no_selectbox(
                "Eat Fruits",
                "Do you consume fruit 1 or more times per day?"
            )
            HvyAlcoholConsump = yes_no_selectbox(
                "Heavy Alcohol Consumption",
                "Adult men: >14 drinks/week. Adult women: >7 drinks/week"
            )
            
        with col2:
            Veggies = yes_no_selectbox(
                "Eat Vegetables",
                "Do you consume vegetables 1 or more times per day?"
            )
            MentHlth = st.number_input(
                "Days of poor mental health (0-30)*",
                0, 30, 0,
                help="Number of days your mental health was not good in past 30 days"
            )
            PhysHlth = st.number_input(
                "Days of poor physical health (0-30)*",
                0, 30, 0,
                help="Number of days your physical health was not good in past 30 days"
            )
        
        st.markdown("<small>* Required fields</small>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Diabetes Risk", use_container_width=True)

    if submitted:
        artifacts = load_artifacts()
        if not artifacts:
            st.error("Model not loaded properly. Please try again later.")
            st.stop()
        
        is_valid, error_msg = validate_inputs(BMI, Age, MentHlth, PhysHlth)
        
        if not is_valid:
            st.error(f"Please correct the following errors:\n{error_msg}")
        else:
            with st.spinner('Calculating your diabetes risk...'):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for percent_complete in range(101):
                    progress_bar.progress(percent_complete)
                    status_text.text(f"Analyzing... {percent_complete}%")
                    time.sleep(0.02)
                
                # Create dictionary of all possible features
                all_features = {
                    'HighBP': HighBP,
                    'HighChol': HighChol,
                    'CholCheck': CholCheck,
                    'BMI': BMI,
                    'Smoker': Smoker,
                    'Stroke': Stroke,
                    'HeartDiseaseorAttack': HeartDiseaseorAttack,
                    'PhysActivity': PhysActivity,
                    'Fruits': Fruits,
                    'Veggies': Veggies,
                    'HvyAlcoholConsump': HvyAlcoholConsump,
                    'GenHlth': GenHlth,
                    'MentHlth': MentHlth,
                    'PhysHlth': PhysHlth,
                    'DiffWalk': DiffWalk,
                    'Sex': Sex,
                    'Age': Age
                }
                
                # Get only the selected features in the correct order
                selected_features = artifacts['selected_features']
                input_values = [all_features[feature] for feature in selected_features]
                
                # Create DataFrame with proper feature names
                input_data = pd.DataFrame(
                    [input_values],
                    columns=selected_features
                )
                
                try:
                    # Display model information
                    model_type = "LightGBM"  # Hardcoded since we know it's LightGBM
                    st.markdown(f"""
                    ### Model Information
                    **Algorithm:** {model_type}  
                    **Training Accuracy:** 84.59%  
                    **Training Weighted F1-Score:** 82.09%
                    """)
                    
                    # 1. Scale the data
                    input_scaled = artifacts['scaler'].transform(input_data)
                    
                    # 2. Make prediction
                    prediction = artifacts['model'].predict(input_scaled)[0]
                    probabilities = artifacts['model'].predict_proba(input_scaled)[0]
                    
                    class_names = {
                        0: "No diabetes",
                        1: "Pre-diabetes", 
                        2: "Diabetes"
                    }
                    
                    st.success("Analysis complete!")
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.markdown(f"""
                    ### 🩺 Prediction Result
                    **Risk Level:** {get_risk_class(prediction)}  
                    **Prediction:** {class_names[prediction]}
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 📊 Prediction Confidence")
                    
                    fig = go.Figure(go.Bar(
                        x=probabilities * 100,
                        y=[class_names[i] for i in range(len(class_names))],
                        orientation='h',
                        text=[f"{p*100:.1f}%" for p in probabilities],
                        textposition='auto',
                        marker_color=['#2ca02c', '#ff7f0e', '#d62728']
                    ))
                    
                    fig.update_layout(
                        xaxis_title="Probability (%)",
                        yaxis_title="",
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=0, r=0, t=30, b=30),
                        height=300,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("### 💡 Recommended Actions")
                    st.markdown(get_prevention_tips(prediction))
                    
                    st.markdown("### 🔍 Potential Risk Factors")
                    risk_factors = []
                    if BMI >= 25:
                        risk_factors.append(f"BMI of {BMI} (overweight)")
                    if HighBP:
                        risk_factors.append("High blood pressure")
                    if HighChol:
                        risk_factors.append("High cholesterol")
                    if not PhysActivity:
                        risk_factors.append("Physical inactivity")
                    if Smoker:
                        risk_factors.append("Smoking")
                    
                    if risk_factors:
                        st.markdown("The following factors may be contributing to your risk:")
                        for factor in risk_factors:
                            st.markdown(f"- {factor}")
                    else:
                        st.markdown("No significant risk factors identified from your inputs.")
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
            
            st.markdown("""
            <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute 
            for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
            or other qualified health provider with any questions you may have regarding a medical condition.
            </div>
            """, unsafe_allow_html=True)

elif menu_option == "🔍 Model Analysis":
    st.title("🔍 Model Analysis")
    artifacts = load_artifacts()
    
    if not artifacts:
        st.error("Model artifacts not loaded. Please check the model file.")
        st.stop()
    
    st.markdown("""
    ### Understanding the Diabetes Risk Prediction Model
    
    This section provides insights into how the prediction model works and its performance characteristics.
    """)
    
    with st.expander("📈 Model Performance Metrics"):
        st.markdown("""
        **Model Evaluation Results (LightGBM):**
        - Accuracy: 84.59%
        - Weighted F1: 82.09%
        - ROC AUC (OvR): 76.30%
        
        **Detailed Classification Report:**
        """)
        
        # Create a DataFrame for the classification report
        report_data = {
            'Class': ['0.0 (No Diabetes)', '1.0 (Pre-diabetes)', '2.0 (Diabetes)', 'Macro Avg', 'Weighted Avg'],
            'Precision': [0.88, 0.00, 0.50, 0.46, 0.81],
            'Recall': [0.96, 0.00, 0.28, 0.41, 0.85],
            'F1-Score': [0.91, 0.00, 0.36, 0.43, 0.82],
            'Support': [64111, 1389, 10604, 76104, 76104]
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df.style.format({
            'Precision': '{:.2f}',
            'Recall': '{:.2f}',
            'F1-Score': '{:.2f}',
            'Support': '{:,}'
        }))
    
    with st.expander("⚖️ Feature Importance"):
        st.markdown("""
        The following features have the most significant impact on the prediction:
        """)
        
        if hasattr(artifacts['model'], 'feature_importances_'):
            features = artifacts['selected_features']
            importances = artifacts['model'].feature_importances_
            
            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            
            for idx in sorted_idx:
                st.markdown(f"""
                <div class="feature-importance">
                <strong>{features[idx]}:</strong> 
                <progress value="{importances[idx]}" max="{importances.max()}" style="width:100%; height:10px;"></progress>
                {importances[idx]:.4f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Feature importance data not available for this model")
    
    with st.expander("🛠️ Technical Details"):
        st.markdown("""
        **Model Architecture:**
        - Algorithm: LightGBM
        - Objective: multiclass
        - Classes: [0.0, 1.0, 2.0]
        - Features: 15 (selected via SelectKBest)
        
        **Training Performance:**
        - Best model selected from: [XGBoost, LightGBM, RandomForest, LogisticRegression, DecisionTree]
        - LightGBM achieved highest weighted F1-score (82.09%)
        
        **Data Preprocessing:**
        - Standard Scaling: Yes
        - Feature Selection: SelectKBest (k=15)
        - Class balancing: SMOTE
        
        **Training Data:**
        - Source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
        - Samples: 253,680 (original), balanced via SMOTE
        - Year: 2015
        """)

elif menu_option == "ℹ️ About":
    st.title("ℹ️ About This Tool")
    st.markdown("""
    ### Diabetes Risk Predictor
    
    This application helps assess an individual's risk of developing diabetes based on 
    health indicators and lifestyle factors.
    """)
    
    with st.expander("📚 Purpose"):
        st.markdown("""
        - Provide early risk assessment for diabetes
        - Increase awareness of diabetes risk factors
        - Encourage preventive healthcare measures
        - Support clinical decision-making (but not replace it)
        """)
    
    with st.expander("👨‍⚕️ Medical Disclaimer"):
        st.markdown("""
        **Important:** This tool does not provide medical advice and is not a substitute 
        for professional medical evaluation, diagnosis, or treatment. Always seek the 
        advice of your physician or other qualified health provider with any questions 
        you may have regarding a medical condition.
        
        The predictions are based on statistical models and may not be accurate for all 
        individuals. Many factors beyond those included in this tool can affect diabetes risk.
        """)
    
    with st.expander("🛠️ Development Team"):
        st.markdown("""
        - **Data Scientists**: [Your Name], [Colleague Name]
        - **Medical Advisors**: Dr. [Name], Dr. [Name]
        - **Developers**: [Your Name], [Team Member]
        
        **Version**: 1.2.0
        **Last Updated**: June 2024
        """)
    
    with st.expander("📧 Contact Us"):
        st.markdown("""
        For questions or feedback about this tool:
        
        Email: [healthtools@example.com](mailto:healthtools@example.com)  
        Phone: (555) 123-4567  
        Address: 123 Health Street, Suite 100, Anytown, ST 12345
        """)
