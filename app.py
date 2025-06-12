import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go
import time
from typing import Tuple
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime
import uuid
import hashlib
import requests
import json
from google.cloud.firestore_v1 import FieldFilter  # Add this import

# Set page config with improved metadata
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# PWA Installation Prompt
st.markdown("""
<script>
// Track whether the prompt has been shown
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
  // Prevent the mini-infobar from appearing on mobile
  e.preventDefault();
  // Stash the event so it can be triggered later
  deferredPrompt = e;
  // Show the install button
  document.getElementById('installContainer').style.display = 'block';
});

async function installApp() {
  if (deferredPrompt) {
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      document.getElementById('installContainer').style.display = 'none';
    }
    deferredPrompt = null;
  }
}
</script>

<div id="installContainer" style="display: none; position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
  <button onclick="installApp()" style="
    background-color: #2b5876;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 16px;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  ">
    Install App
  </button>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

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
    .history-item {
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .history-item:hover {
        background-color: #e2e6ea;
    }
    .history-date {
        font-size: 0.8em;
        color: #666;
    }
    .history-risk {
        font-weight: bold;
    }
    .auth-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#2b5876">
""", unsafe_allow_html=True)
# Initialize Firebase
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            # Directly use the private key from secrets (it should maintain formatting)
            cred = credentials.Certificate({
                "type": st.secrets["firebase_creds"]["type"],
                "project_id": st.secrets["firebase_creds"]["project_id"],
                "private_key_id": st.secrets["firebase_creds"]["private_key_id"],
                "private_key": st.secrets["firebase_creds"]["private_key"],
                "client_email": st.secrets["firebase_creds"]["client_email"],
                "client_id": st.secrets["firebase_creds"]["client_id"],
                "auth_uri": st.secrets["firebase_creds"]["auth_uri"],
                "token_uri": st.secrets["firebase_creds"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase_creds"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase_creds"]["client_x509_cert_url"],
                "universe_domain": st.secrets["firebase_creds"]["universe_domain"]
            })
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {str(e)}")
        st.error("Please check your Firebase credentials configuration")
        return None

# Initialize Firebase connection
db = initialize_firebase()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(email, password):
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return user
    except Exception as e:
        st.error(f"Error creating user: {str(e)}")
        return None

def authenticate_user(email, password):
    if not email or not password:
        st.error("Please enter both email and password")
        return None
    
    try:
        # Firebase REST API key from your config
        api_key = st.secrets["firebase_creds"]["api_key"]
        
        # Sign in with email and password using Firebase REST API
        auth_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        auth_data = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        
        response = requests.post(auth_url, json=auth_data)
        result = response.json()
        
        if 'error' in result:
            st.error(f"Authentication failed: {result['error']['message']}")
            return None
        
        # If successful, get the user details via Admin SDK
        user = auth.get_user_by_email(email)
        return user
        
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return None

# Session state management
if 'user' not in st.session_state:
    st.session_state.user = None
if 'page' not in st.session_state:
    st.session_state.page = "login"
if 'history' not in st.session_state:
    st.session_state.history = []

# Store prediction in Firestore
def store_prediction(prediction_data):
    if db and st.session_state.user:
        try:
            # Add timestamp if not exists
            if 'timestamp' not in prediction_data:
                prediction_data['timestamp'] = datetime.now().isoformat()
            
            # Add user ID to prediction data
            prediction_data['user_id'] = st.session_state.user.uid
            
            # Add a new document with a generated ID
            doc_ref = db.collection("predictions").add(document_data=prediction_data)
            return True
        except Exception as e:
            st.error(f"Failed to store prediction: {str(e)}")
            return False
    return False

# Get user prediction history
def get_user_history():
    if db and st.session_state.user:
        try:
            
            predictions_ref = db.collection("predictions").where(filter=FieldFilter("user_id", "==", st.session_state.user.uid))
            docs = predictions_ref.stream()
            
            history = []
            for doc in docs:
                pred_data = doc.to_dict()
                pred_data['id'] = doc.id
                history.append(pred_data)
            
            # Sort by timestamp descending
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            return history
        except Exception as e:
            st.error(f"Failed to fetch history: {str(e)}")
            return []
    return []

# Cache model and artifacts loading for efficiency
@st.cache_data(show_spinner=False)
def load_artifacts():
    try:
        artifacts = joblib.load('diabetes_artifacts_compressed.joblib')
        
        if hasattr(artifacts['selected_features'], 'tolist'):
            artifacts['selected_features'] = artifacts['selected_features'].tolist()
        if hasattr(artifacts['classes'], 'tolist'):
            artifacts['classes'] = artifacts['classes'].tolist()
        
        return artifacts
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None

# Helper functions
def yes_no_selectbox(label: str, help_text: str = "", key: str = None) -> int:
    choice = st.selectbox(
        label,
        ['No', 'Yes'],
        help=help_text,
        key=key
    )
    return 1 if choice == 'Yes' else 0

def sex_selectbox(label: str, key: str = None) -> int:
    choice = st.selectbox(
        label,
        ['Female', 'Male'],
        help="Select your biological sex",
        key=key
    )
    return 1 if choice == 'Male' else 0

def validate_inputs(bmi: float, age: int, ment_hlth: int, phys_hlth: int) -> Tuple[bool, str]:
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
    classes = {
        0: ("Low Risk", "risk-low"),
        1: ("Medium Risk", "risk-medium"),
        2: ("High Risk", "risk-high")
    }
    text, style = classes.get(prediction, ("Unknown", ""))
    return f'<span class="{style}">{text}</span>'

# Authentication Pages
def login_page():
    with st.sidebar:
        st.title("🔐 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    user = authenticate_user(email, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.history = get_user_history()
                        st.session_state.page = "app"
                        st.rerun()
        
        with col2:
            if st.button("Create Account"):
                st.session_state.page = "signup"
                st.rerun()
                
def signup_page():
    with st.sidebar:
        st.title("📝 Create Account")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Sign Up"):
                if password != confirm_password:
                    st.error("Passwords don't match!")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        user = create_user(email, password)
                        if user:
                            st.session_state.user = user
                            st.session_state.page = "app"
                            st.success("Account created successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Account creation failed: {str(e)}")
        
        with col2:
            if st.button("Back to Login"):
                st.session_state.page = "login"
                st.rerun()

# Add this new function before the main app flow
def show_landing_page():
    st.title("🩺 Diabetes Risk Predictor")
    st.markdown("""
    ## Welcome to the Diabetes Risk Assessment Tool
    
    This application helps you assess your risk of developing diabetes based on 
    health indicators and lifestyle factors. Please login or create an account 
    from the sidebar to get started.
    """)
    
    with st.expander("📊 How it works"):
        st.markdown("""
        - Answer questions about your health and lifestyle
        - Our advanced machine learning model analyzes your risk factors
        - Get a personalized risk assessment with actionable recommendations
        - Track your risk over time (when logged in)
        """)
    
    with st.expander("🔬 About the Model"):
        st.markdown("""
        - **Algorithm**: LightGBM (Gradient Boosting)
        - **Accuracy**: 84.59%
        - **Training Data**: CDC Behavioral Risk Factor Surveillance System
        - **Features**: 17 key health indicators including BMI, age, and lifestyle factors
        """)
    
    with st.expander("👨‍⚕️ Medical Disclaimer"):
        st.markdown("""
        **Important:** This tool does not provide medical advice and is not a substitute 
        for professional medical evaluation, diagnosis, or treatment. Always seek the 
        advice of your physician with any questions you may have regarding a medical condition.
        """)
    
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <small>Please login from the sidebar to access the full assessment tool</small>
    </div>
    """, unsafe_allow_html=True)

# Main App Pages
def main_app():
    # Sidebar navigation and user info
    with st.sidebar:
        if st.session_state.user:
            st.markdown(f"### 👤 {st.session_state.user.email}")
            if st.button("Logout"):
                st.session_state.user = None
                st.session_state.page = "login"
                st.session_state.history = []
                st.rerun()
        
        st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=80)
        menu_option = st.radio("Navigation Menu",
                             ["📊 Prediction", 
                              "📋 History",
                              "🔍 Model Analysis", 
                              "ℹ️ About"],
                             index=0,
                             label_visibility="visible")
    
    # Prediction History Page
    if menu_option == "📋 History":
        st.title("📋 Prediction History")
        
        if not st.session_state.user:
            st.warning("Please login to view your prediction history")
            return
        
        if not st.session_state.history:
            st.info("No prediction history found")
            return
        
        st.markdown(f"### Your Recent Predictions ({len(st.session_state.history)})")
        
        for pred in st.session_state.history:
            risk_class = pred.get('risk_class', 'Unknown')
            timestamp = datetime.fromisoformat(pred['timestamp']).strftime("%B %d, %Y %H:%M")
            
            with st.expander(f"Prediction on {timestamp}"):
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-date">{timestamp}</div>
                    <div class="history-risk">Risk Level: <span class="risk-{risk_class.lower().replace(' ', '-')}">{risk_class}</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                if 'probabilities' in pred:
                    class_names = {
                        0: "No diabetes",
                        1: "Pre-diabetes", 
                        2: "Diabetes"
                    }
                    
                    fig = go.Figure(go.Bar(
                        x=[p * 100 for p in pred['probabilities']],
                        y=[class_names[i] for i in range(len(class_names))],
                        orientation='h',
                        text=[f"{p*100:.1f}%" for p in pred['probabilities']],
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
                
                if 'user_inputs' in pred:
                    st.markdown("**Input Parameters:**")
                    inputs = pred['user_inputs']
                    cols = st.columns(2)
                    for i, (key, value) in enumerate(inputs.items()):
                        cols[i % 2].write(f"**{key}:** {value}")
    
    # Prediction Page
    elif menu_option == "📊 Prediction":
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
                        #model_type = "LightGBM"
                        #st.markdown(f"""
                        ### Model Information
                        #**Algorithm:** {model_type}  
                        #**Training Accuracy:** 84.59%  
                        #**Training Weighted F1-Score:** 82.09%
                        #""")
                        
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
                        
                        # Store prediction in Firebase
                        prediction_data = {
                            "user_inputs": all_features,
                            "prediction": int(prediction),
                            "probabilities": [float(p) for p in probabilities],
                            "risk_class": get_risk_class(prediction).split('>')[1].split('<')[0],
                            "model_version": "1.0",
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        if store_prediction(prediction_data):
                            st.toast("Prediction saved successfully!", icon="✅")
                            # Refresh history
                            st.session_state.history = get_user_history()
                        else:
                            st.warning("Prediction completed but couldn't save to database")
                        
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {str(e)}")
                
                st.markdown("""
                <div class="disclaimer">
                <strong>Disclaimer:</strong> This tool is for informational purposes only and is not a substitute 
                for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician 
                or other qualified health provider with any questions you may have regarding a medical condition.
                </div>
                """, unsafe_allow_html=True)
    
    # Model Analysis Page
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
            - Features: 17 (selected via SelectKBest)
            
            **Training Performance:**
            - Best model selected from: [XGBoost, LightGBM, RandomForest, LogisticRegression, DecisionTree]
            - LightGBM achieved highest weighted F1-score (82.09%)
            
            **Data Preprocessing:**
            - Standard Scaling: Yes
            - Feature Selection: SelectKBest (k=17)
            - Class balancing: SMOTE
            
            **Training Data:**
            - Source: CDC Behavioral Risk Factor Surveillance System (BRFSS)
            - Samples: 253,680 (original), balanced via SMOTE
            - Year: 2015
            """)
    
    # About Page
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
            - **Data Scientists**: [], []
            - **Medical Advisors**: Dr. [], Dr. []
            - **Developers**: [], []
            
            **Version**: 1.2.0
            **Last Updated**: June 2024
            """)

# Main App Flow
if st.session_state.page == "login":
    login_page()
    show_landing_page()  # Add this function call
    
elif st.session_state.page == "signup":
    signup_page()
    show_landing_page()
elif st.session_state.page == "app":
    main_app()

