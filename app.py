import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="Predictive Analytics Dashboard",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for a sleek, dark, minimalist aesthetic
st.markdown("""
    <style>
    /* Main background and text alignment */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Header Container styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 1.1rem;
    }

    /* Style form containers */
    div[data-testid="stForm"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Target the Submit Button specifically */
    div.stButton > button {
        background-color: #6366F1 !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    div.stButton > button:hover {
        background-color: #4F46E5 !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.4) !important;
    }

    /* Custom result cards */
    .result-box {
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .positive {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10B981;
        color: #10B981;
    }
    .negative {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #EF4444;
        color: #EF4444;
    }
    </style>
""", unsafe_allowed_value=True)

# 2. Header Section
st.markdown("""
    <div class="main-header">
        <h1>GaussianNB Intelligence Engine</h1>
        <p>Input target demographics to compute system classification</p>
    </div>
""", unsafe_allowed_value=True)

# 3. Load the Model Safely
@st.cache_resource
def load_model():
    # Make sure your uploaded pickle file is named 'model.pkl' in the root directory
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'model.pkl' file not found. Please ensure your pickle file is uploaded to the root repository.")
    st.stop()

# 4. Interactive Input Form
with st.form(key="prediction_form"):
    st.subheader("System Parameters")
    
    # Feature 1: Gender (Encoded mapping matches training logic)
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    gender_encoded = 1 if gender_input == "Male" else 0
    
    # Feature 2: Age
    age = st.slider("Age", min_value=18, max_value=100, value=35, step=1)
    
    # Feature 3: Estimated Salary
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0, max_value=500000, value=50000, step=1000)
    
    # Submit row
    submit_button = st.form_submit_button(label="Execute Classification")

# 5. Model Inference & Layout Response
if submit_button:
    # Prepare features matching the ['Gender', 'Age', 'EstimatedSalary'] sequence
    features = np.array([[gender_encoded, age, salary]])
    
    # Generate prediction
    prediction = model.predict(features)[0]
    
    st.markdown("---")
    st.subheader("Classification Outcome")
    
    # Dynamic styling output card based on prediction result
    if prediction == 1:
        st.markdown("""
            <div class="result-box positive">
                Target Identified: Action Affirmative (Class 1)
            </div>
        """, unsafe_allowed_value=True)
    else:
        st.markdown("""
            <div class="result-box negative">
                Target Ignored: Action Negative (Class 0)
            </div>
        """, unsafe_allowed_value=True)
