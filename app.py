import streamlit as st
import numpy as np
import joblib
import time

st.set_page_config(
    page_title="Student ML App",
    page_icon="📊",
    layout="centered"
)

import numpy as np
import joblib
import time

@st.cache_resource
def load_model():
    return joblib.load("student_performance_model.pkl")

model = load_model()

st.sidebar.markdown("## 📂 Choose a Section")

if 'page' not in st.session_state:
    st.session_state.page = 'predictor'

nav_items = {
    "📊 Predict Student Score": "predictor",
    "📘 About This Project": "details",
}

for label, page in nav_items.items():
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.page = page

if st.session_state.page == "predictor":
    st.markdown("<h1 style='text-align: center;'>📊 Student Score Predictor</h1>", unsafe_allow_html=True)
    st.write("Fill in the details below to get an estimate of a student's performance index.")

    st.subheader("Student Information")

    hours_studied = st.number_input("Hours Studied (per day)", min_value=1.0, max_value=12.0, value=6.0, step=0.5)
    previous_scores = st.number_input("Previous Scores (%)", min_value=0.0, max_value=100.0, value=70.0, step=0.5)
    extracurricular = st.selectbox("Is the student involved in extracurricular activities?", options=["Yes", "No"])
    extracurricular_val = 1 if extracurricular == "Yes" else 0
    sleep_hours = st.number_input("Sleep Hours (per day)", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
    papers_practiced = st.number_input("Practice Papers Solved (per week)", min_value=0.0, max_value=20.0, value=5.0, step=1.0)

    if st.button("Submit of Predictions"):
        steps = st.empty()

        steps.markdown("**Step 1:** Capturing input data...")
        time.sleep(0.3)

        steps.markdown("**Step 2:** Preparing data for prediction...")
        features = np.array([[hours_studied, previous_scores, extracurricular_val, sleep_hours, papers_practiced]])
        time.sleep(0.3)

        steps.markdown("**Step 3:** Validating inputs...")
        time.sleep(0.3)

        steps.markdown("**Step 4:** Making prediction...")
        prediction = model.predict(features)
        predicted_score = max(0, round(prediction[0], 2))
        time.sleep(0.3)

        st.success("✅ Prediction complete!")

        st.markdown(
            f"<div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center;'>"
            f"<h3 style='color: #1f77b4;'>Estimated Performance Index: {predicted_score}</h3>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("---")
    st.subheader("Model Accuracy")

    col1, col2 = st.columns(2)
    col1.metric(label="Root Mean Squared Error (RMSE)", value="2.27")
    col2.metric(label="R² Score", value="0.99")

elif st.session_state.page == "details":
    st.markdown("<h1>📘 Project Information</h1>", unsafe_allow_html=True)

    st.markdown("#### 👥 Developed By:")
    st.markdown("Vikas Kumar Sahu (G24AIT2086) and Rohan Tiwari (G24AIT2177) 💬")

    st.markdown("#### 🎯 Project Goal")
    st.write(
        "This app helps estimate a student's academic performance using a simple and explainable machine learning model. "
        "By entering information about their study habits and lifestyle, users can get a predicted performance index."
    )

    st.markdown("#### 🧾 Dataset Highlights")
    st.write("The prediction is based on the following inputs:")
    st.markdown("""
    - Hours Studied  
    - Previous Scores  
    - Extracurricular Activity  
    - Sleep Duration  
    - Weekly Practice Paper Count
    """)

    st.markdown("#### ⚙️ Model Summary")
    st.markdown("""
    - Binary encoding was applied for categorical fields (like extracurriculars).  
    - Data was already clean, so scaling wasn't required.  
    - A **Multiple Linear Regression** model from scikit-learn was trained on the data.
    """)

    st.markdown("#### 🔍 How It Works")
    st.markdown("""
    1. Users fill out the input form  
    2. Inputs are formatted and passed to the model  
    3. The model calculates the expected performance index  
    4. The result is displayed on screen
    """)

    st.markdown("#### 📊 Evaluation Results")
    st.markdown("""
    - **RMSE**: ~2.27  
    - **R² Score**: 0.99  
    These values indicate that the model is accurate on the test data.
    """)

    st.markdown("#### 📝 Final Thoughts")
    st.write(
        "This project demonstrates how basic data and a simple regression model can be deployed using Streamlit "
        "to build a functional and user-friendly web app. It is great for educational or prototyping purposes."
    )

    st.info("Note: Predictions are based on sample data and may not reflect real-world performance.")
