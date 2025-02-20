import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Diabetes Classifier", page_icon="🌟", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the model
diabetes_model_path = r"diabetes_model.sav"
data_path = r"diabetes.csv"

try:
    with open(diabetes_model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# App Title
st.title("🔬 Diabetes Prediction using Machine Learning")
st.write("Fill in the details below to check your diabetes prediction.")

# Load dataset
data = pd.read_csv(data_path)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# Input fields
st.subheader("📝 Enter Your Details")
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, step=1)
    skin_thickness = st.number_input("🩹 Skin Thickness (mm)", min_value=0)
    diabetes_pedigree_function = st.number_input("📊 Diabetes Pedigree Function", min_value=0.0)

with col2:
    glucose = st.number_input("🍬 Glucose Level", min_value=0)
    insulin = st.number_input("💉 Insulin Level", min_value=0)
    age = st.number_input("🎂 Age", min_value=21, max_value=81, step=1)

with col3:
    blood_pressure = st.number_input("❤️ Blood Pressure", min_value=0)
    bmi = st.number_input("⚖️ BMI", min_value=0.0)

st.markdown("---")

# Prediction Button
if st.button("🔍 Predict"):
    try:
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        prediction = diabetes_model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("🚨 You have Diabetes. Please consult a doctor.")
        else:
            st.success("✅ You don't have Diabetes. Maintain a healthy lifestyle!")
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")

# Clear Button
if st.button("🗑️ Clear Inputs"):
    st.experimental_rerun()

st.markdown("---")

# Model Accuracy Display
if st.button("📊 Show Model Accuracy"):
    test_data = pd.read_csv(data_path)
    X_test = test_data.drop(columns=['Outcome'])
    y_test = test_data['Outcome']
    y_pred = diabetes_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### 🔢 Model Accuracy")
    st.progress(float(accuracy))
    st.success(f"✅ Accuracy: {accuracy:.2%}")
