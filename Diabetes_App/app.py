import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------
# Load model & scaler (safe path)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "logistic_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# App Title
# -------------------------------
st.title("Diabetes Prediction using Logistic Regression")
st.write("Enter patient details to predict diabetes outcome")

# -------------------------------
# Sidebar User Inputs
# -------------------------------
st.sidebar.header("Patient Input Features")

def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)

    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -------------------------------
# Display Input Data
# -------------------------------
st.subheader("User Input Data")
st.write(input_df)

# -------------------------------
# Scaling
# -------------------------------
scaled_input = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# -------------------------------
# Output
# -------------------------------
st.subheader("Prediction Result")

if prediction[0] == 1:
    st.error("🔴 Diabetes Detected")
else:
    st.success("🟢 No Diabetes Detected")

st.subheader("Prediction Probability")
st.write(f"Probability of No Diabetes: {prediction_proba[0][0]:.2f}")
st.write(f"Probability of Diabetes: {prediction_proba[0][1]:.2f}")
