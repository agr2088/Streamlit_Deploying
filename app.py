import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------------
# 1. Load the artefacts you created with train_and_save.py
# ------------------------------------------------------------------
model   = joblib.load("logistic_model.pkl")   # LogisticRegression
scaler  = joblib.load("scaler.pkl")          # StandardScaler
# The imputer was saved in the training script; if it is missing we skip it
try:
    imputer = joblib.load("imputer.pkl")    # SimpleImputer (median)
    pipeline = make_pipeline(imputer, scaler, model)
except Exception:
    pipeline = make_pipeline(scaler, model)   # fallback – no imputation

# ------------------------------------------------------------------
# 2️. Feature definition (label,  min, max, step, default, dtype)
# ------------------------------------------------------------------
FEATURES = [
    ("Pregnancies",            0,  20, 1,   2,   int),
    ("Glucose",                0, 200, 1, 120,   int),
    ("BloodPressure",          0, 150, 1,  70,   int),
    ("SkinThickness",          0, 100, 1,  20,   int),
    ("Insulin",                0,1000, 1,  80,   int),
    ("BMI",                 0.0,  80, 0.1, 30.0, float),
    ("DiabetesPedigreeFunction",0.0,2.5,0.01,0.5, float),
    ("Age",                    1, 120, 1,  33,   int),
]

def get_user_df() -> pd.DataFrame:
    """Create the eight sidebar number‑inputs **once** and return a 1‑row DataFrame."""
    vals = {}
    for name, lo, hi, step, default, typ in FEATURES:
        if typ is int:
            vals[name] = st.sidebar.number_input(
                label=name,
                min_value=int(lo),
                max_value=int(hi),
                value=int(default),
                step=int(step),
                key=name,                 # unique key → prevents duplicate‑ID errors
                format="%d",
            )
        else:  # float
            vals[name] = st.sidebar.number_input(
                label=name,
                min_value=float(lo),
                max_value=float(hi),
                value=float(default),
                step=float(step),
                key=name,
                format="%.2f",
            )
    return pd.DataFrame([vals])

# ------------------------------------------------------------------
# 3. Streamlit page layout

st.set_page_config(page_title="Diabetes predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction – Logistic Regression")
st.write(
    "Enter the eight clinical measurements in the left sidebar, "
    "press **Predict**, and see the probability of diabetes."
)

# Build the input DataFrame **once**
user_df = get_user_df()

if st.button("🔮 Predict"):
    prob = pipeline.predict_proba(user_df)[0, 1]      # prob. of class 1 (diabetes)
    pred = int(prob >= 0.5)                         # binary decision

    col1, col2 = st.columns(2)
    col1.metric("Probability of Diabetes", f"{prob*100:.1f}%")
    col2.metric("Predicted class (0 = No, 1 = Yes)", pred)

    if pred:
        st.error("⚠️ High risk – the model predicts diabetes.")
    else:
        st.success("✅ Low risk – the model predicts no diabetes.")

# ------------------------------------------------------------------
# 4️. Show the data that was fed to the model (optional)
# ------------------------------------------------------------------
with st.expander("🔎 Input data (what the model sees)"):
    st.dataframe(user_df)
