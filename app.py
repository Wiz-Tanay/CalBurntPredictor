import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="Calorie Burnt Predictor", layout="wide")

# ================= LOAD FILES =================
lr = pickle.load(open("lr.pkl","rb"))
dt = pickle.load(open("dt.pkl","rb"))
rf = pickle.load(open("rf.pkl","rb"))
xgb = pickle.load(open("xgb.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost (Best)": xgb
}

# ================= SIDEBAR =================
menu = st.sidebar.radio("Navigation", ["Predict", "Model Explorer", "Comparison"])

st.title("🔥 Calorie Burn Prediction System")

# ================= INPUT SECTION =================
def user_input():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100)
        height = st.number_input("Height (cm)", 100, 220)
        weight = st.number_input("Weight (kg)", 30, 150)
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        duration = st.number_input("Duration (min)", 1, 300)
        heart_rate = st.number_input("Heart Rate", 60, 200)
        body_temp = st.number_input("Body Temperature", 35.0, 42.0)

    gender = 0 if gender == "male" else 1
    bmi = weight / ((height/100)**2)

    data = np.array([[age, height, weight, duration, heart_rate, body_temp, gender, bmi]])
    return data

# ================= PREDICT PAGE =================
if menu == "Predict":

    st.header("Enter User Details")

    data = user_input()

    if st.button("Predict Calories"):

        data_scaled = scaler.transform(data)

        best_pred = xgb.predict(data_scaled)[0]

        st.subheader("🔥 Best Model Prediction")
        st.metric("XGBoost", round(best_pred,2))

        st.subheader("📊 Other Models")

        for name, model in models.items():
            pred = model.predict(data_scaled)[0]
            st.write(f"{name}: {round(pred,2)}")

# ================= MODEL EXPLORER =================
elif menu == "Model Explorer":

    st.header("🔍 Model Explorer")

    model_choice = st.selectbox("Select Model", list(models.keys()))

    if model_choice == "Linear Regression":
        st.image("lr_plot.png")
        st.image("lr_feature.png")

    elif model_choice == "Decision Tree":
        st.image("dt_plot.png")
        st.image("dt_feature.png")

    elif model_choice == "Random Forest":
        st.image("rf_plot.png")
        st.image("rf_feature.png")

    elif model_choice == "XGBoost (Best)":
        st.image("xgb_plot.png")
        st.image("xgb_feature.png")

    st.write("""
    These visualizations show:
    - Actual vs Predicted values
    - Feature importance of the model
    """)

# ================= COMPARISON =================
elif menu == "Comparison":

    st.header("📊 Model Comparison Dashboard")

    st.image("comparison.png")

    st.subheader("Performance Table")

    df = pd.read_csv("results.csv")
    st.dataframe(df)

    st.write("""
    XGBoost performs best due to its ability to handle non-linear relationships 
    and minimize prediction error effectively.
    """)