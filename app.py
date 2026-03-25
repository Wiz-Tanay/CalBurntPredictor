import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Calorie Burnt Predictor", 
    page_icon="🏃‍♂️", 
    layout="centered", 
    initial_sidebar_state="collapsed"
)
# Initialize session state to hold live predictions across tabs
if 'latest_predictions' not in st.session_state:
    st.session_state.latest_predictions = None

# Created cache of the models for less load on frequent model refreshes/user inputs 
@st.cache_resource
def load_models():
    lr = pickle.load(open("lr.pkl", "rb"))
    dt = pickle.load(open("dt.pkl", "rb"))
    rf = pickle.load(open("rf.pkl", "rb"))
    xgb = pickle.load(open("xgb.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    models = {
        "Linear Regression": lr,
        "Decision Tree": dt,
        "Random Forest": rf,
        "XGBoost": xgb
    }
    return models, scaler, xgb

models, scaler, best_model = load_models()

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏃‍♂️ Calorie Burn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Machine Learning powered fitness tracking.</p>", unsafe_allow_html=True)
st.divider()
# Centered tabs for better look
tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "🔍 Model Explorer", "⚖️ Comparison"])

# First tab - Calorie Predictor
with tab1:
    st.subheader("Enter Your Workout Details")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**👤 Personal Metrics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age (Years)", min_value=10, max_value=100, value=25, step=1)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0)
        with col2:
            st.markdown("**🏋️‍♀️ Session Metrics**")
            duration = st.number_input("Duration (Minutes)", min_value=1.0, max_value=120.0, value=30.0, step=1.0)
            heart_rate = st.number_input("Avg Heart Rate (BPM)", min_value=60.0, max_value=220.0, value=100.0, step=1.0)
            body_temp = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=37.0, step=0.1)

    gender_encoded = 1 if gender == "Male" else 0
    bmi = weight / ((height / 100) ** 2)
    
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp, bmi]])
    
    st.write("") 

    if st.button("🔥 Calculate Calories Burned", use_container_width=True):
        
        if duration < 5 and heart_rate > 160:
            st.warning("⚠️ That's a very high heart rate for such a short duration. Ensure your inputs are correct.")
        elif body_temp < 36.5 and heart_rate > 120:
            st.warning("⚠️ High heart rate but low body temperature detected. Results may be skewed.")
        else:
            progress_text = "Analyzing physiological data..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(0.5)
            my_bar.empty() 
            
            data_scaled = scaler.transform(input_data)
            
            live_preds = {}
            for name, model in models.items():
                live_preds[name] = round(float(model.predict(data_scaled)[0]), 2)
                
            st.session_state.latest_predictions = live_preds
            pred_rounded = live_preds["XGBoost"]
            
            st.success("Analysis Complete!")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.metric(label="Calories Burned", value=f"{pred_rounded} kcal")
                st.metric(label="Calculated BMI", value=round(bmi, 1))
                
            with res_col2:
                st.markdown("### Session Feedback")
                if pred_rounded < 0:
                    st.error("⚠️ **Anomaly Detected:** The model predicted a negative calorie burn. Please check if your inputs (like height/weight) are realistic.")
                elif pred_rounded > 1500:
                    st.warning("⚠️ **Unrealistic Burn:** Over 1500 kcal is an extreme amount for a single session. Ensure your duration and heart rate are accurate.")
                elif pred_rounded < 50:
                    st.info("🚶‍♂️ **Light Movement:** Every little bit counts! Great job staying active.")
                elif 50 <= pred_rounded < 150:
                    st.success("🔥 **Solid Burn:** A steady and healthy effort. Keep it up!")
                elif 150 <= pred_rounded < 300:
                    st.warning("💪 **Intense Workout:** You are really pushing your limits! Remember to hydrate.")
                else:
                    st.error("🚀 **Absolute Machine:** Incredible energy expenditure! Outstanding session!")
                    st.balloons() 


# Second tab - Individual Model Visualization (Actual vs Predicted) and Feature Information
with tab2:
    st.subheader("🔍 Model Insights & Visualization")
    st.write("Select a model below to view its actual vs. predicted performance and feature importance.")
    
    model_choice = st.selectbox("Select Model to Inspect", list(models.keys()))
    
    try:
        if model_choice == "Linear Regression":
            st.image("lr_plot.png", caption="Linear Regression: Actual vs Predicted")
            st.image("lr_feature.png", caption="Linear Regression: Feature Importance")
        elif model_choice == "Decision Tree":
            st.image("dt_plot.png", caption="Decision Tree: Actual vs Predicted")
            st.image("dt_feature.png", caption="Decision Tree: Feature Importance")
        elif model_choice == "Random Forest":
            st.image("rf_plot.png", caption="Random Forest: Actual vs Predicted")
            st.image("rf_feature.png", caption="Random Forest: Feature Importance")
        elif model_choice == "XGBoost":
            st.image("xgb_plot.png", caption="XGBoost: Actual vs Predicted")
            st.image("xgb_feature.png", caption="XGBoost: Feature Importance")
    except FileNotFoundError:
        st.error("⚠️ Visualization images not found. Please ensure plot/feature files are in the root directory.")


# Third tab - Comparison between all the models (live comparison, and metrics table)
with tab3:
    st.subheader("⚡ Live Prediction Comparison")
    
    if st.session_state.latest_predictions:
        st.write("Here is how all four algorithms predicted your specific calorie burn based on your **current inputs**:")
        
        chart_data = pd.DataFrame(
            list(st.session_state.latest_predictions.values()),
            index=list(st.session_state.latest_predictions.keys()),
            columns=["Predicted Calories (kcal)"]
        )
        st.bar_chart(chart_data)
        
        cols = st.columns(4)
        for i, (model_name, prediction_value) in enumerate(st.session_state.latest_predictions.items()):
            cols[i].metric(label=model_name, value=prediction_value)
            
        # Short descriptions for the live comparison
        st.markdown("""
        **Live Prediction Insights:**
        * **Linear Regression:** Often underestimates or overestimates slightly because it assumes a strict straight-line relationship between heart rate, duration, and calories.
        * **Decision Tree:** Can be erratic. It makes decisions in 'steps' which might lead to jumping to a higher or lower calorie bracket abruptly.
        * **Random Forest:** Very stable. It averages the results of hundreds of Decision Trees, meaning your prediction is well-rounded and avoids extremes.
        * **XGBoost:** The most precise. It learns directly from the complex biological interactions in the data to give you the most accurate calorie burn possible for your specific inputs.
        """)
            
    else:
        st.info("👈 Go to the 'Predictor' tab and calculate your calories to see a live comparison here!")

    st.divider()

    st.subheader("🏆 Overall Algorithm Analysis")
    st.write("Below is a breakdown of how each model performed on the testing dataset.")
    
    # Read metrics directly from results.csv generated by the ipynb file
    try:
        metrics_df = pd.read_csv("results.csv")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    except FileNotFoundError:
        st.error("⚠️ 'results.csv' not found. Please ensure your Jupyter notebook has generated this file and it is placed in the same folder as app.py.")
    
    # Adding the explanation for the high R2 score
    st.info("""
    **🤔 Why is the accuracy so high?** The dataset used for training was likely generated using a deterministic biological formula (such as the Harris-Benedict equation or fitness tracker logic) that calculates calories directly from Duration, Heart Rate, Weight, and Age. Because these machine learning models (especially XGBoost) are so powerful, they essentially 'reverse-engineered' the exact mathematical formula used to create the data. This results in near-perfect, artificially high accuracy that is rarely seen in noisy, real-world data!
    """)

    try:
        st.image("comparison.png", use_container_width=True)
    except FileNotFoundError:
        pass
