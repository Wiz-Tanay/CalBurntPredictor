🔥 Calorie Burnt Predictor

A Machine Learning-based web application that predicts calories burned using user-specific physiological and exercise-related inputs.

🚀 Live App:
👉 https://wiz-tan-cal-burnt-predictor321.streamlit.app/

📂 GitHub Repository:
👉 https://github.com/Wiz-Tanay/CalBurntPredictor.git

📌 Features
📊 Predict calories burned in real-time
🤖 Multiple ML models implemented and compared
📈 Visual comparison of model performance
📉 Feature importance and prediction plots
⚡ Interactive UI using Streamlit
🧠 Models Implemented
Linear Regression (lr.pkl)
Decision Tree (dt.pkl)
Random Forest (rf.pkl)
XGBoost (xgb.pkl, xgb_model.pkl)

Each model is evaluated and compared using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
📊 Visual Outputs

The project includes generated visualizations for each model:

Feature Importance Plots
lr_feature.png, dt_feature.png, rf_feature.png, xgb_feature.png
Prediction Comparison Plots
lr_plot.png, dt_plot.png, rf_plot.png, xgb_plot.png
Overall Model Comparison
comparison.png
🛠️ Tech Stack
Frontend: Streamlit
Backend: Python
ML Libraries:
Scikit-learn
XGBoost
Data Handling:
Pandas, NumPy
Visualization:
Matplotlib, Seaborn
📂 Project Structure
CalBurntPredictor/
│
├── .vscode/                 # Editor settings
├── .gitignore
│
├── app.py                   # Streamlit web app
├── CalBurntAnalysis.ipynb   # Model training & analysis
│
├── calories.csv             # Dataset
├── results.csv              # Model evaluation results
│
├── requirements.txt         # Dependencies
│
├── scaler.pkl               # Data scaler
│
├── lr.pkl                   # Linear Regression model
├── dt.pkl                   # Decision Tree model
├── rf.pkl                   # Random Forest model
├── xgb.pkl                  # XGBoost model
├── xgb_model.pkl            # Optimized XGBoost model
│
├── comparison.png           # Model comparison chart
│
├── lr_feature.png
├── dt_feature.png
├── rf_feature.png
├── xgb_feature.png
│
├── lr_plot.png
├── dt_plot.png
├── rf_plot.png
├── xgb_plot.png
⚙️ How It Works
User inputs:
Age
Gender
Height & Weight
Duration of exercise
Heart rate
Body temperature
Input is scaled using scaler.pkl
Data is passed into trained ML models
Predictions are generated
Best-performing model output is displayed
▶️ Run Locally
1. Clone the repository
git clone https://github.com/Wiz-Tanay/CalBurntPredictor.git
cd CalBurntPredictor
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py
📌 Key Highlights
✔️ End-to-end ML pipeline (Data → Training → Deployment)
✔️ Multiple models with comparative analysis
✔️ Real-time prediction system
✔️ Fully deployed Streamlit application
🚧 Future Improvements
Add deep learning-based models
Improve UI (navigation tabs, responsiveness)
Add user input history & tracking
Deploy backend API (FastAPI/Flask)
👨‍💻 Author

Tanay Pandey
🔗 https://github.com/Wiz-Tanay

⭐ Support

If you found this useful, consider giving it a ⭐ on GitHub!
