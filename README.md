# 🔥 Calorie Burnt Predictor

A Machine Learning-based web application that predicts calories burned using user-specific physiological and exercise-related inputs.

🚀 **Live App:**  
https://wiz-tan-cal-burnt-predictor321.streamlit.app/

📂 **GitHub Repository:**  
https://github.com/Wiz-Tanay/CalBurntPredictor.git

---

## 📌 Features

- 📊 Real-time calorie prediction  
- 🤖 Multiple ML models implemented and compared  
- 📈 Model performance comparison  
- 📉 Feature importance & prediction visualizations  
- ⚡ Interactive UI built with Streamlit  

---

## 🧠 Models Implemented

- Linear Regression (`lr.pkl`)  
- Decision Tree (`dt.pkl`)  
- Random Forest (`rf.pkl`)  
- XGBoost (`xgb.pkl`, `xgb_model.pkl`)  

### 📊 Evaluation Metrics

- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- R² Score  

---

## 📊 Visual Outputs

- Feature Importance Plots  
  - `lr_feature.png`, `dt_feature.png`, `rf_feature.png`, `xgb_feature.png`

- Prediction Comparison Plots  
  - `lr_plot.png`, `dt_plot.png`, `rf_plot.png`, `xgb_plot.png`

- Model Comparison  
  - `comparison.png`

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Libraries:** Scikit-learn, XGBoost  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  

---

## 📂 Project Structure

- `.vscode/` – editor settings  
- `app.py` – Streamlit web app  
- `CalBurntAnalysis.ipynb` – model training & analysis  
- `calories.csv` – dataset  
- `results.csv` – model evaluation results  
- `requirements.txt` – dependencies  
- `scaler.pkl` – data scaler  

### Models
- `lr.pkl` – Linear Regression  
- `dt.pkl` – Decision Tree  
- `rf.pkl` – Random Forest  
- `xgb.pkl`, `xgb_model.pkl` – XGBoost  

### Visualizations
- `comparison.png`  
- `*_feature.png` – feature importance plots  
- `*_plot.png` – prediction plots  

---

## ⚙️ How It Works

1. User inputs:
   - Age  
   - Gender  
   - Height & Weight  
   - Duration of exercise  
   - Heart rate  
   - Body temperature  

2. Input is scaled using `scaler.pkl`  
3. Data is passed into trained ML models  
4. Predictions are generated  
5. Best-performing model result is displayed  

---

## ▶️ Run Locally

### 1. Clone the repository

git clone https://github.com/Wiz-Tanay/CalBurntPredictor.git  
cd CalBurntPredictor  

### 2. Install dependencies

pip install -r requirements.txt  

### 3. Run the app

streamlit run app.py  

---

## 📌 Key Highlights

- ✔️ End-to-end ML pipeline (Data → Training → Deployment)  
- ✔️ Multiple models with comparison  
- ✔️ Real-time prediction system  
- ✔️ Deployed Streamlit application  

---

## 🚧 Future Improvements

- Add deep learning models  
- Improve UI navigation  
- Add user history tracking  
- Deploy backend API  

---

## 👨‍💻 Author

**Tanay Pandey**  
https://github.com/Wiz-Tanay  

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
