# 🍔 Food Delivery Time & Delivery Fee Prediction System

A Machine Learning web application that predicts **Food Delivery Time** and **Delivery Fee** based on delivery-related factors. The project is developed using **Python**, **Scikit-Learn**, and **Streamlit** and deployed on **Streamlit Community Cloud**.

## 📌 Project Overview

The Food Delivery Time & Delivery Fee Prediction System helps estimate:

- ⏱ Estimated Delivery Time (minutes)
- 💰 Estimated Delivery Fee

The predictions are based on several delivery-related features such as distance, weather, traffic conditions, preparation time, courier experience, vehicle type, and time of day.

Two separate Machine Learning models were trained and integrated into a single Streamlit application.

---

## ✨ Features

- Predict Food Delivery Time
- Predict Delivery Fee
- Interactive Streamlit Interface
- Dataset Information Page
- Model Information Page
- Data Visualizations
- Developer Information
- Responsive UI

---

## 📊 Dataset Features

The models use the following input features:

- Distance (km)
- Weather
- Traffic Level
- Time of Day
- Vehicle Type
- Preparation Time (minutes)
- Courier Experience (years)

### Target Variables

- Delivery Time (minutes)
- Delivery Fee

---

## 🧹 Data Preprocessing

The dataset was preprocessed before model training using:

- Missing value handling
- Feature selection
- One-Hot Encoding for categorical features
- Standard Scaling for numerical features
- ColumnTransformer Pipeline

---

## 🤖 Machine Learning Models Tested

The following regression algorithms were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## 🏆 Best Model

### Delivery Time Prediction

**Selected Model:** Linear Regression

Evaluation Metrics:

- MAE: **5.90**
- RMSE: **8.82**
- R² Score: **0.8263**

### Delivery Fee Prediction

**Selected Model:** Linear Regression

Evaluation Metrics:

- MAE: **5.24**
- RMSE: **6.15**
- R² Score: **0.9950**

Linear Regression achieved the best balance of prediction accuracy and generalization performance among all evaluated models.

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross Validation

---

## 🛠 Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- XGBoost

---

## 📂 Project Structure

```
Food-Delivery-Time-and-Fee-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── delivery_time_model.pkl
├── delivery_fee_model.pkl
├── Food_delivery_dataset.csv
└── images/
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/EngrTayab/Food-Delivery-Time-and-Fee-Prediction.git
```

Move into the project directory

```bash
cd Food-Delivery-Time-and-Fee-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```


## 📄 License

This project is developed for educational and learning purposes.

---

⭐ If you found this project useful, consider giving it a star on GitHub.
