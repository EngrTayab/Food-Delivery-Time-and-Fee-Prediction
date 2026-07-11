import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import streamlit as st
import pandas as pd
import joblib


# -------------------------------
# Page Configuration
# -------------------------------`
st.set_page_config(
    page_title="Food Delivery Prediction System",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Models
# -------------------------------
time_model = joblib.load("delivery_time_model.pkl")
price_model = joblib.load("delivery_fee_model.pkl")
df = pd.read_csv('Food_delivery_dataset.csv')

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#ff4b4b;
    text-align:center;
}

.subtitle{
    font-size:20px;
    text-align:center;
    color:gray;
}

# .card{
#     background:white;
#     padding:20px;
#     border-radius:15px;
#     box-shadow:0px 4px 15px rgba(0,0,0,0.12);
#     margin-bottom:20px;
# }

# .metric-card{
#     background:#ffffff;
#     padding:20px;
#     border-radius:15px;
#     text-align:center;
#     box-shadow:0px 4px 12px rgba(0,0,0,0.15);
# }

# .feature-card{
#     background:#fff;
#     padding:15px;
#     border-radius:12px;
#     box-shadow:0px 2px 10px rgba(0,0,0,0.10);
# }
.card{
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.12);
    margin-bottom:20px;
}

.metric-card{
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.feature-card{
    background: var(--secondary-background-color);
    color: var(--text-color);
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.10);
}

.footer{
    text-align:center;
    color:gray;
    padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1046/1046784.png",
    width=120
)

st.sidebar.title("Menu")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Predict",
        "📁 Dataset",
        "🤖 Models",
        "📈 Visualizations",
        "👨‍💻 Developer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
### Project

Food Delivery Time &
Price Prediction

Using Machine Learning
""")

st.sidebar.success("Model Status\n\n✅ Loaded Successfully")

# ==========================================================
# HOME PAGE
# ==========================================================

if page=="🏠 Home":

    st.markdown(
        "<p class='title'>🍔 Food Delivery Time & Price Prediction System</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>Predict delivery time and delivery fee using Machine Learning</p>",
        unsafe_allow_html=True
    )

    st.write("")

    col1,col2=st.columns([2,1])

    with col1:

        st.markdown("""
<div class="card">

## 📌 About the Project

The **Food Delivery Time & Price Prediction System** is a Machine Learning application
designed to estimate both:

- ⏱ Delivery Time
- 💰 Delivery Fee

using important delivery-related features.

The system predicts results based on:

- 📍 Delivery Distance
- 🌦 Weather Condition
- 🚦 Traffic Level
- 🕒 Time of Day
- 🛵 Vehicle Type
- 🍽 Preparation Time
- 👨‍💼 Courier Experience

Two separate Machine Learning models were developed,
trained and evaluated to provide accurate predictions.

</div>
""", unsafe_allow_html=True)

    with col2:

        st.markdown("""
<div class="metric-card">

# 📊 Project Info

**Models**

2

---

**Algorithms Tested**

4

---

**Prediction Tasks**

2

---

**Deployment**

Streamlit

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("✨ Main Features")

    c1,c2,c3=st.columns(3)

    with c1:

        st.markdown("""
<div class="feature-card">

### ⏱ Delivery Time

Predicts the estimated
food delivery time using
Machine Learning.

</div>
""", unsafe_allow_html=True)

    with c2:

        st.markdown("""
<div class="feature-card">

### 💰 Delivery Fee

Predicts the estimated
delivery fee based on
delivery conditions.

</div>
""", unsafe_allow_html=True)

    with c3:

        st.markdown("""
<div class="feature-card">

### 🤖 Machine Learning

Built using Scikit-Learn,
Pandas, NumPy,
and Streamlit.

</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("🛠 Technologies Used")

    tech1,tech2,tech3,tech4=st.columns(4)

    tech1.success("Python")
    tech2.success("Scikit-Learn")
    tech3.success("Streamlit")
    tech4.success("Pandas")

    st.write("")

    st.info("""
### Workflow

Dataset → Data Cleaning → Feature Engineering → Model Training → Model Evaluation → Deployment

Four Machine Learning algorithms were evaluated:

• Linear Regression

• Decision Tree

• Random Forest

• XGBoost

The best-performing model was selected based on **MAE**, **RMSE**, and **R² Score**.
""")
    
# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "📊 Predict":

    st.title("📊 Food Delivery Prediction")

    st.write(
        "Enter the delivery information below to predict the estimated delivery "
        "time and delivery fee."
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        distance = st.number_input(
            "📍 Distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=5.0,
            step=0.5
        )

        weather = st.selectbox(
            "🌦 Weather",
            [
                "Clear",
                "Foggy",
                "Rainy",
                "Snowy",
                "Windy"
            ]
        )

        traffic = st.selectbox(
            "🚦 Traffic Level",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        time_day = st.selectbox(
            "🕒 Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )

    with col2:

        vehicle = st.selectbox(
            "🛵 Vehicle Type",
            [
                "Bike",
                "Scooter",
                "Car"
            ]
        )

        prep_time = st.number_input(
            "🍽 Preparation Time (minutes)",
            min_value=1,
            max_value=120,
            value=20
        )

        experience = st.number_input(
            "👨‍💼 Courier Experience (Years)",
            min_value=0,
            max_value=30,
            value=3
        )

    st.markdown("---")

    predict = st.button(
        "🚀 Predict",
        use_container_width=True
    )

    if predict:

        input_df = pd.DataFrame({
            "Distance_km": [distance],
            "Weather": [weather],
            "Traffic_Level": [traffic],
            "Time_of_Day": [time_day],
            "Vehicle_Type": [vehicle],
            "Preparation_Time_min": [prep_time],
            "Courier_Experience_yrs": [experience]
        })

        delivery_time = time_model.predict(input_df)[0]
        delivery_fee = price_model.predict(input_df)[0]

        st.success("Prediction completed successfully!")

        st.markdown("## 📈 Prediction Results")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                label="⏱ Estimated Delivery Time",
                value=f"{delivery_time:.2f} Minutes"
            )

        with c2:

            st.metric(
                label="💰 Estimated Delivery Fee",
                value=f"Rs. {delivery_fee:.2f}"
            )

        st.markdown("---")

        st.subheader("📋 Input Summary")

        st.dataframe(
            input_df,
            use_container_width=True
        )

        st.info(
            """
            **Note**

            These predictions are generated using trained Machine Learning
            models and should be considered as estimated values.
            Actual delivery time and delivery fee may vary depending on
            real-world conditions.
            """
        )
        
# ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📁 Dataset":

    st.title("📁 Dataset Information")

    st.markdown("""
    This project uses a **Food Delivery Dataset** to predict:

    - ⏱ Delivery Time
    - 💰 Delivery Fee

    The dataset contains numerical and categorical features
    describing delivery conditions.
    """)

    st.markdown("---")

    st.subheader("📊 Dataset Overview")

    overview = pd.DataFrame({
        "Attribute":[
            "Prediction Tasks",
            "Machine Learning Type",
            "Target Variables",
            "Input Features",
            "Algorithms Evaluated"
        ],

        "Value":[
            "Delivery Time & Delivery Fee",
            "Regression",
            "Delivery_Time_min, Delivery_Fee",
            "7",
            "4"
        ]
    })

    st.table(overview)

    st.markdown("---")

    st.subheader("📋 Input Features")

    features = pd.DataFrame({

        "Feature":[
            "Distance_km",
            "Weather",
            "Traffic_Level",
            "Time_of_Day",
            "Vehicle_Type",
            "Preparation_Time_min",
            "Courier_Experience_yrs"
        ],

        "Description":[
            "Distance between restaurant and customer",
            "Current weather condition",
            "Traffic congestion level",
            "Morning, Afternoon, Evening or Night",
            "Bike, Scooter or Car",
            "Restaurant preparation time",
            "Experience of delivery courier"
        ]

    })

    st.dataframe(features, use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 Target Variables")

    col1,col2=st.columns(2)

    with col1:

        st.success("""
### ⏱ Delivery_Time_min

Predicts the estimated time required to
deliver the food order.

Regression Problem
""")

    with col2:

        st.success("""
### 💰 Delivery_Fee

Predicts the estimated delivery fee
charged for an order.

Regression Problem
""")

    st.markdown("---")

    st.subheader("🧹 Data Preprocessing")

    st.markdown("""

The following preprocessing steps were performed before model training:

✅ Removed unnecessary columns

✅ Checked and handled missing values

✅ Created Delivery Fee feature

✅ Separated numerical and categorical features

✅ Applied StandardScaler to numerical columns

✅ Applied OneHotEncoder to categorical columns

✅ Built preprocessing pipeline using ColumnTransformer

✅ Split dataset into training and testing sets

""")

    st.markdown("---")

    st.subheader("⚙ Machine Learning Pipeline")

    st.code("""

Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train Test Split
      │
      ▼
ColumnTransformer
      │
      ├── StandardScaler
      │
      └── OneHotEncoder
      │
      ▼
Linear Regression
      │
      ▼
Prediction

""")

    st.markdown("---")

    st.subheader("🛠 Technologies Used")

    tech1,tech2,tech3=st.columns(3)

    with tech1:

        st.info("""
### Python Libraries

• Pandas

• NumPy

• Joblib
""")

    with tech2:

        st.info("""
### Machine Learning

• Scikit-Learn

• Linear Regression

• XGBoost

• Random Forest

• Decision Tree
""")

    with tech3:

        st.info("""
### Deployment

• Streamlit

• GitHub

• Streamlit Community Cloud
""")

    st.markdown("---")

    st.success("""
### Summary

This project predicts both food delivery time and delivery fee using
Machine Learning techniques. The dataset was carefully preprocessed,
multiple regression algorithms were evaluated, and the best-performing
model was selected for deployment.
""")
    
# ==========================================================
# MODELS PAGE
# ==========================================================

elif page == "🤖 Models":

    st.title("🤖 Machine Learning Models")

    st.write("""
    Four regression algorithms were trained and evaluated for predicting
    **Delivery Time** and **Delivery Fee**.
    The best-performing model was selected based on **MAE**, **RMSE**,
    **R² Score**, and overall generalization performance.
    """)

    st.markdown("---")

    st.subheader("📚 Algorithms Used")

    # -----------------------------------------------------

    with st.expander("📈 Linear Regression (Selected Model)", expanded=True):

        st.write("""

**Description**

Linear Regression is a supervised machine learning algorithm used for
predicting continuous numerical values. It models the relationship between
input features and the target variable using a straight-line equation.

### Advantages

✅ Simple and fast

✅ Easy to interpret

✅ Excellent for linear relationships

✅ Low computational cost

✅ Less prone to overfitting on linear datasets

### Why was it selected?

Among all evaluated models, Linear Regression produced the:

• Highest R² Score

• Lowest MAE

• Lowest RMSE

for both Delivery Time and Delivery Fee prediction.

The dataset exhibited relationships that were close to linear,
allowing Linear Regression to outperform more complex models.

""")

    # -----------------------------------------------------

    with st.expander("🌳 Decision Tree"):

        st.write("""

Decision Tree Regression predicts values by recursively splitting
the dataset into smaller regions.

### Advantages

✔ Easy to understand

✔ Captures nonlinear relationships

### Limitations

• Can overfit training data

• Sensitive to small data changes

• Lower prediction accuracy on this dataset

""")

    # -----------------------------------------------------

    with st.expander("🌲 Random Forest"):

        st.write("""

Random Forest is an ensemble learning algorithm that combines
multiple Decision Trees to improve prediction performance.

### Advantages

✔ Better than a single Decision Tree

✔ Handles nonlinear relationships

✔ Reduces overfitting

### Limitations

Although Random Forest achieved good performance,
its prediction error was higher than Linear Regression
for this dataset.

""")

    # -----------------------------------------------------

    with st.expander("🚀 XGBoost"):

        st.write("""

XGBoost is a powerful gradient boosting algorithm
widely used for structured datasets.

### Advantages

✔ High predictive power

✔ Handles complex datasets

✔ Excellent performance on many ML competitions

### Limitations

Despite its advanced architecture,
XGBoost achieved slightly lower R²
and higher prediction errors compared
to Linear Regression on this dataset.

""")

    st.markdown("---")

    st.subheader("📊 Delivery Time Model Comparison")

    delivery_time = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "Decision Tree",
            "XGBoost"
        ],

        "MAE":[
            5.90,
            6.80,
            10.46,
            7.18
        ],

        "RMSE":[
            8.82,
            9.61,
            15.29,
            10.24
        ],

        "R² Score":[
            0.8263,
            0.7938,
            0.4781,
            0.7658
        ]

    })

    st.dataframe(delivery_time, use_container_width=True)

    st.success("""
🏆 Best Model

Linear Regression

R² Score : 0.8263

Reason:
Highest R² with the lowest prediction error.
""")

    st.markdown("---")

    st.subheader("💰 Delivery Fee Model Comparison")

    delivery_fee = pd.DataFrame({

        "Model":[
            "Linear Regression",
            "Random Forest",
            "Decision Tree",
            "XGBoost"
        ],

        "MAE":[
            5.24,
            7.81,
            11.08,
            6.35
        ],

        "RMSE":[
            6.15,
            9.53,
            13.89,
            7.64
        ],

        "R² Score":[
            0.9950,
            0.9879,
            0.9742,
            0.9922
        ]

    })

    st.dataframe(delivery_fee, use_container_width=True)

    st.success("""
🏆 Best Model

Linear Regression

R² Score : 0.9950

Reason:
Highest R² with the lowest MAE and RMSE.
""")

    st.markdown("---")

    st.subheader("📈 Cross Validation")

    st.info("""

5-Fold Cross Validation was performed to evaluate
the generalization capability of the Delivery Time model.

Average Cross Validation R² Score

**0.7685**

This indicates that the model performs consistently
across multiple train-test splits and is not overfitted.

""")

    st.markdown("---")

    st.subheader("🎯 Final Model Selection")

    st.success("""

After evaluating all four regression algorithms,
Linear Regression was selected as the final model
for deployment.

Reasons:

✅ Highest R² Score

✅ Lowest MAE

✅ Lowest RMSE

✅ Fast prediction speed

✅ Simple and interpretable

✅ Consistent Cross Validation performance

Therefore, Linear Regression was deployed
for both Delivery Time Prediction and
Delivery Fee Prediction.

""")
    
    
# ==========================================================
# VISUALIZATIONS PAGE
# ==========================================================

elif page == "📈 Visualizations":

    st.title("📈 Data Visualizations")

    st.write("""
These visualizations help understand the dataset,
feature relationships and model performance.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Delivery Time Distribution
    # -------------------------------------------------

    st.subheader("1️⃣ Delivery Time Distribution")

    fig, ax = plt.subplots(figsize=(8,4))

    ax.hist(
        df["Delivery_Time_min"],
        bins=20,
        edgecolor="black"
    )

    ax.set_xlabel("Delivery Time (Minutes)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Delivery Time")

    st.pyplot(fig)

    st.info("""
This histogram illustrates how delivery times are distributed
across all food orders.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Distance vs Delivery Time
    # -------------------------------------------------

    st.subheader("2️⃣ Distance vs Delivery Time")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(
        df["Distance_km"],
        df["Delivery_Time_min"]
    )

    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Delivery Time (Minutes)")
    ax.set_title("Distance vs Delivery Time")

    st.pyplot(fig)

    st.info("""
As delivery distance increases,
the delivery time generally increases.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------

    st.subheader("3️⃣ Correlation Heatmap")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(8,6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    st.pyplot(fig)

    st.info("""
The heatmap shows the correlation between numerical variables.
Higher values indicate stronger relationships.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Model Comparison
    # -------------------------------------------------

    st.subheader("4️⃣ Delivery Time Model Comparison")

    models = [
        "Linear",
        "Random Forest",
        "Decision Tree",
        "XGBoost"
    ]

    scores = [
        0.8263,
        0.7938,
        0.4781,
        0.7658
    ]

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(models, scores)

    ax.set_ylabel("R² Score")
    ax.set_title("Delivery Time Models")

    st.pyplot(fig)

    st.success("""
Linear Regression achieved the highest R² score,
making it the best model for delivery time prediction.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Delivery Fee Model Comparison
    # -------------------------------------------------

    st.subheader("5️⃣ Delivery Fee Model Comparison")

    fee_scores = [
        0.9950,
        0.9879,
        0.9742,
        0.9922
    ]

    fig, ax = plt.subplots(figsize=(8,4))

    ax.bar(models, fee_scores)

    ax.set_ylabel("R² Score")
    ax.set_title("Delivery Fee Models")

    st.pyplot(fig)

    st.success("""
Linear Regression also achieved the best performance
for delivery fee prediction.
""")

    st.markdown("---")

    # -------------------------------------------------
    # Project Summary
    # -------------------------------------------------

    st.subheader("📊 Visualization Summary")

    st.write("""
The visualizations provide valuable insights into the dataset:

- 📍 Delivery distance has a positive relationship with delivery time.
- 🌦 Weather and traffic indirectly affect delivery performance.
- 📈 Linear Regression achieved the highest prediction accuracy.
- 🤖 Both deployed models showed excellent generalization.
- 📊 Visual analysis supports the model evaluation results.
""")
    
# ==========================================================
# DEVELOPER PAGE
# ==========================================================

elif page == "👨‍💻 Developer":

    st.title("👨‍💻 Developer")

   

    st.markdown("""
# Muhammad Tayyab Asif

### BS Software Engineering

**The Islamia University of Bahawalpur**

6th Semester

---

### About Me

I am a Software Engineering student with a strong interest in
Machine Learning, Artificial Intelligence, Python Development,
and Web Technologies.

I enjoy building intelligent applications that solve real-world
problems through data-driven solutions.
""")
    st.markdown("---")

    st.subheader("📞 Contact Information")

    st.markdown("""
                   
                📧 **Email:** tayyabasif.dev@gmail.com

                

                💼 **LinkedIn:** www.linkedin.com/in/muhammad-tayyab-asif-5391bb307

                💻 **GitHub:** https://github.com/EngrTayab
            """)


    st.markdown("---")

    st.subheader("🛠 Technical Skills")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.success("""
### Programming

• Python

• Java

• JavaScript

• SQL
""")

    with c2:

        st.success("""
### Machine Learning

• Scikit-Learn

• Pandas

• NumPy

• XGBoost

• Data Preprocessing
""")

    with c3:

        st.success("""
### Development

• Streamlit

• HTML

• CSS

• Bootstrap

• Git & GitHub
""")

    st.markdown("---")

    st.subheader("📂 Project Information")

    st.info("""
### Food Delivery Time & Price Prediction System

This project predicts:

✅ Food Delivery Time

✅ Delivery Fee

using Machine Learning Regression algorithms.

Four algorithms were evaluated:

• Linear Regression

• Decision Tree

• Random Forest

• XGBoost

The final deployed model was selected after comparing
MAE, RMSE and R² Score.

Linear Regression achieved the best performance and
was deployed for both prediction tasks.
""")

    st.markdown("---")

    st.subheader("🎯 Project Objectives")

    st.write("""

✔ Predict food delivery time accurately.

✔ Estimate delivery fee using delivery conditions.

✔ Compare multiple machine learning algorithms.

✔ Evaluate models using regression metrics.

✔ Deploy the best model using Streamlit.

✔ Build a professional and user-friendly application.

""")



    

    st.markdown("---")

    st.success("""
### Thank You

Thank you for exploring the Food Delivery Prediction System.

This application demonstrates the complete Machine Learning workflow:

✔ Data Cleaning

✔ Feature Engineering

✔ Model Training

✔ Model Evaluation

✔ Model Deployment

using modern Python technologies.
""")

    st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:25px;
<div style="
text-align:center;
padding:25px;
background:#ff8a4b;
border-radius:12px;
color:white;
margin-top:30px;
">
border-radius:12px;
color:white;
margin-top:30px;
">

<h3>🍔 Food Delivery Time & Price Prediction System</h3>

<p>
Predict delivery time and delivery fee using Machine Learning.
</p>

<hr style="border:1px solid #444;">

<p>
<b>Instructor:</b> Sir Zafar Iqbal
</p>

<p>
<b>Developer:</b> Muhammad Tayyab Asif
</p>

<p>
BS Software Engineering | 6th Semester
</p>

<p>
The Islamia University of Bahawalpur
</p>

<p style="color:#A9A9A9;">
Built with ❤️ using Python, Streamlit & Scikit-Learn
</p>

<p style="font-size:13px;color:gray;">
© 2026 Muhammad Tayyab Asif. All Rights Reserved.
</p>

</div>
""", unsafe_allow_html=True)
    
            
                
            
