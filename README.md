# 📊 Customer Churn Prediction System

A complete end-to-end machine learning project to predict customer churn and provide actionable strategies to improve customer retention.

---

## 🔍 Overview

Customer churn is a critical problem for businesses. This project builds a machine learning system to:

- Predict whether a customer will churn
- Identify high-risk customers early
- Provide data-driven retention strategies

The solution includes data analysis, model training, and a deployed interactive Streamlit application.

---

## 📁 Dataset

The dataset contains both categorical and numerical features related to customer behavior.

### Key Features:
- Tenure (customer lifetime)
- Satisfaction Score
- Support Calls
- Late Payments
- Contract Type
- Payment Method
- Region

These features help capture customer engagement, financial behavior, and service experience.

---

## 🧹 Data Preprocessing

- Handled missing values using appropriate strategies (drop/imputation)
- Applied encoding techniques:
  - One-Hot Encoding for categorical variables
- Ensured consistent preprocessing using pipelines
- Converted raw data into model-ready format

---

## 📊 Exploratory Data Analysis (EDA)

Performed detailed analysis to understand customer behavior:

- Churn distribution analysis
- Feature relationships and trends
- Visualizations using Matplotlib and Seaborn

### Key Observations:
- Low satisfaction → High churn  
- High support calls → Customer frustration → Higher churn  
- Late payments → Increased churn probability  
- Monthly contracts → Higher churn risk  
- Long tenure → Lower churn (loyal customers)  

---

## ⚙️ Feature Engineering & Selection

- Removed irrelevant or redundant features
- Focused on high-impact variables using:
  - Correlation analysis
  - Domain knowledge
- Improved model performance and interpretability

---

## 🚨 Outlier Handling

- Checked numerical features for outliers
- Applied minimal filtering to avoid unnecessary data loss
- Maintained balance between data quality and quantity

---

## 🤖 Model Training

Trained multiple machine learning models:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- XGBoost  

---

## 🏆 Model Selection

Models were evaluated based on:

- Accuracy  
- Recall (priority metric for churn detection)

### Final Model:
**Gradient Boosting**

### Why?
- Balanced performance  
- Strong recall (better churn detection)  
- No significant overfitting  

---

## ⚡ Hyperparameter Tuning

- Used GridSearchCV / RandomizedSearchCV
- Applied cross-validation (optimized for dataset size)
- Improved model generalization

---

## 📈 Final Performance

- Accuracy: ~88%  
- Recall (Churn Detection): ~86%  

The model effectively identifies high-risk customers while maintaining overall performance.

---

## 🔄 Machine Learning Pipeline

Built a complete pipeline to ensure consistency:

- Data preprocessing  
- Encoding  
- Model training  

This ensures:
- Clean training workflow  
- Reliable predictions during deployment  

---

## 🛠️ Tech Stack

- Python  
- NumPy, Pandas  
- Scikit-learn  
- XGBoost  
- Matplotlib, Seaborn  
- Joblib (model persistence)  
- Streamlit (deployment)  

---

## 🚀 Deployment

The project is deployed using Streamlit with an interactive UI.

### Features:
- Customer input simulation  
- Real-time churn prediction  
- Risk classification (Low / Medium / High)  
- Personalized retention strategies  

---

## 🌐 Live Demo

🔗 https://customerchurnprediction-gsawlwntpmgq3ruxscepg7.streamlit.app/

---

## 📂 Project Structure

