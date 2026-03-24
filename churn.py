import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Page Configuration
st.set_page_config(page_title="Customer Churn Intelligence System", layout="wide")

st.title("Customer Churn Intelligence System")


# Load Data and Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR,"models", "processed.csv")
model_path = os.path.join(BASE_DIR,"data","churn_model.pkl")

df = pd.read_csv(data_path)
model = joblib.load(model_path)


# Sidebar Input (Customer Profile)
st.sidebar.title("Customer Profile")
st.sidebar.write("Adjust the values to simulate customer behavior")

tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
support_calls = st.sidebar.slider("Support Calls", 0, 10, 2)
late_payments = st.sidebar.slider("Late Payments", 0, 10, 1)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 10, 5)

contract_type = st.sidebar.selectbox("Contract Type", df["contract_type"].unique())
payment_method = st.sidebar.selectbox("Payment Method", df["payment_method"].unique())
region = st.sidebar.selectbox("Region", df["region"].unique())


# Tabs Navigation
tab1, tab2, tab3 = st.tabs(["Project & Data", "Analysis", "Prediction"])


# Project and Data Explanation
with tab1:
    st.header("Project Overview")

    st.write("""
    This system predicts whether a customer will churn based on behavioral, financial, and service-related factors.

    The goal is to help businesses identify high-risk customers early and take preventive actions.
    """)

    st.subheader("Dataset Information")

    st.write("""
    The dataset includes customer-related features such as tenure, satisfaction score, support interactions,
    payment behavior, contract type, and region.

    These features help in understanding customer engagement and identifying patterns that lead to churn.
    """)

    st.subheader("Key Insights from Data")

    st.write("""
    - Customers with low satisfaction scores are more likely to churn
    - Higher number of support calls indicates frustration and higher churn risk
    - Late payments increase the probability of churn
    - Customers with shorter tenure are less loyal and more likely to leave
    - Long-term customers are more stable and less likely to churn
    """)

    st.subheader("Model Information")

    st.write("""
    Multiple models were tested including Logistic Regression, AdaBoost, and Gradient Boosting.

    Gradient Boosting was selected as the final model because:
    - It provides better recall for churn detection
    - It maintains balanced performance without overfitting
    - It achieves strong accuracy (~88%) and churn detection (~86%)
    """)


# Data Analysis Section
with tab2:
    st.header("Customer Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="satisfaction_score", data=df, ax=ax)
        st.pyplot(fig)
        st.write("Lower satisfaction scores show higher churn probability")

    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x="churn", y="support_calls", data=df, ax=ax)
        st.pyplot(fig)
        st.write("Higher support calls indicate higher churn")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.write("""
    Correlation analysis shows that satisfaction score and tenure are negatively related to churn,
    while support calls and late payments increase churn probability.
    """)


# Prediction and Recommendation Section
with tab3:
    st.header("Customer Risk Prediction and Strategy")

    input_data = pd.DataFrame({
        "tenure_months": [tenure],
        "support_calls": [support_calls],
        "late_payments": [late_payments],
        "satisfaction_score": [satisfaction_score],
        "contract_type": [contract_type],
        "payment_method": [payment_method],
        "region": [region]
    })

    if st.button("Analyze Customer"):
        probability = model.predict_proba(input_data)[0][1]

        st.subheader(f"Churn Probability: {round(probability * 100, 2)}%")

        if probability > 0.75:
            st.error("High Risk Customer")
        elif probability > 0.4:
            st.warning("Moderate Risk Customer")
        else:
            st.success("Low Risk Customer")

        st.subheader("Recommended Actions")

        if satisfaction_score <= 3:
            st.write("Improve customer satisfaction through direct engagement and feedback")

        if support_calls >= 5:
            st.write("Assign a dedicated support agent to resolve issues quickly")

        if late_payments >= 3:
            st.write("Provide flexible payment options or reminders")

        if tenure < 12:
            st.write("Offer onboarding benefits or early retention discounts")

        if contract_type.lower() in ["monthly", "3 months", "6 months"]:
            st.write("Encourage long-term subscription plans with discounts")

        if probability < 0.4:
            st.write("Provide loyalty rewards and referral benefits") 
