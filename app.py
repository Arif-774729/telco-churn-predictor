import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained logistic regression model
with open("log_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📉 Telco Customer Churn Predictor")
st.markdown("This app predicts whether a customer is likely to churn based on their profile.")

# Collect user input
gender = st.selectbox("Gender", ['Male', 'Female'])
senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
partner = st.selectbox("Has Partner?", ['Yes', 'No'])
dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
tenure = st.slider("Tenure (in months)", 0, 72, 1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)
contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])

# Encode categorical variables (you may need to adjust based on your feature engineering)
def preprocess_input():
    data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'InternetService': internet_service
    }

    input_df = pd.DataFrame([data])
    # Encode like in training (dummy encoding)
    cat_features = ['gender', 'Partner', 'Dependents', 'Contract', 'InternetService']
    input_df = pd.get_dummies(input_df, columns=cat_features)

    # Add missing columns if any
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure order
    input_df = input_df[model.feature_names_in_]

    return input_df

# Predict
if st.button("Predict Churn"):
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {1 - prob:.2f})")
