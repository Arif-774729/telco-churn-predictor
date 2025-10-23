import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load Model ---
# NOTE: The model must be saved as 'naive_bayes_model.pkl' in the same directory.
try:
    with open("naive_bayes_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'naive_bayes_model.pkl' not found. Please ensure it is saved.")
    st.stop()

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📉 Telco Customer Churn Predictor")
st.markdown("This app predicts whether a customer is likely to churn based on their profile. We use features proven most important by the Naive Bayes model.")

# --- Collect User Input (Excluding Gender) ---

st.header("1. Financial & Contract")
col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0, step=1.0)
with col2:
    total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
with col3:
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

st.header("2. Service Subscriptions")
col4, col5, col6 = st.columns(3)
with col4:
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['Yes', 'No internet service', 'No'])
    online_backup = st.selectbox("Online Backup", ['Yes', 'No internet service', 'No'])
with col5:
    device_protection = st.selectbox("Device Protection", ['Yes', 'No internet service', 'No'])
    tech_support = st.selectbox("Tech Support", ['Yes', 'No internet service', 'No'])
    streaming_tv = st.selectbox("Streaming TV", ['Yes', 'No internet service', 'No'])
with col6:
    streaming_movies = st.selectbox("Streaming Movies", ['Yes', 'No internet service', 'No'])
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No phone service', 'No'])

st.header("3. Demographics")
col7, col8 = st.columns(2)
with col7:
    senior = st.selectbox("Senior Citizen", ['Yes', 'No'])
    partner = st.selectbox("Has Partner?", ['Yes', 'No'])
with col8:
    dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])

# --- Preprocessing Function (INCLUDES ENGINEERED FEATURES) ---

def preprocess_input():
    """
    Converts user inputs into the model's required DataFrame format, 
    including re-creating engineered features and one-hot encoding.
    """
    
    # 1. Create Initial DataFrame (Note: 'gender' is excluded as per analysis)
    data = {
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        # 'gender' is implicitly handled by absence, which aligns with drop_first=True
    }
    input_df = pd.DataFrame([data])

    # 2. Recreate Engineered Features (CRITICAL STEP)
    
    # A. Average_Charge_per_Month
    # Handle division by zero for new customers (tenure=0)
    input_df['Average_Charge_per_Month'] = input_df['TotalCharges'] / input_df['tenure']
    input_df.loc[input_df['tenure'] == 0, 'Average_Charge_per_Month'] = 0
    input_df['Average_Charge_per_Month'].fillna(0, inplace=True) 
    
    # B. Service_Count
    # Service Count relies on 6 internet/streaming services. Must align categories first.
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_df = input_df[internet_services].replace('No internet service', 'No') # Align categories
    input_df['Service_Count'] = service_df.apply(
        lambda row: sum(row == 'Yes'), axis=1
    )

    # C. Long_Term_Customer
    input_df['Long_Term_Customer'] = (input_df['tenure'] >= 30).astype(int)

    # 3. One-Hot Encoding
    cat_features = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    # We must include 'gender' in the list to ensure the correct number of dummy columns is created, 
    # even if its input value isn't used (we'll manually add the 'gender' column to the df to handle the encoding).
    input_df['gender'] = 'Male' # Add 'gender' placeholder for encoding consistency
    cat_features.append('gender')
    
    input_df = pd.get_dummies(input_df, columns=cat_features, drop_first=True)

    # 4. Align Columns with Trained Model (CRITICAL STEP)
    
    # Create a skeleton DataFrame with all expected features initialized to 0
    final_features = pd.DataFrame(columns=model.feature_names_in_)
    final_features.loc[0] = 0

    # Map the input values onto this skeleton DataFrame
    for col in input_df.columns:
        if col in final_features.columns:
            final_features.loc[0, col] = input_df.loc[0, col]

    # Return the DataFrame with features in the model's expected order
    return final_features[model.feature_names_in_]

# --- Prediction ---
if st.button("Predict Churn"):
    try:
        input_data = preprocess_input()
        
        # Check if the feature count matches the model's expectation
        if len(input_data.columns) != len(model.feature_names_in_):
             st.error(f"Error: Feature count mismatch. Input has {len(input_data.columns)} features, model expects {len(model.feature_names_in_)}.")
        else:
            prediction = model.predict(input_data)[0]
            # Naive Bayes predict_proba returns [prob_0, prob_1]
            prob = model.predict_proba(input_data)[0][1]

            if prediction == 1:
                st.error(f"⚠️ This customer is likely to **CHURN**. (Probability: {prob:.2f})")
            else:
                st.success(f"✅ This customer is likely to **STAY**. (Probability: {1 - prob:.2f})")
                
    except Exception as e:
        st.error(f"An error occurred during preprocessing or prediction: {e}")
