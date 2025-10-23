📉 Telco Customer Churn Prediction

An end-to-end machine learning project to predict whether a telecom customer is likely to churn. This project demonstrates data preprocessing, exploratory data analysis, model building, evaluation, and deployment using Streamlit.

-- Dataset Overview

Source: Telco Customer Churn dataset

Records: 7043

Features: 21 (including the target Churn)

Key Feature Categories:

Demographics: Gender, SeniorCitizen, Partner, Dependents

Services: InternetService, PhoneService, TechSupport, StreamingTV, StreamingMovies

Billing & Contracts: Contract type, Payment method, PaperlessBilling

Usage & Charges: Tenure, MonthlyCharges, TotalCharges

-- Exploratory Data Analysis (EDA) & Insights

✅ Key Observations:

Churn Rate: ~26.58%

Groups with High Churn:

Senior Citizens (~90% churn in class 1)

Users without Tech Support

Month-to-Month contract users

Fiber Optic internet subscribers

Groups with Low Churn:

Users with Two-Year contracts

Customers who opted for Tech Support

Users with no Internet service

--Visual Highlights:

Contract Type vs Churn:

Month-to-Month: ~43% churn rate

Two Year: ~3.6% churn rate

Tech Support Impact:

Without Tech Support: ~45% churn

With Tech Support: ~15% churn

Streaming Services:

Users without Streaming TV churn slightly more

Tenure & Charges:

Customers with short tenure are more likely to churn

Higher Monthly Charges correlates with increased churn

--Feature Engineering & Preprocessing

Applied one-hot encoding to categorical variables

Cleaned and converted TotalCharges to numeric

Mapped Churn: Yes = 1, No = 0

Dropped unnecessary identifier: customerID


--Streamlit Deployment

Built and deployed a user-friendly web app using Streamlit where users can input customer data and receive real-time churn predictions.

-- Live Demo: https://telco-churn-predictor-k6vuh7nehjf6dprtmykawq.streamlit.app/

📝 App Features:

Prediction result: Churn or No Churn

Confidence score (prediction probability)

-- Repository Structure

|- app.py                      # Streamlit app code
|- log_model.pkl               # Saved trained model
|- telco_churn_analysis.ipynb  # Kaggle notebook with full analysis
|- requirements.txt            # Project dependencies
|- README.md                   # Project documentation

-- Key Takeaways

Achieved 80% accuracy using Logistic Regression

Identified high-risk customer segments through EDA

Fully deployed an interactive web app with Streamlit

💼 Author

Arif-774729

