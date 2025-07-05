📉 Telco Customer Churn Prediction

A complete end-to-end data science project to predict whether a telecom customer is likely to churn. This project covers data analysis, machine learning modeling, and deployment using Streamlit.

-- Dataset Overview

Source: Telco Customer Churn dataset

Size: 7043 records, 21 features

Target Variable: Churn (Yes/No)

Features include:

Demographics: Gender, SeniorCitizen, Partner, Dependents

Services: InternetService, PhoneService, TechSupport, StreamingTV, etc.

Contract & Payment: Contract type, Payment method, PaperlessBilling

Usage & Charges: Tenure, MonthlyCharges, TotalCharges

-- Exploratory Data Analysis (EDA) & Insights

✅ Key Observations:

Churn Rate: ~26.58%

High Churn Groups:

Senior Citizens (with ~90% churn for label 1)

Customers without Tech Support

Customers on Month-to-Month contracts

Fiber optic internet users

Low Churn Groups:

Customers with Two-Year Contracts

Customers with Tech Support

Those with no internet service (also low engagement)

-- Visual Highlights:

Contract Type vs Churn:

Month-to-Month: Highest churn (~43%)

Two Year: Lowest (~3.6%)

Tech Support Impact:

No Tech Support: Churn Rate ~45%

With Tech Support: Churn ~15%

Streaming Services:

No Streaming TV: Slightly higher churn

Tenure & Monthly Charges:

Low tenure users are more likely to churn

High charges increase churn likelihood

-- Feature Engineering & Preprocessing

Encoded categorical variables using one-hot encoding

Converted TotalCharges to numeric (handled empty string conversion)

Mapped target variable Churn to binary: Yes = 1, No = 0

Removed irrelevant feature: customerID

--Model Building & Evaluation

Trained multiple models and compared their performance.

Final Chosen Model: Logistic Regression

Model

Accuracy

Precision

Recall

F1 Score

Logistic Regression

0.80

0.64

0.57

0.60

Random Forest

0.79

0.63

0.52

0.57

SVM

0.79

0.63

0.49

0.55

XGBoost

0.77

0.57

0.52

0.54

Naive Bayes

0.64

0.42

0.86

0.56

Confusion Matrix (Logistic Regression)

[[913 120]
 [161 213]]

--Streamlit Deployment

Built an interactive Streamlit web app where users can input customer details and get real-time churn predictions.

Live Demo:

🌐 https://telco-churn-predictor-k6vuh7nehjf6dprtmykawq.streamlit.app/

Inputs:

Gender, Senior Status, Partner, Dependents

Tenure, Monthly Charges, Total Charges

Contract Type, Internet Service, Tech Support, etc.

Output:

Prediction: Churn / No Churn

Confidence Score (probability)

📁 Repository Structure

|- app.py                # Streamlit app script
|- log_model.pkl         # Trained logistic regression model
|- telco_churn_analysis.ipynb  # Full Kaggle notebook (EDA + modeling)
|- requirements.txt      # Python dependencies
|- README.md             # Project documentation

-- Key Takeaways

Model achieved 80% accuracy on test set

Streamlit app deployed successfully for real-time predictions

End-to-end workflow covered: from EDA to deployment

Project ready for portfolio and resume

💼 Author

Arif Raza
