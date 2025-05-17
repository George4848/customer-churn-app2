
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature columns
model = joblib.load("model.pkl")
feature_columns = joblib.load("features.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("Customer Churn Prediction")
st.sidebar.title("Prediction Mode")

mode = st.sidebar.radio("Choose input method:", ["Manual Input", "Upload CSV File"])

def preprocess_input(data):
    # One-hot encode and align columns with training data
    data_encoded = pd.get_dummies(data)
    data_aligned = data_encoded.reindex(columns=feature_columns, fill_value=0)
    return data_aligned

if mode == "Manual Input":
    st.header("Enter Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    if st.button("Predict"):
        processed = preprocess_input(input_data)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        result = "Yes" if prediction == 1 else "No"
        st.subheader(f"Will the customer churn? **{result}**")
        st.write(f"Churn Probability: **{probability:.2%}**")

elif mode == "Upload CSV File":
    st.header("Upload a CSV file with customer data")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        try:
            processed = preprocess_input(data)
            predictions = model.predict(processed)
            probabilities = model.predict_proba(processed)[:, 1]
            data["Churn Prediction"] = np.where(predictions == 1, "Yes", "No")
            data["Churn Probability"] = (probabilities * 100).round(2).astype(str) + '%'
            st.success("Predictions generated successfully!")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error in processing file: {e}")
