import requests
import streamlit as st
import yaml  # type: ignore

# Load config
with open("configs/app.yaml") as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title=config["streamlit"]["page_title"],
    layout=config["streamlit"]["layout"]
)

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn risk.")

# Use local API for zero latency
API_URL = "http://localhost:8000/predict"

# Layout with columns for a cleaner look
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])

with col3:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

st.markdown("---")

if st.button("Predict Churn", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)

        if response.status_code == 200:
            result = response.json()

            # Display results in a visually appealing way
            res_col1, res_col2, res_col3 = st.columns(3)

            prob = result['churn_probability']
            risk = result['risk_level']

            res_col1.metric("Churn Probability", f"{prob * 100:.1f}%")
            res_col2.metric("Risk Level", risk)
            res_col3.metric("Recommended Action", result["recommended_action"])

            if risk == "High":
                st.error("High risk of churn! Immediate action required.")
            elif risk == "Medium":
                st.warning("Medium risk. Consider retention offers.")
            else:
                st.success("Low risk. Customer is likely to stay.")

        else:
            st.error(f"API Error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure 'python scripts/serve.py' is running!")
