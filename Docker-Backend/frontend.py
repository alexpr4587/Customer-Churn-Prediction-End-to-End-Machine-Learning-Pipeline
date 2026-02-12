import streamlit as st
import requests
import os

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.title("Customer Churn Prediction")
st.markdown("Enter the customer's data to find out if they are at risk of leaving.")

API_URL = os.getenv("API_URL", "http://churn-api:8000")

# --- SIDEBAR (Inputs) ---
st.sidebar.header("Client Data")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior)", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
    partner = st.sidebar.selectbox("Does he have a partner?", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Does he have dependencies?", ["Yes", "No"])
    tenure = st.sidebar.slider("Months of stay (Tenure)", 0, 72, 12)
    
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.sidebar.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, value=29.85)
    total_charges = st.sidebar.text_input("Total Charges ($)", value="29.85")

    data = [{
        "customerID": "WEB-USER", # Dummy Value
        "gender": gender,
        "SeniorCitizen": senior_citizen,
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
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }]
    return data

input_data = user_input_features()

# --- Prediction Button ---
if st.button("Predict Churn Probability"):
    with st.spinner('Consulting the ML model...'):
        try:
            response = requests.post(f"{API_URL}/predict", json=input_data)
            
            if response.status_code == 200:
                result = response.json()[0]
                prob = result['churn_probability']
                risk = result['risk_category']
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Churn Probability", value=f"{prob*100:.1f}%")
                
                with col2:
                    if risk == "High Risk":
                        st.error(f"⚠️ {risk}")
                        st.markdown("**¡Atention!** This client requires and inmediate retention offer.")
                    else:
                        st.success(f"✅ {risk}")
                        st.markdown("This client seems to be secured and comfortable.")
                
                # Progress Bar
                st.progress(prob)
                
            else:
                st.error(f"Error in the API: {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Could not connect to the API in {API_URL}")
            st.error(str(e))
