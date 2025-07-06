import streamlit as st
import joblib
import numpy as np

model = joblib.load('loan_model.pkl')  


st.title("🏦 Loan Eligibility Predictor")

# Model selection
model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest"])

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term (in days)", value=360)
credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Encode helper
def encode(val, mapping):
    return mapping[val]

if st.button("Predict"):
    input_data = np.array([[
        encode(gender, {"Male":1, "Female":0}),
        encode(married, {"Yes":1, "No":0}),
        encode(education, {"Graduate":1, "Not Graduate":0}),
        encode(self_employed, {"Yes":1, "No":0}),
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        encode(credit_history, {"Good (1)":1, "Bad (0)":0}),
        encode(property_area, {"Urban":2, "Semiurban":1, "Rural":0})
    ]])

    prediction = model.predict(input_data)[0]


    # Show result
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved.")
