import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
#model = pickle.load(open(r'C:\Users\yrangeg\Downloads\credit_model.pkl', 'rb'))
model = pickle.load(open('credit_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("🏦 Credit Default Prediction App (BFSI)")

st.markdown("Enter customer details to predict loan default risk.")

# User Inputs
age = st.slider("Age", 18, 75, 30)
income = st.number_input("Annual Income (in USD)", 1000, 1000000, 50000)
credit_score = st.slider("Credit Score", 300, 850, 650)
loan_amount = st.number_input("Loan Amount (USD)", 1000, 500000, 10000)
loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 60])
num_dependents = st.slider("Number of Dependents", 0, 5, 1)

# Prepare input for model
input_data = pd.DataFrame({
    'age': [age],
    'income': [income],
    'credit_score': [credit_score],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'num_dependents': [num_dependents]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"❌ High Risk of Default (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")
