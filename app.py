import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("model.pkl")

st.set_page_config(page_title="AI Loan Approval", layout="centered")
st.title("🏦 AI Loan Approval Prediction System")

# Inputs
name = st.text_input("Applicant Name")
income = st.number_input("Monthly Income (TZS)", 100_000, 5_000_000, 500_000)
loan_amount = st.number_input("Loan Amount Requested (TZS)", 50_000, 3_000_000, 200_000)
repayment_history = st.selectbox("Repayment History", ['Good', 'Average', 'Poor'])
existing_loans = st.number_input("Number of Existing Loans", 0, 5, 0)
age = st.number_input("Age", 18, 65, 25)

if st.button("Predict Loan Approval"):
    repayment_dict = {'Good':2,'Average':1,'Poor':0}
    repayment_encoded = repayment_dict[repayment_history]
    input_data = pd.DataFrame([[income, loan_amount, repayment_encoded, existing_loans, age]],
                              columns=['Income','Loan_Amount','Repayment_History','Existing_Loans','Age'])
    pred = model.predict(input_data)[0]
    result = "✅ Approved" if pred==1 else "❌ Declined"
    st.success(f"Prediction for {name}: {result}")
