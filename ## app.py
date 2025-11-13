# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# App title
st.title("Customer Churn Prediction")

# Input from user
st.header("Enter customer details")
tenure = st.number_input("Tenure (months)")
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

# Predict button
if st.button("Predict Churn"):
    data = pd.DataFrame([[tenure, monthly_charges, total_charges]], 
                        columns=["tenure", "monthly_charges", "total_charges"])
    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("Customer is likely to churn 😢")
    else:
        st.success("Customer is likely to stay 🙂")