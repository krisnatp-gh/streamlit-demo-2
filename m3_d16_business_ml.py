import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained model
with open('car_insurance_prediction.sav', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Car Insurance Claim Predictor", layout="centered")
st.title("🚘 Car Insurance Claim Prediction")

st.sidebar.header("Enter Customer Information")

# Sidebar input for features
credit_score = st.sidebar.slider("Credit Score", 0.05, 0.96, 0.52)
annual_mileage = st.sidebar.number_input("Annual Mileage", min_value=2000, max_value=22000, value=12000)
speeding_violations = st.sidebar.number_input("Speeding Violations", min_value=0, max_value=22, value=0)
past_accidents = st.sidebar.number_input("Past Accidents", min_value=0, max_value=15, value=0)

age = st.sidebar.selectbox("Age Group", ['65+', '16-25', '26-39', '40-64'])
gender = st.sidebar.selectbox("Gender", ['female', 'male'])
race = st.sidebar.selectbox("Race", ['majority', 'minority'])
driving_experience = st.sidebar.selectbox("Driving Experience", ['0-9y', '10-19y', '20-29y', '30y+'])
education = st.sidebar.selectbox("Education", ['high school', 'none', 'university'])
income = st.sidebar.selectbox("Income", ['upper class', 'poverty', 'working class', 'middle class'])
vehicle_ownership = st.sidebar.selectbox("Vehicle Ownership", ['nan', 'no', 'yes'])
vehicle_year = st.sidebar.selectbox("Vehicle Year", ['after 2015', 'before 2015'])
married = st.sidebar.selectbox("Married", ['no', 'yes'])
children = st.sidebar.selectbox("Has Children", ['yes', 'no'])
city = st.sidebar.selectbox("City", ['santa rosa', 'oviedo', 'san diego', 'baltimore'])
region = st.sidebar.selectbox("Region", ['west', 'south', 'nan'])
state = st.sidebar.selectbox("State", ['california', 'florida', 'nan', 'maryland'])

# Prepare input for prediction (raw DataFrame with feature names)
input_data = pd.DataFrame({
    'CREDIT_SCORE': [credit_score],
    'ANNUAL_MILEAGE': [annual_mileage],
    'SPEEDING_VIOLATIONS': [speeding_violations],
    'PAST_ACCIDENTS': [past_accidents],
    'AGE': [age],
    'GENDER': [gender],
    'RACE': [race],
    'DRIVING_EXPERIENCE': [driving_experience],
    'EDUCATION': [education],
    'INCOME': [income],
    'VEHICLE_OWNERSHIP': [vehicle_ownership],
    'VEHICLE_YEAR': [vehicle_year],
    'MARRIED': [married],
    'CHILDREN': [children],
    'CITY': [city],
    'REGION': [region],
    'STATE': [state]
})

# Display input for confirmation
st.subheader("Customer Input Preview")
st.write(input_data)

# Predict
if st.button("Predict Insurance Claim"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ This customer is likely to make a car insurance claim.")
    else:
        st.info("ℹ️ This customer is unlikely to make a car insurance claim.")


 # How to run in terminal: streamlit run day25.py
 # Use pipreqs/piplist to make requirements
