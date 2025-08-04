import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('car_insurance_prediction.sav', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Car Insurance Claim Prediction", layout="centered")

st.title("🚗 Car Insurance Claim Prediction")

st.write(
    "Fill in the details in the sidebar to predict the likelihood of a car insurance claim."
)

# Sidebar for user input
st.sidebar.header("Input Features")

credit_score = st.sidebar.slider(
    "Credit Score (normalized)", 0.05, 0.96, 0.52, 0.01
)
annual_mileage = st.sidebar.number_input(
    "Annual Mileage", min_value=2000, max_value=22000, value=12000, step=500
)
speeding_violations = st.sidebar.number_input(
    "Speeding Violations (last 5 years)", min_value=0, max_value=22, value=1, step=1
)
past_accidents = st.sidebar.number_input(
    "Past Accidents", min_value=0, max_value=15, value=1, step=1
)

age = st.sidebar.selectbox(
    "Age Group", ["16-25", "26-39", "40-64", "65+"]
)
gender = st.sidebar.selectbox(
    "Gender", ["male", "female"]
)
race = st.sidebar.selectbox(
    "Race", ["majority", "minority"]
)
driving_experience = st.sidebar.selectbox(
    "Driving Experience", ["0-9y", "10-19y", "20-29y", "30y+"]
)
education = st.sidebar.selectbox(
    "Education", ["none", "high school", "university"]
)
income = st.sidebar.selectbox(
    "Income", ["poverty", "working class", "middle class", "upper class"]
)
vehicle_ownership = st.sidebar.selectbox(
    "Vehicle Ownership", ["yes", "no", "nan"]
)
vehicle_year = st.sidebar.selectbox(
    "Vehicle Year", ["before 2015", "after 2015"]
)
married = st.sidebar.selectbox(
    "Married", ["yes", "no"]
)
children = st.sidebar.selectbox(
    "Children", ["yes", "no"]
)
city = st.sidebar.selectbox(
    "City", ["santa rosa", "oviedo", "san diego", "baltimore"]
)
region = st.sidebar.selectbox(
    "Region", ["west", "south", "nan"]
)
state = st.sidebar.selectbox(
    "State", ["california", "florida", "maryland", "nan"]
)

# Prepare input for prediction
input_data = [
    credit_score, annual_mileage, speeding_violations, past_accidents,
    age, gender, race, driving_experience, education, income,
    vehicle_ownership, vehicle_year, married, children, city, region, state
]

# Model expects a 2D array
def predict_claim(input_data):
    # Model handles preprocessing, so just pass as is
    input_array = np.array([input_data], dtype=object)
    prediction = model.predict(input_array)
    return prediction[0]

if st.button("Predict Claim"):
    result = predict_claim(input_data)
    if result == 1:
        st.success("⚠️ The model predicts a **claim is likely**.")
    else:
        st.info("✅ The model predicts a **claim is unlikely**.")

st.markdown("---")
st.caption("Model: car_insurance_prediction.sav | Powered by Streamlit")