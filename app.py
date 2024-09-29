import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('cnf_status_model.pkl')

st.title("IRCTC Confirmation Status Predictor")

# Input fields for new booking details
train_no = st.number_input("Train No", min_value=10000, max_value=99999)
wl_position = st.number_input("WL Position", min_value=0)
quota = st.selectbox("Quota", ['General', 'Tatkal', 'Ladies', 'Senior Citizen'])
train_class = st.selectbox("Class", ['Sleeper', 'AC 3-Tier', 'AC 2-Tier', 'General'])
days_to_journey = st.number_input("Days to Journey", min_value=1)

# Map input fields to numerical values (assuming same LabelEncoder mappings)
quota_map = {'General': 0, 'Tatkal': 1, 'Ladies': 2, 'Senior Citizen': 3}
class_map = {'Sleeper': 0, 'AC 3-Tier': 1, 'AC 2-Tier': 2, 'General': 3}

# Prepare input data for model
input_data = np.array([[train_no, wl_position, quota_map[quota], class_map[train_class], days_to_journey]])

# Make a prediction when user clicks the button
if st.button("Predict Confirmation Status"):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.success("Prediction: Confirmed")
    else:
        st.error("Prediction: Waitlist")

