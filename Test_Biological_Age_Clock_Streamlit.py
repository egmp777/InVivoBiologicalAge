import os
from my_package.BiologicalAgeClock import BiologicalAgeClock
import streamlit as st

# Instantiate your web clock manager
clock = BiologicalAgeClock(model_path='gb_bio_age_clock_pipeline.joblib')

# Mock JSON request data mapping payload from an API request
mock_api_payload = {
    'RIDAGEYR': 52,     # 52 years old
    'RIAGENDR': 2,      # Female
    'BMXBMI': 24.1,     # BMI
    'BPXOPLS1': 68,     # Resting Heart Rate
    'PAD675': 60,       # 60 minutes of daily activity
    'LBXGLU': 92.5,     # Glucose
    'LBXIN': 5.1,       # Insulin
    'LBXHSCRP': 0.08,   # hs-CRP
    'LBXGH': 5.1        # HbA1c
}

# Run prediction
api_response = clock.predict(mock_api_payload)
print(api_response["chronological_age"])
st.write("Your chronological age is:\n")
st.write(api_response["chronological_age"])
st.write("Your predicted biological age is:\n")
st.write(api_response["predicted_biological_age"])
st.write("Your age acceleration gap is:\n")
st.write(api_response["age_acceleration_gap"])
st.write("Your aging status is:\n")
st.write(api_response["aging_status"])