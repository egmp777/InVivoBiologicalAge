import os
import streamlit as st
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder


from my_package.BiologicalAgeClock import BiologicalAgeClock

# Instantiate your web clock manager
clock = BiologicalAgeClock(model_path='gb_bio_age_clock_pipeline.joblib')

RIDAGEY = st.number_input("Enter your age: ")
RIAGENDR = st.number_input("Enter your gender (1 = male, 2 = female): ")
BMXBMI = st.number_input("Enter your Body Max Index: ")
BPXOPLS1 = st.number_input("Enter your Resting Heart Rate: ")
PAD675 = st.number_input("Enter how many minutes of daily physical activity you do: ")
LBXGLU = st.number_input("Enter your glucose level: ")
LBXIN = st.number_input("Enter your insulin level: ")
LBXHSCRP = st.number_input("Enter your CRP reading: ")
LBXGH = st.number_input("Enter your HbA1c reading: ")


api_payload = {
    'RIDAGEYR': RIDAGEY,     # Age
    'RIAGENDR': RIAGENDR,      # Gender
    'BMXBMI': BMXBMI,     # BMI
    'BPXOPLS1': BPXOPLS1,     # Resting Heart Rate
    'PAD675': PAD675,       # Minutes of daily activity
    'LBXGLU': LBXGLU,     # Glucose
    'LBXIN': LBXIN,       # Insulin
    'LBXHSCRP': LBXHSCRP,   # hs-CRP
    'LBXGH': LBXGH        # HbA1c
}

if st.button("Submit"):

    if api_payload == {} or api_payload == []:
        print("Payload is empty")  # Ensure all fields are filled
    else:
        api_response = clock.predict(api_payload)
        st.write("Your Chronological age is:\n")
        st.write(api_response["chronological_age"])
        st.write("Your predicted biological age is:\n")
        st.write(api_response["predicted_biological_age"])
        st.write("Your age acceleration gap is:\n")
        st.write(api_response["age_acceleration_gap"])
        st.write("Your aging status is:\n")
        st.write(api_response["aging_status"])




