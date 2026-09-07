#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import joblib
import pandas as pd
import numpy as np

class BiologicalAgeClock:
    def __init__(self, model_path='gb_bio_age_clock_pipeline.joblib'):
        """Loads the serialized pipeline asset directly into the application context."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Verify path configuration.")

        # Load the pipeline once into memory when the server boots
        self.pipeline = joblib.load(model_path)

        # Strict alignment features required by your model step
        self.model_features = ['LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 'BMXBMI', 'BPXOPLS1', 'EST_VO2_MAX']

    def calculate_vo2_max(self, age, gender, bmi, resting_pulse, daily_activity_mins):
        """Engineers the Non-Exercise Cardiorespiratory Fitness Proxy (Est_VO2_Max)."""
        # Map gender safely (Algorithm uses Male = 1, Female = 0)
        gender_binary = 1 if int(gender) == 1 else 0

        # Scale Physical Activity daily minutes to the 0-3 ordinal index
        if pd.isna(daily_activity_mins) or daily_activity_mins <= 0:
            pa_index = 0.0
        elif daily_activity_mins <= 30:
            pa_index = 1.0
        elif daily_activity_mins <= 120:
            pa_index = 2.0
        else:
            pa_index = 3.0

        # Core epidemiological formula execution
        est_vo2 = (
            56.363 + 
            (15.809 * gender_binary) - 
            (0.431 * float(age)) - 
            (0.396 * float(bmi)) - 
            (0.103 * float(resting_pulse)) + 
            (2.731 * pa_index)
        )
        return est_vo2

    def predict(self, raw_data: dict) -> dict:
        """
        Accepts a dictionary of raw form values, engineers features, 
        and extracts the biological age prediction.
        """
        # Extract base metrics safely
        age = raw_data.get('RIDAGEYR')
        gender = raw_data.get('RIAGENDR')
        bmi = raw_data.get('BMXBMI')
        pulse = raw_data.get('BPXOPLS1')
        act_mins = raw_data.get('PAD675', 0)

        # 1. Synthesize the missing VO2 Max Proxy feature
        vo2_proxy = self.calculate_vo2_max(age, gender, bmi, pulse, act_mins)

        # 2. Build input dataframe to align with expected scikit-learn training columns
        input_row = {
            'LBXGLU': [raw_data.get('LBXGLU')],
            'LBXIN': [raw_data.get('LBXIN')],
            'LBXHSCRP': [raw_data.get('LBXHSCRP')],
            'LBXGH': [raw_data.get('LBXGH')],
            'BMXBMI': [bmi],
            'BPXOPLS1': [pulse],
            'EST_VO2_MAX': [vo2_proxy]
        }
        df_input = pd.DataFrame(input_row)[self.model_features]

        # 3. Predict via loaded pipeline object (handles imputation/scaling seamlessly)

        ## SEPTEMBER 7  2026
        predicted_bio_age = float(self.pipeline.predict(df_input)[0])
        age_gap = predicted_bio_age - float(age)

        # Determine clinical evaluation string
        if age_gap > 2.0:
            status = "Accelerated"
        elif age_gap < -2.0:
            status = "Decelerated"
        else:
            status = "Normal"

        return {
            "chronological_age": age,
            "predicted_biological_age": round(predicted_bio_age, 2),
            "age_acceleration_gap": round(age_gap, 2),
            "aging_status": status,
            "calculated_vo2_max_proxy": round(vo2_proxy, 2)
        }



