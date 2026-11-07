#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
from my_package.BiologicalAgeClock import BiologicalAgeClock

model_path='gb_bio_age_clock_pipeline.joblib'
if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Verify path configuration.")

# Instantiate your web clock manager
clock = BiologicalAgeClock()

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
print(api_response)


# In[8]:


from my_package.BiologicalAgeClock import BiologicalAgeClock


# In[9]:


clock = BiologicalAgeClock(model_path='gb_bio_age_clock_pipeline.joblib')


# In[10]:


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
print(api_response)


# In[ ]:




