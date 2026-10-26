#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#########    OCTOBER 12 2026 ####################################
######### WALKNG BACK PROCESS OF PRODUCING THIS CODE ############


import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os
import streamlit as st
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import ElasticNetCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer


glu = pd.read_sas('GLU_L.xpt', format='xport')
demo = pd.read_sas('DEMO_L.xpt' ,format='xport')
ins = pd.read_sas('INS_L.xpt', format = 'xport')
crp = pd.read_sas('HSCRP_L.xpt', format = 'xport')
hba1 = pd.read_sas('GHB_L.xpt', format = 'xport' )
## Reading the blood pressure and body mass index data since to see if we can improve R2
bp = pd.read_sas('BPXO_L.xpt', format = 'xport')
bmi = pd.read_sas('BMX_L.xpt', format = 'xport')
paq = pd.read_sas('PAQ_L.xpt', format = 'xport')
exam = pd.read_sas('BPXO_L.xpt', format = 'xport')


glu.set_index('SEQN', inplace = True)
demo.set_index('SEQN', inplace = True)
ins.set_index('SEQN', inplace = True)
crp.set_index('SEQN', inplace = True)
hba1.set_index('SEQN', inplace = True)
bp.set_index('SEQN', inplace = True)
bmi.set_index('SEQN', inplace = True)
paq.set_index('SEQN', inplace = True)
print(paq.head())

exam.set_index('SEQN', inplace = True)
#print(exam.head())


## Dropping weight column since gluc dataset has it
ins.drop(columns=['WTSAF2YR'], inplace=True)
hba1.drop(columns = ['WTPH2YR'], inplace=True)

merged1 = pd.merge(glu, demo, on="SEQN", how="inner")
merged2 = pd.merge(ins, merged1, on="SEQN", how="inner")


merged3 = pd.merge(crp, merged2, on="SEQN", how="inner")
##print(merged3.info())

merged4 = pd.merge(bp, merged3, on="SEQN", how="inner" )
#print(merged4.info())

merged5 = pd.merge(bmi, merged4, on="SEQN", how="inner")
# print(merged5.info())


merged6 = pd.merge(paq, merged5, on="SEQN", how="inner")
# print(merged5.info())




dfs = [hba1 ,merged6]

df_merged = pd.concat(dfs, axis=1, join="inner").reset_index()
print("-----------------------------df_merged------------------------")
print(df_merged.info())




## Training a custom biological age clock using the August 2021–August 2023 NHANES

## Python Pipeline for Weighted ElasticNet / Ridge Clock Training
df_merged = df_merged[['SEQN', 'WTSAF2YR', 'RIDAGEYR', 'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 'SDMVPSU', 'SDMVSTRA',
                      'BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3', 'BMXBMI', 'BMXWAIST', 'BPXOPLS1', 
                       'RIAGENDR', 'PAD800']]
print(df_merged.info())


biomarkers = ['LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH']


model_vars = ['RIDAGEYR'] + biomarkers + ['WTSAF2YR']
clean_data = df_merged.dropna(subset=model_vars).copy()
clean_data
print(clean_data.info())


# In[ ]:


X = clean_data[biomarkers]
y = clean_data['RIDAGEYR']
weights = clean_data['WTSAF2YR']

# 3. Train a weighted ElasticNet clock using scikit-learn's sample_weight parameter
# Standardize features and fit with survey weights scaled to sample size
scaled_weights = weights / weights.mean() # Normalize weights for stable optimization convergence

###### FIRST CLOCK PIPELINE CODE SUGGESTED   ##########
####### THIS CLOCK PIPELINE WORKED     ##############


clock_pipeline = make_pipeline( 
    StandardScaler(),
    ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=42)
)

# Fit model incorporating survey weights
clock_pipeline.fit(X, y, elasticnetcv__sample_weight=scaled_weights)
clean_data['predicted_bio_age'] = clock_pipeline.predict(X)
clean_data['age_acceleration'] = clean_data['predicted_bio_age'] - clean_data['RIDAGEYR']

print(clean_data[['RIDAGEYR', 'predicted_bio_age', 'age_acceleration']].head())


# In[ ]:


print(clean_data.head())


# In[ ]:





# In[ ]:





# In[ ]:


print(clean_data[['RIDAGEYR', 'predicted_bio_age', 'age_acceleration']].head(100))


# In[ ]:


######### NOT WALKED BACK ###########
####joblib.dump(clock_pipeline, 'my_clock_pipeline1.joblib')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42
)


# In[ ]:


################ NOT WALKED BACK #################
y_pred_test = clock_pipeline.predict(X_test)


# In[ ]:


# 3. Define custom Weighted Evaluation Functions
def weighted_r2_score(y_true, y_pred, w):
    """Calculates survey-weighted R-squared."""
    w_mean = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - w_mean) ** 2)
    return 1 - (ss_res / ss_tot)

def weighted_rmse(y_true, y_pred, w):
    """Calculates survey-weighted Root Mean Squared Error."""
    return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

def weighted_mae(y_true, y_pred, w):
    """Calculates survey-weighted Mean Absolute Error."""
    return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)


# In[ ]:



import numpy as np
# 4. Print the rigorous survey-adjusted test metrics
print(f"Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
print(f"Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
print(f"Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")


# In[ ]:


#######  WE HAVE THE TWO FUNCTIONS TO PREPARE THE FEATURE SET 

######### INITIAL FUNCTION   ##########
# We need to improve R2 (closer to 1) with fearure engineering
from sklearn.compose import ColumnTransformer
# def engineer_nhanes_features(df):
#     """
#     Takes a raw DataFrame with LBXGLU, LBXIN, LBXHSCRP, LBXGH
#     and outputs an enhanced DataFrame with non-linear and interaction features.
#     """
#     df_eng = df.copy()
#
#     # Avoid division by zero or negative values in logs/ratios
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#     # Core Clinical Ratios & Indexes
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#
#     # Interaction terms representing systemic metabolic-inflammatory stress
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#
#     # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
#
#     ### Note: This will generate the following error KeyError: 'BPXDI1' b/c as explained
#     ### in the Gemini answer: As of August 2021 - August 2023,
#     ### NHANES transitioned to automated oscillometric blood pressure measurements
#     ### stored in variables in n BPXO_L (not BPX_L), and the readings are named BPXOSY1 / BPXODI1
#     df_eng['MAP'] = df_eng['BPXDI1'] + (1.0 / 3.0) * (df_eng['BPXSY1'] - df_eng['BPXDI1'])
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#
#     return df_eng
#
# #######    FIRST FEATURE LIST #################
# # Define the expanded feature list including the physical exam variables
# extended_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP',
#     'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1', 'MAP',
#     'BMI_X_WAIST', 'BMI_X_HOMA'
# ]
# # 1. Assume 'clean_data' has base variables and you applied your feature engineering function
# enhanced_data = engineer_nhanes_features(clean_data)
# X_eng = enhanced_data[engineered_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
# ######################################### OCTOBER 13 2026 TESTING THE R2 #############################
#
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
#
# # --- STEP B: Train / Test Split Alignment ---
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # --- STEP C: Build Stable, High-Capacity Pipeline ---
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), # Extracts structural pairings
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .3, .5, .7, .9, .95, 1],
#         max_iter=50000,   # Guarantees convergence on massive polynomial matrices
#         tol=1e-3,
#         random_state=42
#     )
# )
#
# # --- STEP D: Model Fit (With Survey Weights Normalized) ---
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # --- STEP E: Generate Predictions and Score via Weighted Matrix ---
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# # --- STEP F: Print performance results ---
# print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)
# #######################################################################################################

#########################################################OCTOBER 16 2026 TESTING THE engineer_nhanes_extended_features  #####################

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline

############################ ALREADY TESTED - GAVE THE KEY ERROR - OCTOBER 20 2026 ####################
# def engineer_nhanes_extended_features(df):
#     """
#     Expands the feature space to include laboratory markers,
#     anthropometrics (BMI, Waist), and vascular metrics (MAP).
#     """
#     df_eng = df.copy()
#
#     # Ensure safe non-zero/non-negative clipping
#     df_eng['LBXCRP_safe'] = df_eng['LBXCRP'].clip(lower=1e-3)
#
#     # 1. Original Lab Features & Interactions
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#
#     # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
#     ### Note: This will generate the following error KeyError: 'BPXDI1' b/c as explained
#     ### in the Gemini answer: As of August 2021 - August 2023,
#     ### NHANES transitioned to automated oscillometric blood pressure measurements
#     ### stored in variables in n BPXO_L (not BPX_L), and the readings are named BPXOSY1 / BPXODI1
#     df_eng['MAP'] = df_eng['BPXDI1'] + (1.0 / 3.0) * (df_eng['BPXSY1'] - df_eng['BPXDI1'])
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#
#     return df_eng
#
#
# # Define the expanded feature list including the physical exam variables
# extended_features = [
#     'LBXGLU', 'LBXIN', 'LBXCRP', 'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP',
#     'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1', 'MAP',
#     'BMI_X_WAIST', 'BMI_X_HOMA'
# ]
#
# # 1. Assume 'clean_data' has base variables and you applied your feature engineering function
# enhanced_data = engineer_nhanes_extended_features(clean_data)
# X_eng = enhanced_data[extended_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
# ######################################### OCTOBER 13 2026 TESTING THE R2 #############################
#
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
#
# # --- STEP B: Train / Test Split Alignment ---
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # --- STEP D: Model Fit (With Survey Weights Normalized) ---
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # --- STEP E: Generate Predictions and Score via Weighted Matrix ---
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# # 3. Stable Pipeline configuration with high iteration count for convergence
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .5, .7, .9, 1],
#         max_iter=50000,
#         tol=1e-3,
#         random_state=42
#     )
# )
#
#
# print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)
##############################################################################################################################################




############################ ALREADY TESTED - R2 WAS 0.4131 - OCTOBER 20 2026 ####################
########### SECOND FUNCTION ########
# import numpy as np
# def engineer_nhanes_comprehensive_clock(df):
#     """
#     Ingests lab metrics, body measures, and automated oscillometric vitals
#     from the NHANES 2021-2023 cycle to engineer an enhanced feature space.
#     """
#     df_eng = df.copy()
#
#     # 1. Handle Blood Pressure (Averaging the 2021-2023 BPXO readings)
#     sys_cols = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
#     dia_cols = ['BPXODI1', 'BPXODI2', 'BPXODI3']
#
#     df_eng['MEAN_SYS'] = df_eng[sys_cols].mean(axis=1)
#     df_eng['MEAN_DIA'] = df_eng[dia_cols].mean(axis=1)
#
#         # Derive Mean Arterial Pressure (MAP)
#     df_eng['MAP'] = df_eng['MEAN_DIA'] + (1.0 / 3.0) * (df_eng['MEAN_SYS'] - df_eng['MEAN_DIA'])
#
#     # Pulse Pressure (Stiffness Indicator)
#     df_eng['PULSE_PRESSURE'] = df_eng['MEAN_SYS'] - df_eng['MEAN_DIA']
#
#     # 2. Original Metabolic Panel Enhancements
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#
#     # 3. Cross-System Synergy Interactions (Metabolic x Adiposity x Vascular)
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#     df_eng['MAP_X_HOMA'] = df_eng['MAP'] * df_eng['HOMA_IR']
#     df_eng['MAP_X_LOGCRP'] = df_eng['MAP'] * df_eng['LOG_CRP']
#
#     return df_eng
#
#
# ######## SECOND FEATURE SET   ##########
# # Define the complete clinical array
# comprehensive_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',                     # Lab markers
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',                    # Lab ratios
#     'BMXBMI', 'BMXWAIST', 'MEAN_SYS', 'MEAN_DIA', 'MAP', 'PULSE_PRESSURE', # Vitals
#     'BMI_X_WAIST', 'BMI_X_HOMA', 'MAP_X_HOMA', 'MAP_X_LOGCRP' # Cross-system interactions
# ]
#
#
# df_merged = df_merged.dropna().copy()
# print(df_merged.head())
#
#
#
#
#
# # --- STEP A: Apply Feature Engineering ---
# # (Assumes your merged dataframe contains: BMXBMI, BMXWAIST, BPXOSY1..3, BPXODI1..3, etc.)
# enhanced_dataset = engineer_nhanes_comprehensive_clock(df_merged)
#
# X_comp = enhanced_dataset[comprehensive_features]
# y = enhanced_dataset['RIDAGEYR']
# weights = enhanced_dataset['WTSAF2YR']
#
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
#
# # --- STEP B: Train / Test Split Alignment ---
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_comp, y, weights, test_size=0.3, random_state=42
# )
#
# # --- STEP C: Build Stable, High-Capacity Pipeline ---
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), # Extracts structural pairings
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .3, .5, .7, .9, .95, 1],
#         max_iter=50000,   # Guarantees convergence on massive polynomial matrices
#         tol=1e-3,
#         random_state=42
#     )
# )
#
# # --- STEP D: Model Fit (With Survey Weights Normalized) ---
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # --- STEP E: Generate Predictions and Score via Weighted Matrix ---
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# # --- STEP F: Print performance results ---
# print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)


###################################  OCTOBER 22 2026 - REPLICATING: TRYING TO DETERMINE WHERE WE GOT THE 0.261 R2
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import ElasticNetCV
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
#
#
# # 1. Feature Engineering Function for NHANES Data
# def engineer_nhanes_features(df):
#     """
#     Takes a raw DataFrame with LBXGLU, LBXIN, LBXCRP, LBXGH
#     and outputs an enhanced DataFrame with non-linear and interaction features.
#     """
#     df_eng = df.copy()
#
#     # Avoid division by zero or negative values in logs/ratios
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#
#     # Core Clinical Ratios & Indexes
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#     sys_cols = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
#     dia_cols = ['BPXODI1', 'BPXODI2', 'BPXODI3']
#
#     # Interaction terms representing systemic metabolic-inflammatory stress
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#     df_eng['MEAN_DIA'] = df_eng[dia_cols].mean(axis=1)
#     df_eng['MEAN_SYS'] = df_eng[sys_cols].mean(axis=1)
#
#     # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
#
#     df_eng['MAP'] = df_eng['MEAN_DIA'] + (1.0 / 3.0) * (df_eng['MEAN_SYS'] - df_eng['MEAN_DIA'])
#
#
#
#     # Below line produces an error. The correct line is above
#     # df_eng['MAP'] = df_eng['BPXDI1'] + (1.0 / 3.0) * (df_eng['BPXSY1'] - df_eng['BPXDI1'])
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#
#     return df_eng
#
#
#
# # Define updated feature set including the new engineered parameters
# engineered_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP',  'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP'
# ]
#
#
# extended_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 'MEAN_DIA',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP',
#     'BMXBMI', 'BMXWAIST',  'MAP',
#     'BMI_X_WAIST', 'BMI_X_HOMA'
# ]
#
#
# # 3. Build the Improved Machine Learning Pipeline
# improved_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
#     # Captures pairwise interactions automatically
#     ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, 1], max_iter=50000, random_state=42)
# )
#
#
#
# # 1. Assume 'clean_data' has base variables and you applied your feature engineering function
# enhanced_data = engineer_nhanes_features(df_merged)
# X_eng = enhanced_data[extended_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
#
# # 2. Split data into train and test sets, keeping sample weights aligned
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # 3. Fit the pipeline on training data with normalized sample weights
# scaled_train_weights = w_train / w_train.mean()
# improved_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # 4. Generate predictions on the test set
# y_pred_test = improved_clock_pipeline.predict(X_test)
#
# # 5. Define Weighted Evaluation Metrics
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# def weighted_mae(y_true, y_pred, w):
#     return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)
#
# # 6. Print out the final weighted test scores
# print(f"Engineered Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Engineered Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
# print(f"Engineered Model Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)



# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import ElasticNetCV
# from sklearn.pipeline import make_pipeline
#
#
# def engineer_nhanes_comprehensive_clock(df):
#     """
#     Ingests lab metrics, body measures, and automated oscillometric vitals
#     from the NHANES 2021-2023 cycle to engineer an enhanced feature space.
#     """
#     df_eng = df.copy()
#
#     # 1. Handle Blood Pressure (Averaging the 2021-2023 BPXO readings)
#     sys_cols = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
#     dia_cols = ['BPXODI1', 'BPXODI2', 'BPXODI3']
#
#     df_eng['MEAN_SYS'] = df_eng[sys_cols].mean(axis=1)
#     df_eng['MEAN_DIA'] = df_eng[dia_cols].mean(axis=1)
#
#     # Derive Mean Arterial Pressure (MAP)
#     df_eng['MAP'] = df_eng['MEAN_DIA'] + (1.0 / 3.0) * (df_eng['MEAN_SYS'] - df_eng['MEAN_DIA'])
#
#     # Pulse Pressure (Stiffness Indicator)
#     df_eng['PULSE_PRESSURE'] = df_eng['MEAN_SYS'] - df_eng['MEAN_DIA']
#
#     # 2. Original Metabolic Panel Enhancements
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#
#     # 3. Cross-System Synergy Interactions (Metabolic x Adiposity x Vascular)
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#     df_eng['MAP_X_HOMA'] = df_eng['MAP'] * df_eng['HOMA_IR']
#     df_eng['MAP_X_LOGCRP'] = df_eng['MAP'] * df_eng['LOG_CRP']
#
#     return df_eng
#
#
# # Define the complete clinical array
# comprehensive_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',  # Lab markers
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',  # Lab ratios
#     'BMXBMI', 'BMXWAIST', 'MEAN_SYS', 'MEAN_DIA', 'MAP', 'PULSE_PRESSURE',  # Vitals
#     'BMI_X_WAIST', 'BMI_X_HOMA', 'MAP_X_HOMA', 'MAP_X_LOGCRP'  # Cross-system interactions
# ]
#
# df_merged = df_merged.dropna().copy()
#
# # --- STEP A: Apply Feature Engineering ---
# # (Assumes your merged dataframe contains: BMXBMI, BMXWAIST, BPXOSY1..3, BPXODI1..3, etc.)
# enhanced_dataset = engineer_nhanes_comprehensive_clock(df_merged)
#
# X_comp = enhanced_dataset[comprehensive_features]
# y = enhanced_dataset['RIDAGEYR']
# weights = enhanced_dataset['WTSAF2YR']
#
# # --- STEP B: Train / Test Split Alignment ---
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_comp, y, weights, test_size=0.3, random_state=42
# )
#
# # --- STEP C: Build Stable, High-Capacity Pipeline ---
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), # Extracts structural pairings
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .3, .5, .7, .9, .95, 1],
#         max_iter=50000,   # Guarantees convergence on massive polynomial matrices
#         tol=1e-3,
#         random_state=42
#     )
# )
#
# # --- STEP D: Model Fit (With Survey Weights Normalized) ---
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # --- STEP E: Generate Predictions and Score via Weighted Matrix ---
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# # --- STEP F: Print performance results ---
# print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")


#################################### OCTOBER 23 2026   ####################################################
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import ElasticNetCV
# from sklearn.pipeline import make_pipeline
# from sklearn.compose import ColumnTransformer
#
#
# # 1. Feature Engineering Function for NHANES Data
# def engineer_nhanes_features(df):
#     """
#     Takes a raw DataFrame with LBXGLU, LBXIN, LBXCRP, LBXGH
#     and outputs an enhanced DataFrame with non-linear and interaction features.
#     """
#     df_eng = df.copy()
#
#     # Avoid division by zero or negative values in logs/ratios
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#
#     # Core Clinical Ratios & Indexes
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#     # Define updated feature set including the new engineered parameters
#
#     return df_eng
#
# # Define updated feature set including the new engineered parameters
# engineered_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP'
# ]
#
# df_merged = df_merged.dropna().copy()
#
# # 3. Build the Improved Machine Learning Pipeline
# improved_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
#     # Captures pairwise interactions automatically
#     ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, 1],  max_iter=50000, random_state=42)
# )
#
# enhanced_data = engineer_nhanes_features(df_merged)
# X_eng = enhanced_data[engineered_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
# # 2. Split data into train and test sets, keeping sample weights aligned
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # 3. Fit the pipeline on training data with normalized sample weights
# scaled_train_weights = w_train / w_train.mean()
# improved_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # 4. Generate predictions on the test set
# y_pred_test = improved_clock_pipeline.predict(X_test)
#
# # 5. Define Weighted Evaluation Metrics
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# def weighted_mae(y_true, y_pred, w):
#     return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)
#
# # 6. Print out the final weighted test scores
# print(f"Engineered Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Engineered Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
# print(f"Engineered Model Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)


########################################   GOT R2 26.01 !!!!!  OCTOBER 23 2026 MORE SEARCHING FOR THE 0.26 R2    ###########################
# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import ElasticNetCV
# from sklearn.pipeline import make_pipeline
#
#
# def engineer_nhanes_extended_features(df):
#     """
#     Expands the feature space to include laboratory markers,
#     anthropometrics (BMI, Waist), and vascular metrics (MAP).
#     """
#     df_eng = df.copy()
#
#     # Ensure safe non-zero/non-negative clipping
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#
#     # 1. Original Lab Features & Interactions
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#
#     # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
#     df_eng['MAP'] = df_eng['BPXODI1'] + (1.0 / 3.0) * (df_eng['BPXOSY1'] - df_eng['BPXODI1'])
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#
#     return df_eng
#
#
# # Define the expanded feature list including the physical exam variables
# extended_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP',
#     'BMXBMI', 'BMXWAIST', 'BPXOSY1', 'BPXODI1', 'MAP',
#     'BMI_X_WAIST', 'BMI_X_HOMA'
# ]
#
# # 3. Stable Pipeline configuration with high iteration count for convergence
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .5, .7, .9, 1],
#         max_iter=50000,
#         tol=1e-3,
#         random_state=42
#     )
# )
#
# ###### OCTOBER 27 2026 #############################
# ###################### ELIMINATING NAN VALUES FROM THE DATASET R2 was increased to R 0.3753 ######################
# ##clean_data = df_merged.dropna(subset=model_vars).copy()
# ##df_merged = df_merged.dropna().copy()
# ###################### ELIMINATING NAN VALUES FROM THE DATASET ######################
#
#
# enhanced_data = engineer_nhanes_extended_features(df_merged)
# X_eng = enhanced_data[extended_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
# # 2. Split data into train and test sets, keeping sample weights aligned
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # 3. Fit the pipeline on training data with normalized sample weights
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # 4. Generate predictions on the test set
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# # 5. Define Weighted Evaluation Metrics
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# def weighted_mae(y_true, y_pred, w):
#     return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)
#
# # 6. Print out the final weighted test scores
# print(f"Engineered Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Engineered Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
# print(f"Engineered Model Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)
#
##################################################################################################################################


##############################################################   OCTOBER 24 2026  0.37 R2   ###############################################

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.linear_model import ElasticNetCV
# from sklearn.pipeline import make_pipeline
#
#
# def engineer_nhanes_extended_features(df):
#     """
#     Expands the feature space to include laboratory markers,
#     anthropometrics (BMI, Waist), and vascular metrics (MAP).
#     """
#     df_eng = df.copy()
#
#     # Ensure safe non-zero/non-negative clipping
#     df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
#
#     # 1. Original Lab Features & Interactions
#     df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
#     df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
#     df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)
#     df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
#     df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']
#
#     # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
#     df_eng['MAP'] = df_eng['BPXODI1'] + (1.0 / 3.0) * (df_eng['BPXOSY1'] - df_eng['BPXODI1'])
#     df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
#     df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
#
#     return df_eng
#
#
# # Define the expanded feature list including the physical exam variables
# extended_features = [
#     'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',
#     'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',
#     'GLU_X_CRP', 'INSU_X_CRP',
#     'BMXBMI', 'BMXWAIST', 'BPXOSY1', 'BPXODI1', 'MAP',
#     'BMI_X_WAIST', 'BMI_X_HOMA'
# ]
#
# # 3. Stable Pipeline configuration with high iteration count for convergence
# extended_clock_pipeline = make_pipeline(
#     SimpleImputer(strategy='median'),
#     StandardScaler(),
#     PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
#     ElasticNetCV(
#         cv=5,
#         l1_ratio=[.1, .5, .7, .9, 1],
#         max_iter=50000,
#         tol=1e-3,
#         random_state=42
#     )
# )
#
# clean_data = df_merged.dropna(subset=model_vars).copy()
# df_merged = df_merged.dropna().copy()
# enhanced_data = engineer_nhanes_extended_features(df_merged)
# X_eng = enhanced_data[extended_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']
#
# # 2. Split data into train and test sets, keeping sample weights aligned
# X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
#     X_eng, y, weights, test_size=0.3, random_state=42
# )
#
# # 3. Fit the pipeline on training data with normalized sample weights
# scaled_train_weights = w_train / w_train.mean()
# extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)
#
# # 4. Generate predictions on the test set
# y_pred_test = extended_clock_pipeline.predict(X_test)
#
# # 5. Define Weighted Evaluation Metrics
# def weighted_r2_score(y_true, y_pred, w):
#     w_mean = np.average(y_true, weights=w)
#     ss_res = np.sum(w * (y_true - y_pred) ** 2)
#     ss_tot = np.sum(w * (y_true - w_mean) ** 2)
#     return 1 - (ss_res / ss_tot)
#
# def weighted_rmse(y_true, y_pred, w):
#     return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))
#
# def weighted_mae(y_true, y_pred, w):
#     return np.sum(w * np.abs(y_true - y_pred)) / np.sum(w)
#
# # 6. Print out the final weighted test scores
# print(f"Engineered Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
# print(f"Engineered Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
# print(f"Engineered Model Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")
#
# exit(0)

##################################################################################################################################