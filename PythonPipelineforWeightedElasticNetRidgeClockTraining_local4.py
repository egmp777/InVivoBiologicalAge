#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


X = clean_data[biomarkers]
y = clean_data['RIDAGEYR']
weights = clean_data['WTSAF2YR']

# 3. Train a weighted ElasticNet clock using scikit-learn's sample_weight parameter
# Standardize features and fit with survey weights scaled to sample size
scaled_weights = weights / weights.mean() # Normalize weights for stable optimization convergence

clock_pipeline = make_pipeline( StandardScaler(),
    ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], random_state=42)
)

# Fit model incorporating survey weights
clock_pipeline.fit(X, y, elasticnetcv__sample_weight=scaled_weights)
clean_data['predicted_bio_age'] = clock_pipeline.predict(X)
clean_data['age_acceleration'] = clean_data['predicted_bio_age'] - clean_data['RIDAGEYR']

print(clean_data[['RIDAGEYR', 'predicted_bio_age', 'age_acceleration']].head())


# In[4]:


print(clean_data.head())


# In[ ]:





# In[ ]:





# In[5]:


print(clean_data[['RIDAGEYR', 'predicted_bio_age', 'age_acceleration']].head(100))


# In[6]:


joblib.dump(clock_pipeline, 'my_clock_pipeline1.joblib')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42
)


# In[9]:


y_pred_test = clock_pipeline.predict(X_test)


# In[10]:


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


# In[11]:


import numpy as np
# 4. Print the rigorous survey-adjusted test metrics
print(f"Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
print(f"Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")
print(f"Weighted MAE:  {weighted_mae(y_test, y_pred_test, w_test):.4f} years")


# In[12]:


# We need to improve R2 (closer to 1) with fearure engineering
from sklearn.compose import ColumnTransformer
def engineer_nhanes_features(df):
    """
    Takes a raw DataFrame with LBXGLU, LBXIN, LBXHSCRP, LBXGH 
    and outputs an enhanced DataFrame with non-linear and interaction features.
    """
    df_eng = df.copy()

    # Avoid division by zero or negative values in logs/ratios
    df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
    # Core Clinical Ratios & Indexes
    df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
    df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
    df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)

    # Interaction terms representing systemic metabolic-inflammatory stress
    df_eng['GLU_X_CRP'] = df_eng['LBXGLU'] * df_eng['LOG_CRP']
    df_eng['INSU_X_CRP'] = df_eng['LBXIN'] * df_eng['LOG_CRP']

    # 2. Physical Exam Transformations (MAP Calculation: Diastolic + 1/3 Pulse Pressure)
    df_eng['MAP'] = df_eng['BPXDI1'] + (1.0 / 3.0) * (df_eng['BPXSY1'] - df_eng['BPXDI1'])
    df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
    df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']

    return df_eng

# Define the expanded feature list including the physical exam variables
extended_features = [
    'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 
    'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO', 
    'GLU_X_CRP', 'INSU_X_CRP',
    'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1', 'MAP', 
    'BMI_X_WAIST', 'BMI_X_HOMA'
]
# 1. Assume 'clean_data' has base variables and you applied your feature engineering function
# enhanced_data = engineer_nhanes_features(clean_data)
# X_eng = enhanced_data[engineered_features]
# y = enhanced_data['RIDAGEYR']
# weights = enhanced_data['WTSAF2YR']

import numpy as np
def engineer_nhanes_comprehensive_clock(df):
    """
    Ingests lab metrics, body measures, and automated oscillometric vitals
    from the NHANES 2021-2023 cycle to engineer an enhanced feature space.
    """
    df_eng = df.copy()

    # 1. Handle Blood Pressure (Averaging the 2021-2023 BPXO readings)
    sys_cols = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3']
    dia_cols = ['BPXODI1', 'BPXODI2', 'BPXODI3']

    df_eng['MEAN_SYS'] = df_eng[sys_cols].mean(axis=1)
    df_eng['MEAN_DIA'] = df_eng[dia_cols].mean(axis=1)

        # Derive Mean Arterial Pressure (MAP)
    df_eng['MAP'] = df_eng['MEAN_DIA'] + (1.0 / 3.0) * (df_eng['MEAN_SYS'] - df_eng['MEAN_DIA'])

    # Pulse Pressure (Stiffness Indicator)
    df_eng['PULSE_PRESSURE'] = df_eng['MEAN_SYS'] - df_eng['MEAN_DIA']

    # 2. Original Metabolic Panel Enhancements
    df_eng['LBXCRP_safe'] = df_eng['LBXHSCRP'].clip(lower=1e-3)
    df_eng['HOMA_IR'] = (df_eng['LBXGLU'] * df_eng['LBXIN']) / 405.0
    df_eng['LOG_CRP'] = np.log(df_eng['LBXCRP_safe'])
    df_eng['GLU_A1C_RATIO'] = df_eng['LBXGLU'] / (df_eng['LBXGH'] + 1e-3)

    # 3. Cross-System Synergy Interactions (Metabolic x Adiposity x Vascular)
    df_eng['BMI_X_WAIST'] = df_eng['BMXBMI'] * df_eng['BMXWAIST']
    df_eng['BMI_X_HOMA'] = df_eng['BMXBMI'] * df_eng['HOMA_IR']
    df_eng['MAP_X_HOMA'] = df_eng['MAP'] * df_eng['HOMA_IR']
    df_eng['MAP_X_LOGCRP'] = df_eng['MAP'] * df_eng['LOG_CRP']

    return df_eng


# Define the complete clinical array
comprehensive_features = [
    'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',                     # Lab markers
    'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',                    # Lab ratios
    'BMXBMI', 'BMXWAIST', 'MEAN_SYS', 'MEAN_DIA', 'MAP', 'PULSE_PRESSURE', # Vitals
    'BMI_X_WAIST', 'BMI_X_HOMA', 'MAP_X_HOMA', 'MAP_X_LOGCRP' # Cross-system interactions
]


df_merged = df_merged.dropna().copy()
print(df_merged.head())



# --- STEP A: Apply Feature Engineering ---
# (Assumes your merged dataframe contains: BMXBMI, BMXWAIST, BPXOSY1..3, BPXODI1..3, etc.)
enhanced_dataset = engineer_nhanes_comprehensive_clock(df_merged)

X_comp = enhanced_dataset[comprehensive_features]
y = enhanced_dataset['RIDAGEYR']
weights = enhanced_dataset['WTSAF2YR']

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# --- STEP B: Train / Test Split Alignment ---
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_comp, y, weights, test_size=0.3, random_state=42
)

# --- STEP C: Build Stable, High-Capacity Pipeline ---
extended_clock_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), # Extracts structural pairings
    ElasticNetCV(
        cv=5, 
        l1_ratio=[.1, .3, .5, .7, .9, .95, 1], 
        max_iter=50000,   # Guarantees convergence on massive polynomial matrices
        tol=1e-3, 
        random_state=42
    )
)

# --- STEP D: Model Fit (With Survey Weights Normalized) ---
scaled_train_weights = w_train / w_train.mean()
extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)

# --- STEP E: Generate Predictions and Score via Weighted Matrix ---
y_pred_test = extended_clock_pipeline.predict(X_test)

def weighted_r2_score(y_true, y_pred, w):
    w_mean = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - w_mean) ** 2)
    return 1 - (ss_res / ss_tot)

def weighted_rmse(y_true, y_pred, w):
    return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

# --- STEP F: Print performance results ---
print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")


# In[13]:


# 2. Engineer the Non-Exercise Cardiorespiratory Fitness Proxy (Est_VO2_Max)
def add_vo2_proxy(df):
    df_eng = df.copy()

    # Map gender safely (Algorithm uses Male = 1, Female = 0)
    gender_binary = (df_eng['RIAGENDR'] == 1).astype(int)
    # Simple categorical scaling for Physical Activity Index (0 to 3 scale)
    pa_index = pd.cut(df_eng['PAD800'], bins=[-1, 1, 31, 120], labels=[0, 1, 2]).astype(float).fillna(0)
    # Calculate physiological proxy
    df_eng['EST_VO2_MAX'] = (
        56.363 + 
        (15.809 * gender_binary) - 
        (0.431 * df_eng['RIDAGEYR']) - 
        (0.396 * df_eng['BMXBMI']) - 
        (0.103 * df_eng['BPXOPLS1']) + 
        (2.731 * pa_index)
    )

    return df_eng

##enhanced_dataset['RIAGENDR'] = demo['RIAGENDR']
##enhanced_dataset['PAD800'] = paq['PAD800']
##enhanced_dataset['BPXOPLS1'] = exam['BPXOPLS1']
##print(enhanced_dataset['BPXOPLS1'].head())
##print(df_merged['BPXOPLS1'].head())

enhanced_dataset['RIAGENDR'] = df_merged['RIAGENDR']
enhanced_dataset['PAD800'] = df_merged['PAD800']
enhanced_dataset['BPXOPLS1'] = df_merged['BPXOPLS1']
print(enhanced_dataset['BPXOPLS1'].head())




processed_df = add_vo2_proxy(enhanced_dataset)

# Define the complete clinical array
comprehensive_features = [
    'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH',                     # Lab markers
    'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO',                    # Lab ratios
    'BMXBMI', 'BMXWAIST', 'MEAN_SYS', 'MEAN_DIA', 'MAP', 'PULSE_PRESSURE', # Vitals
    'BMI_X_WAIST', 'BMI_X_HOMA', 'MAP_X_HOMA', 'MAP_X_LOGCRP', 'EST_VO2_MAX'# Cross-system interactions
]





# --- STEP A: Apply Feature Engineering ---
# (Assumes your merged dataframe contains: BMXBMI, BMXWAIST, BPXOSY1..3, BPXODI1..3, etc.)

X_comp = processed_df[comprehensive_features]

y = processed_df['RIDAGEYR']
weights = processed_df['WTSAF2YR']

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# In[14]:


# --- STEP D: Model Fit (With Survey Weights Normalized) ---
scaled_train_weights = w_train / w_train.mean()
extended_clock_pipeline.fit(X_train, y_train, elasticnetcv__sample_weight=scaled_train_weights)

# --- STEP E: Generate Predictions and Score via Weighted Matrix ---
y_pred_test = extended_clock_pipeline.predict(X_test)

def weighted_r2_score(y_true, y_pred, w):
    w_mean = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - w_mean) ** 2)
    return 1 - (ss_res / ss_tot)

def weighted_rmse(y_true, y_pred, w):
    return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

# --- STEP F: Print performance results ---
print(f"Extended Model Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
print(f"Extended Model Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


xgb_features = ['LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 'BMXBMI', 'BPXOPLS1', 'EST_VO2_MAX']
X = processed_df[xgb_features]
y = processed_df['RIDAGEYR']
weights = processed_df['WTSAF2YR']
#print(processed_df[processed_df.isna().any(axis=1)])



# 4. Train/Test Alignment
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights, test_size=0.3, random_state=42
)





# 3. Assemble the Preprocessing Transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), # Fills clinical missingness
            ('scaler', StandardScaler())                  # Normalizes numeric scales
        ]), xgb_features)
    ],
    remainder='drop'
)


# 4. Define the Unified GradientBoosting Pipeline
gb_clock_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=100,      # Number of sequential boosting trees
        learning_rate=0.05,    # Step size shrinkage to prevent overfitting
        max_depth=4,           # Limits individual tree complexity
        subsample=0.8,         # Stochastic boosting (fraction of samples to use per tree)
        random_state=42
    ))
])



# 5. Fit the model using scaled NHANES survey weights passed into the regressor step
scaled_train_weights = w_train / w_train.mean()
gb_clock_pipeline.fit(
    X_train, 
    y_train, 
    regressor__sample_weight=scaled_train_weights # Injects complex survey design weighting
)

# 6. Generate Predictions on the Holdout Set
y_pred_test = gb_clock_pipeline.predict(X_test)

# 7. Evaluate the New Pipeline using Weighted Metrics
def weighted_r2_score(y_true, y_pred, w):
    w_mean = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - w_mean) ** 2)
    return 1 - (ss_res / ss_tot)

def weighted_rmse(y_true, y_pred, w):
    return np.sqrt(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

print(f"GradientBoosting Weighted R²:   {weighted_r2_score(y_test, y_pred_test, w_test):.4f}")
print(f"GradientBoosting Weighted RMSE: {weighted_rmse(y_test, y_pred_test, w_test):.4f} years")


# In[16]:


## Conclusion: ElasticCV algorithm has a better R2 tha the  GradientBoostingRegressor 
## TODO: Fix the 64 bit Python req to e able to test the XBoost algorithm


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#---------------------- | Aug 22
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


from sklearn.inspection import permutation_importance




# Extract your fitted regressor step from your pipeline
regressor = gb_clock_pipeline.named_steps['regressor']

# Transform your validation data using the steps preceding the regressor
X_val_transformed = gb_clock_pipeline[:-1].transform(X_val)

# Compute the robust permutation feature importances
result = permutation_importance(regressor, X_val_transformed, y_val, n_repeats=10, random_state=42)

# Extract your true feature importance array safely
stable_features = result.importances_mean
print(stable_features)
#------------  
# 1. Extract the actual feature importances directly from the model step
importances = gb_clock_pipeline.named_steps['regressor'].feature_importances_

# ----------------- | Aug 22
importances = stable_features

# ----------------- |
print(importances)

# 2. Automatically extract ALL feature names from ALL steps prior to the regressor
# 'gb_clock_pipeline[:-1]' automatically takes everything except the final model step
preprocessing_slice = gb_clock_pipeline[:-1]
raw_feature_names = preprocessing_slice.get_feature_names_out()

# 3. Clean up formatting (removes scikit-learn step prefixes like 'polynomialfeatures__')
clean_feature_names = [name.split('__')[-1].replace(" ", " × ") for name in raw_feature_names]

# 4. Rigorous Debug Checklist: Print sizes to verify absolute alignment
print(f"--- Alignment Verification ---")
print(f"Total calculated features from pipeline: {len(clean_feature_names)}")
print(f"Total importances from GradientBoosting: {len(importances)}")

# 5. Build the final sorting DataFrame 
# Because both arrays come directly from the pipeline object, they align perfectly.
importance_df = pd.DataFrame({
    'Feature': clean_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 6. Render the clean horizontal bar plot
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Plot only the top 15 features to keep the visual clean and readable
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=importance_df.head(15), 
    palette='magma',
    hue='Feature',
    legend=False
)

plt.title('Final GradientBoosting Clock: Top 15 Feature Importances (R² = 0.70)', fontsize=14, pad=15)
plt.xlabel('Relative Importance (Split Weight)', fontsize=12)
plt.ylabel('Engineered Feature / Interaction Pair', fontsize=12)
plt.tight_layout()

plt.show()


# In[18]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

# 1. Compute Permutation Importance directly on the pipeline object
# Passing 'sample_weight=w_test' guarantees your importance ranks are survey-adjusted
result = permutation_importance(
    gb_clock_pipeline, 
    X_test, 
    y_test, 
    n_repeats=10,          # Shuffle each feature 10 separate times for stability
    random_state=42,
    n_jobs=-1,             # Use all available CPU cores
    sample_weight=w_test   # CRITICAL: Injects NHANES survey weights into the evaluator
)

# 2. Extract the mean importance drop for each original base feature
# Since it tests inputs BEFORE they enter the pipeline, it matches X_test.columns perfectly!
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance_Drop': result.importances_mean
}).sort_values(by='Importance_Drop', ascending=False)

# 3. Render the clean base-feature importance chart
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.barplot(
    x='Importance_Drop', 
    y='Feature', 
    data=importance_df, 
    palette='viridis',
    hue='Feature',
    legend=False
)

plt.title('Permutation Importance: Clean Base Features (Survey-Weighted)', fontsize=14, pad=15)
plt.xlabel('Drop in Test R² When Feature is Shuffled', fontsize=12)
plt.ylabel('NHANES Core Variables', fontsize=12)
plt.tight_layout()

plt.show()


# In[19]:


import statsmodels.api as sm

age_accelerate_df = clean_data.copy()
# Create dummy columns for your population groups (e.g., Females baseline vs Males)
# 1 = Male, 2 = Female -> Male dummy = 1 if male, 0 if female
age_accelerate_df['IS_MALE'] = (age_accelerate_df['RIAGENDR'] == 1).astype(int)

X = sm.add_constant(age_accelerate_df['IS_MALE'])
y = age_accelerate_df['age_acceleration']
weights = age_accelerate_df['WTSAF2YR']

# Fit the population-weighted model
wls_model = sm.WLS(y, X, weights=weights).fit()

# The intercept is the weighted mean for Females. 
# The Intercept + IS_MALE coefficient is the weighted mean for Males.
print(wls_model.summary())


# In[20]:


import joblib
model_filename = 'gb_bio_age_clock_pipeline.joblib'
joblib.dump(gb_clock_pipeline, model_filename, compress=3)


# In[21]:


print(f"🎉 Success! Pipeline securely saved to: {model_filename}")


# In[22]:


from my_package import BiologicalAgeClock


# In[ ]:




