#######  WE HAVE THE TWO FUNCTIONS TO PREPARE THE FEATURE SET 

######### INITIAL FUNCTION   ##########
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

#######    FIRST FEATURE LIST #################
# Define the expanded feature list including the physical exam variables
extended_features = [
    'LBXGLU', 'LBXIN', 'LBXHSCRP', 'LBXGH', 
    'HOMA_IR', 'LOG_CRP', 'GLU_A1C_RATIO', 
    'GLU_X_CRP', 'INSU_X_CRP',
    'BMXBMI', 'BMXWAIST', 'BPXSY1', 'BPXDI1', 'MAP', 
    'BMI_X_WAIST', 'BMI_X_HOMA'
]
# 1. Assume 'clean_data' has base variables and you applied your feature engineering function
## enhanced_data = engineer_nhanes_features(clean_data)
## X_eng = enhanced_data[engineered_features]
## y = enhanced_data['RIDAGEYR']
## weights = enhanced_data['WTSAF2YR']

######################################### OCTOBER 13 2026 TESTING THE R2 #############################

## from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
## from sklearn.model_selection import train_test_split

# --- STEP B: Train / Test Split Alignment ---
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_eng, y, weights, test_size=0.3, random_state=42
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

exit(0)
#######################################################################################################


########### SECOND FUNCTION ########
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


######## SECOND FEATURE SET   ##########
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
