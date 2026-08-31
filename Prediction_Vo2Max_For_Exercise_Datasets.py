import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, Lasso
import os
import streamlit as st
import plotly.express as px
import seaborn as sns
import pickle
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
import pickle
import joblib
from sklearn import linear_model



PATH = '.'

## file_name_clustering_data =  "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"

#data_for_prediction = "Historical_Data_From_Fitness_Workout_Data_Set_2800_Rows.csv"
##data_for_prediction = "Historical_Data_From_Fitness_Workout_Data_Set_2800_With_Resting_Heart_Rate.csv"
# data_to_predict = "Target_File_From_workout_fitness_tracker_data _with_workout_type_and_workout_duration.csv"
data_to_predict = "Target_File_From_workout_fitness_tracker_data.csv"
##data_for_prediction = "Historical_Data_From_Fitness_Workout_Data_Set_2800_Rows - Sheet1.csv"
data_for_prediction = "Historical_Data_From_Fitness_Workout_Data_Set_2800_Rows - Sheet1 (1).csv"


loaded_data = os.path.join(PATH, data_for_prediction)

pandas_data = pd.read_csv(loaded_data)

###

## ADDING A BMI COLUMN
pandas_data['BMI'] = pandas_data['weight'] + (pandas_data['height'] ** 2)
st.write(pandas_data.isna().sum())

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(pandas_data)
categorical_columns = categorical_columns_selector(pandas_data)



target = pandas_data['vo2_max']

pandas_data = pandas_data.drop(['vo2_max', 'steps', 'weight', 'height'], axis = 1)
##st.write(pandas_data)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()


numerical_columns = numerical_columns_selector(pandas_data)
categorical_columns = categorical_columns_selector(pandas_data)

# from sklearn.compose import make_column_transformer
#
# from sklearn.compose import ColumnTransformer
# preprocessor = make_column_transformer(
#     (categorical_preprocessor, categorical_columns),
#     (numerical_preprocessor, numerical_columns),
# )
#
# from sklearn.pipeline import make_pipeline
#
# lin_regression = linear_model.LinearRegression()
#
#
#
#
# model = make_pipeline(preprocessor, lin_regression)
#
#
#
# from sklearn.model_selection import train_test_split
#
# data_train, data_test, target_train, target_test = train_test_split(
#     pandas_data, target, random_state=42
# )
#
# model.fit(data_train, target_train )
# st.write(model.score(data_test, target_test))
# exit(0)


# target = pandas_data['vo2_max']
#
# pandas_data = pandas_data.drop(['vo2_max', 'heart_rate'], axis = 1)


st.write(pandas_data)
st.write(target)


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    pandas_data, target, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(data_train) # Fit and transform on training data
X_test_scaled = scaler.transform(data_test)       # Only transform testing data


# transformer = ColumnTransformer(
#     transformers=[
#         ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['workout_type'])
#     ],
#     remainder='passthrough' # Keep other columns (like 'Weight')
# )
#
#
# # Fit and transform the training data
# X_train_encoded = transformer.fit_transform(data_train)
# # Transform the test data using the same rules learned from the training data
# X_test_encoded = transformer.transform(data_test)

# regressor = RandomForestRegressor(n_estimators=300, random_state=50, oob_score=True)
# regressor.fit(data_train, target_train)
# st.write(regressor.score(data_test, target_test))
# exit(0)



model = LinearRegression()
# regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
#
# regressor.fit(data_train, target_train)

model.fit(X_train_scaled, target_train )
st.write(model.score(X_test_scaled, target_test))
exit(0)

##model.fit(data_train, target_train)

pred = model.predict(data_test)

st.write(model.score(data_test, target_test))

