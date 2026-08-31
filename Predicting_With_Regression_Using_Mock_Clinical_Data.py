import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image

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

PATH = '.'

## data_file = "clinical_study_exercise_mock_health_Only_Fasting_Insulin_corrected_1000_realistic - clinical_study_exercise_mock_health_corrected_1000_realistic (1).csv"
data_file = "clinical_study_exercise_mock_health_Only_VO2Max_corrected_1000_realistic - clinical_study_exercise_mock_health_corrected_1000_realistic (1) (1).csv"
data = os.path.join(PATH, data_file)


df_data = pd.read_csv(data)

st.write(df_data[df_data.isna().any(axis=1)])

df_data = df_data.dropna()



st.write(df_data[(df_data['baseline_VO2Max'] > df_data['baseline_VO2Max'].quantile(0.975)) |
                                           (df_data['baseline_VO2Max'] <
                                            df_data['baseline_VO2Max'].quantile(0.025))])
outliers_baseline = df_data[(df_data['baseline_VO2Max'] > df_data['baseline_VO2Max'].quantile(0.975)) |
                                           (df_data['baseline_VO2Max'] <
                                            df_data['baseline_VO2Max'].quantile(0.025))]

df_data.drop(outliers_baseline, axis=1)

st.write(outliers_baseline)

st.write("Number of baseline outliers deleted", len(outliers_baseline))

outliers_post = df_data[(df_data['post_exercise_VO2Max'] > df_data['post_exercise_VO2Max'].quantile(0.975)) |
                                           (df_data['post_exercise_VO2Max'] <
                                            df_data['post_exercise_VO2Max'].quantile(0.025))]

df_data.drop(outliers_post, axis = 1)

st.write(outliers_post)
st.write("Number of post-exercise outliers deleted", len(outliers_post))

data = df_data.copy()

data['effect'] = data.apply(lambda row: 'Decline' if (row['difference'] < 0)  else ('No change' if  (row['difference'] == 0) else 'No Decline'), axis=1)


data['target'] = data['effect']

target_name = 'target'
target = data[target_name]
st.subheader("Distribution of the effect of exercise")
fig = px.histogram(data, x='target', nbins=20)
st.plotly_chart(fig)


data = data = data.drop(['effect', 'target', 'post_exercise_VO2Max', 'baseline_VO2Max', 'difference'], axis =1)

st.write("data", data)


from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)


numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))


_ = model.fit(data_train, target_train)

filename = 'trained_model_regression.joblib'
with open(filename, 'wb') as file:
    joblib.dump(model, file)

model.predict(data_test)[:]

st.write(model.score(data_test, target_test))






