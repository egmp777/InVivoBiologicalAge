import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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


### Source: https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline_column_transformer.html
PATH = '.'

file_name_clustering_data =  "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"


load_cluster_data = os.path.join(PATH, file_name_clustering_data)


df_clustering_data = pd.read_csv(load_cluster_data)

st.write(df_clustering_data)



# <editor-fold desc="Removing Outliers">
df_clustering_data.drop(df_clustering_data[(df_clustering_data['Mean_HIIE'] > df_clustering_data['Mean_HIIE'].quantile(0.975)) |
                                           (df_clustering_data['Mean_HIIE'] <
                                            df_clustering_data['Mean_HIIE'].quantile(0.025))].index,inplace=True)
df_clustering_data.drop(df_clustering_data[(df_clustering_data['Mean_MICT'] > df_clustering_data['Mean_MICT'].quantile(0.975)) |
                                           (df_clustering_data['Mean_MICT'] <
                                            df_clustering_data['Mean_MICT'].quantile(0.025))].index,inplace=True)


df_clustering_data.drop(df_clustering_data[(df_clustering_data['age'] > df_clustering_data['age'].quantile(0.975)) |
                                           (df_clustering_data['age'] <
                                            df_clustering_data['age'].quantile(0.025))].index,inplace=True)
df_clustering_data.drop(df_clustering_data[(df_clustering_data['duration'] > df_clustering_data['duration'].quantile(0.975)) |
                                           (df_clustering_data['duration'] <
                                            df_clustering_data['duration'].quantile(0.025))].index,inplace=True)

# </editor-fold>

data = df_clustering_data.copy()

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)



data = st.data_editor(data)

st.write(data)

data['HIIE_success'] = data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
                                                    and row['desired_effect'] == 'decrease') | (row['Mean_HIIE'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)
data['MICT_success'] = data.apply(lambda row: 1 if (row['Mean_MICT'] < 0
                                                    and row['desired_effect'] == 'decrease') | (row['Mean_MICT'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)


# <editor-fold desc="dropping 'desired_effect', 'Mean_HIIE', 'Mean_MICT'">
#### JULY 9 2025 ####

data = data.drop(['desired_effect', 'Mean_HIIE', 'Mean_MICT'], axis=1)







# </editor-fold>


def target_protocol_value(row):
    if (row['HIIE_success'] == 1
            and row['MICT_success'] == 0):
        return 0
    elif (row['HIIE_success'] == 0
          and row['MICT_success'] == 1):
        return 1
    elif (row['HIIE_success'] == 1
          and row['MICT_success'] == 1):
        return 2
    else:
        return 3



data['target_protocol'] = data.apply(target_protocol_value, axis=1)
target_name = 'target_protocol'
target = data['target_protocol']

data = data.drop(['HIIE_success', 'MICT_success', 'target_protocol'], axis=1)
st.write(data)

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

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))



from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

_ = model.fit(data_train, target_train)

model.predict(data_test)[:]

st.write(model.score(data_test, target_test))

filename = 'trained_model_regression.joblib'
with open(filename, 'wb') as file:
    joblib.dump(model, file)


