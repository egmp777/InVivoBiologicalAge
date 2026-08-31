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

data_file = "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"

## file_name_clustering_data = "New Data Points Prediction_Without_Study_Category_Duration_With_Age_with_Duration - DataForClustering.csv"

data = os.path.join(PATH, data_file)


df_data = pd.read_csv(data)

df_data.drop(df_data[(df_data['Mean_HIIE'] > df_data['Mean_HIIE'].quantile(0.975)) |
                                           (df_data['Mean_HIIE'] <
                                            df_data['Mean_HIIE'].quantile(0.025))].index,inplace=True)
df_data.drop(df_data[(df_data['Mean_MICT'] > df_data['Mean_MICT'].quantile(0.975)) |
                                           (df_data['Mean_MICT'] <
                                            df_data['Mean_MICT'].quantile(0.025))].index,inplace=True)


df_data.drop(df_data[(df_data['age'] > df_data['age'].quantile(0.975)) |
                                           (df_data['age'] <
                                            df_data['age'].quantile(0.025))].index,inplace=True)
df_data.drop(df_data[(df_data['duration'] > df_data['duration'].quantile(0.975)) |
                                           (df_data['duration'] <
                                            df_data['duration'].quantile(0.025))].index,inplace=True)


data = df_data.copy()









# <editor-fold desc="dropping 'desired_effect', 'Mean_HIIE', 'Mean_MICT'">
#### JULY 9 2025 ####

data = data.drop(['desired_effect',  'Mean_MICT'], axis=1)







# </editor-fold>










rf = RandomForestClassifier()




from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

data["target"] = data['Mean_HIIE'].astype('int')




target_name = 'target'
target = data[target_name]

data = data.drop(['Mean_HIIE', 'target'], axis=1)
st.write(data)


from sklearn.compose import make_column_selector as selector
numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

from sklearn.compose import make_column_transformer

from sklearn.compose import ColumnTransformer
preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)



from sklearn.pipeline import make_pipeline



model = make_pipeline(preprocessor, rf)


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)







# Fit the grid search to your data
_ = model.fit(data_train, target_train)
# st.write("Best parameters: ", grid_search.best_params_)
# st.write("Best score: ", grid_search.best_score_)




# rf.fit(data_train, target_train)

y_pred = model.predict(data_test)[:]

st.write(model.score(data_test, target_test))

filename = 'trained_model_random_forest.joblib'
with open(filename, 'wb') as file:
    joblib.dump(model, file)


from sklearn.linear_model import LogisticRegression
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


