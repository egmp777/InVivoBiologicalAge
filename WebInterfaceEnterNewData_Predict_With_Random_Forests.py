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

## Source: https://www.google.com/search?q=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&sca_esv=ac35dc61bdce8476&sxsrf=AE3TifNWsNirD54fICsIJwfMI746Y5RcKw%3A1752859582973&ei=voN6aKWQO-fY1sQPkp3M4QY&ved=0ahUKEwil56Lm9saOAxVnrJUCHZIOM2wQ4dUDCBA&uact=5&oq=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&gs_lp=Egxnd3Mtd2l6LXNlcnAiTHNhdmUgdG8gY3N2IGNvbW1hIHNlcGFyYXRlZCBmaWxlIHdpdGggaGVhZGVyIHVzZXIgaW5wdXQgcHl0aG9uIGFuZCBzdHJlYW1saXRI_CZQrwRY6yRwBHgBkAEAmAHaAaAB-w2qAQYxLjEzLjG4AQPIAQD4AQGYAgOgAvoBwgIKEAAYsAMY1gQYR8ICBxAjGLACGCeYAwCIBgGQBgSSBwMxLjKgB-0osgcDMC4yuAf1AcIHBTAuMi4xyAcH&sclient=gws-wiz-serp











PATH = '.'

data_with_variables_used_for_prediction = "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"
load_data = os.path.join(PATH, data_with_variables_used_for_prediction)


df = pd.read_csv(load_data)
endpoint = ""
## Source: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
st.session_state.clear()
endpoint = st.selectbox('Seleccionar que indicador desea optimizar',
                        options=df['endpoint'].unique().tolist())

if endpoint:
    st.write(endpoint)

population = st.selectbox("Seleccione la opción que describa mejor sus estado de salúd: ",
                        options=df['population'].unique().tolist())

duration = st.selectbox("Seleccione que preferencia tiene de duración del ejercicio:",
                        options=df['duration'].unique().tolist())

type_exercise = st.selectbox("Seleccione su preferencia de tipo de ejercicio",
                        options=df['type_exercise'].unique().tolist() )

age = st.number_input("Ingerese su Edad: ")


data = {'endpoint': [endpoint], 'population': [population], 'age': [age], 'duration':[duration],
        'type_exercise': [type_exercise]}

df_new_patient_dataframe = pd.DataFrame(data)
st.write(df_new_patient_dataframe)
new_data = df_new_patient_dataframe


filename = 'trained_model_random_forest.joblib'

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(new_data)
categorical_columns = categorical_columns_selector(new_data)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer


from sklearn.preprocessing import OneHotEncoder, StandardScaler

# categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
# numerical_preprocessor = StandardScaler()
#
# preprocessor = make_column_transformer(
#     (categorical_preprocessor, categorical_columns),
#     (numerical_preprocessor, numerical_columns),
# )
# model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))

try:
    # loaded_model = joblib.load(file)
    loaded_model = joblib.load(open(filename, 'rb'))
except FileNotFoundError:
    st.write("file not found", filename)
except Exception as e:
    st.write("an error occured when loading", filename)

def make_predictions(model, new_data):


    st.write(new_data)


    predictions = model.predict(new_data)

    st.write(predictions)


if st.button("Save Data"):

    if endpoint and age and population:  # Ensure all fields are filled
        st.write(new_data)
        make_predictions(loaded_model, new_data)
        st.write(new_data)
    else:
        st.warning("Please fill in all fields.")



