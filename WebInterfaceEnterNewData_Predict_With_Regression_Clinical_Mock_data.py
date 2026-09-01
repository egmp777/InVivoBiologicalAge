import os
import streamlit as st
import pandas as pd


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
import joblib


## Source: https://www.google.com/search?q=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&sca_esv=ac35dc61bdce8476&sxsrf=AE3TifNWsNirD54fICsIJwfMI746Y5RcKw%3A1752859582973&ei=voN6aKWQO-fY1sQPkp3M4QY&ved=0ahUKEwil56Lm9saOAxVnrJUCHZIOM2wQ4dUDCBA&uact=5&oq=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&gs_lp=Egxnd3Mtd2l6LXNlcnAiTHNhdmUgdG8gY3N2IGNvbW1hIHNlcGFyYXRlZCBmaWxlIHdpdGggaGVhZGVyIHVzZXIgaW5wdXQgcHl0aG9uIGFuZCBzdHJlYW1saXRI_CZQrwRY6yRwBHgBkAEAmAHaAaAB-w2qAQYxLjEzLjG4AQPIAQD4AQGYAgOgAvoBwgIKEAAYsAMY1gQYR8ICBxAjGLACGCeYAwCIBgGQBgSSBwMxLjKgB-0osgcDMC4yuAf1AcIHBTAuMi4xyAcH&sclient=gws-wiz-serp











PATH = '.'

data_with_variables_used_for_prediction = "clinical_study_exercise_mock_health_corrected_1000 (1)_Without_PatientId_And_OnlyFastingInsuline.csv"

load_data = os.path.join(PATH, data_with_variables_used_for_prediction)



df = pd.read_csv(load_data)
endpoint = ""
## Source: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
st.session_state.clear()



health_status = st.selectbox("Seleccione la opción que describa mejor sus estado de salúd: ",
                        options=df['health_status'].unique().tolist())

exercise_program = st.selectbox("Seleccione el tipo de ejercicio que más le acomode: ",
                        options=df['exercise_program'].unique().tolist())

duration = st.selectbox("Seleccione que preferencia tiene de duración del ejercicio:",
                        options=df['duration'].unique().tolist())

duration_weeks = st.selectbox("Seleccione la duración en semnaas: ",
                        options=df['duration_weeks'].unique().tolist() )

age = st.number_input("Ingerese su Edad: " )

gender = st.selectbox("Seleccione su género: ",
                        options=df['gender'].unique().tolist() )



data = {'health_status': [health_status], 'age': [age], 'duration':[duration],
        'duration_weeks': [duration_weeks], 'gender': [gender], 'exercise_program':[exercise_program]}

df_new_patient_dataframe = pd.DataFrame(data)
st.write(df_new_patient_dataframe)
new_data = df_new_patient_dataframe


filename = 'trained_model_regression.joblib'

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
finally:
    st.write("an error occured when loading")
    exit(0)

def make_predictions(model, new_data):


    st.write(new_data)


    predictions = model.predict(new_data)

    st.write(predictions)


if st.button("Save Data"):
    try:
        # loaded_model = joblib.load(file)
        loaded_model = joblib.load(open(filename, 'rb'))
    except FileNotFoundError:
        st.write("file not found", filename)
    except Exception as e:
        st.write("an error occured when loading", filename)

    if health_status and age and gender and duration and duration_weeks:  # Ensure all fields are filled
        st.write(new_data)
        make_predictions(loaded_model, new_data)
        st.write(new_data)
    else:
        st.warning("Please fill in all fields.")



