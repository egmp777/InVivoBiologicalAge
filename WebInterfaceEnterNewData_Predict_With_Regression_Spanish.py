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
from googletrans import Translator


## Source: https://www.google.com/search?q=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&sca_esv=ac35dc61bdce8476&sxsrf=AE3TifNWsNirD54fICsIJwfMI746Y5RcKw%3A1752859582973&ei=voN6aKWQO-fY1sQPkp3M4QY&ved=0ahUKEwil56Lm9saOAxVnrJUCHZIOM2wQ4dUDCBA&uact=5&oq=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&gs_lp=Egxnd3Mtd2l6LXNlcnAiTHNhdmUgdG8gY3N2IGNvbW1hIHNlcGFyYXRlZCBmaWxlIHdpdGggaGVhZGVyIHVzZXIgaW5wdXQgcHl0aG9uIGFuZCBzdHJlYW1saXRI_CZQrwRY6yRwBHgBkAEAmAHaAaAB-w2qAQYxLjEzLjG4AQPIAQD4AQGYAgOgAvoBwgIKEAAYsAMY1gQYR8ICBxAjGLACGCeYAwCIBgGQBgSSBwMxLjKgB-0osgcDMC4yuAf1AcIHBTAuMi4xyAcH&sclient=gws-wiz-serp




##Testing database connection and retrieval
conn = st.connection("postgresql", type="sql")
workout_df = conn.query("SELECT * FROM workout", ttl="10m") # Caches for 10 minutes
st.write(workout_df)
##Testing database connection and retrieval



PATH = '.'

data_with_variables_used_for_prediction = "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"
load_data = os.path.join(PATH, data_with_variables_used_for_prediction)
translator = Translator()
english_spanish_column_dictionary = {}


df = pd.read_csv(load_data)
df_es = df.copy()
df_es.rename(columns=lambda x: translator.translate(x, src='en', dest='es').text, inplace=True)

# for value in df_es.columns:
#     print(value)
#     df_es[value] = df_es[value].apply(lambda x: translator.translate(x,src='en', dest='es').text)
#     break
english_spanish_value_dictionary = {}
poblacion_list = df['population'].unique().tolist()
tipo_ejercicio_list = df['type_exercise'].unique().tolist()
endpoints_list = df['endpoint'].unique().tolist()

for each_value in poblacion_list:
    english_spanish_value_dictionary[each_value] =  translator.translate(each_value, src='en', dest='es').text

for each_value in tipo_ejercicio_list:
    english_spanish_value_dictionary[each_value] = translator.translate(each_value, src='en', dest='es').text


for each_value in endpoints_list:
    english_spanish_column_dictionary[each_value] =  translator.translate(each_value, src='en', dest='es').text

print(english_spanish_value_dictionary)



##print(df_es.info())

# endpoints_list = df['endpoint'].unique().tolist()
# for each_value in endpoints_list:
#     english_spanish_column_dictionary[each_value] =  translator.translate(each_value, src='en', dest='es').text



##print(">>>Printing the translations...")
import textwrap

wrapper = textwrap.TextWrapper(width=50)

translated_term = []
for key in english_spanish_column_dictionary:
    ##print(key + " : " + wrapper.fill(text=english_spanish_column_dictionary[key]))
    translated_term.append(english_spanish_column_dictionary[key])
    ##print("=========================================================")
    df_es['punto final'] = df_es['punto final'].replace(key,english_spanish_column_dictionary[key])

for key in english_spanish_value_dictionary:
    df_es['población'] = df_es['población'].replace(key,english_spanish_value_dictionary[key])
    df_es['tipo_ejercicio'] = df_es['tipo_ejercicio'].replace(key,english_spanish_value_dictionary[key])


pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', None)
print(df_es)


endpoint = ""
## Source: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
st.session_state.clear()

endpoint_list_spanish = df

##DEC 1 2025
# endpoint = st.selectbox('Seleccionar que indicador desea optimizar',
#                         options=df['endpoint'].unique().tolist())

enpoint_esp = st.selectbox('Seleccionar que indicador desea optimizar',
                        options=df_es['punto final'].unique().tolist())

endpoint = translator.translate(enpoint_esp, src='es', dest='en').text

##print(endpoint)

##DES 1 2025
# type_exercise_esp = st.selectbox('Seleccionar que tipo de ejercicio prefiere',
#                                  options=df_es['tipo_ejercicio'].unique().tolist())
#
# type_exercise = translator.translate(type_exercise_esp, src='es', dest='en').text

population_esp = st.selectbox('Seleccionar la opción que describa mejor sus estado de salúd ',
                                 options=df_es['población'].unique().tolist())

population = translator.translate(population_esp, src='es', dest='en').text

duration = st.selectbox("Seleccione que preferencia tiene de duración del ejercicio:",
                        options=df['duration'].unique().tolist())

if endpoint:
    st.write(endpoint)

# population = st.selectbox("Seleccione la opción que describa mejor sus estado de salúd: ",
#                         options=df['population'].unique().tolist())



# type_exercise = st.selectbox("Seleccione su preferencia de tipo de ejercicio",
#                         options=df['type_exercise'].unique().tolist() )

age = st.number_input("Ingerese su Edad: ")

age_range = '55-65'
if (age >= 45 and age <= 55):
    age_range = '45-55'

##DEC 1 2025
# data = {'endpoint': [endpoint], 'population': [population], 'age': [age], 'duration':[duration],
#         'type_exercise': [type_exercise]}

data = {'endpoint': [endpoint], 'population': [population], 'age': [age], 'duration':[duration]}


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

# @st.cache_resource
# def load_model(_filename):
#     try:
#         # loaded_model = joblib.load(file)
#         loaded_model = joblib.load(open(_filename, 'rb'))
#     except FileNotFoundError:
#         st.write("file not found", filename)
#     except Exception as e:
#         st.write("an error occured when loading", filename)
#     return loaded_model
#
# loaded_model = load_model(filename)


def make_predictions(model, new_data):


    st.write(new_data)


    predictions = model.predict(new_data)

    st.write(predictions)

    df_with_columns = pd.DataFrame(predictions, columns=['description'])
    prediction = df_with_columns.iloc[0,0]
    exercise_type = ""
    exercise_type_es = ""
    if prediction == "MICT":
        exercise_type = "For your profile we recommend a Moderate Intensity Interval Training routine. "
        exercise_type_es = translator.translate(exercise_type, src='en', dest='es').text




    st.write(prediction)

    st.write(exercise_type)
    st.write(exercise_type_es)


    # get_details_df = conn.query(f"SELECT description FROM workout WHERE workout_name = 'MICT'" , ttl="10m") # Caches for 10 minutes

    ##DEC 2 2025
    # get_details_df = conn.query(f"SELECT description FROM workout WHERE workout_name = '{prediction}'" , ttl="10m")

    get_details_df = conn.query(f"SELECT description FROM workout WHERE workout_name = '{prediction}' and duration = '{duration}' "
                                f"and age_range = '{age_range}' and population = '{population}'"  , ttl="10m")
    get_details_df_esp = translator.translate(get_details_df.iloc[0,0], src='en', dest='es').text
    # st.write(get_details_df['description'])
    st.markdown('\n\n')
    st.markdown(get_details_df.iloc[0,0])
    st.markdown(get_details_df_esp)
    ##st.write(get_details_df.iloc[0,0])


if st.button("Save Data"):

    if endpoint and age and population:  # Ensure all fields are filled
        st.write(new_data)
        make_predictions(loaded_model, new_data)
        st.write(new_data)
    else:
        st.warning("Please fill in all fields.")



