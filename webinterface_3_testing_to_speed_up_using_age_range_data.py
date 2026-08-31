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

translator = Translator()
english_spanish_column_dictionary = {}
english_spanish_value_dictionary = {}

conn = st.connection("postgresql", type="sql")


@st.cache_data
def load_data_file():
    PATH = '.'
    data_with_variables_used_for_prediction = "DataForClustering_Without_Study_Category_Duration_With_Age_Range_with_Duration.csv"
    load_data = os.path.join(PATH, data_with_variables_used_for_prediction)
    translator = Translator()
    english_spanish_column_dictionary = {}

    df = pd.read_csv(load_data)
    return df


def get_workout_details(prediction, population, age_range, duration):
    get_details_df = conn.query(
        f"SELECT description FROM workout WHERE workout_name = '{prediction}' and duration = '{duration}' "
        f"and age_range = '{age_range}' and population = '{population}'", ttl="10m")
    return get_details_df


@st.cache_resource
def load_model():
    filename = 'trained_model_regression_with_age_range.joblib'
    try:
        # loaded_model = joblib.load(file)
        loaded_model = joblib.load(open(filename, 'rb'))
    except FileNotFoundError:
        st.write("file not found", filename)
    except Exception as e:
        st.write("an error occured when loading", filename)
    return loaded_model


loaded_model = load_model()


def make_predictions(model, new_data):
    ##st.write(new_data)

    predictions = model.predict(new_data)

    return predictions


df = load_data_file()
df_es = df.copy()
df_es.rename(columns=lambda x: translator.translate(x, src='en', dest='es').text, inplace=True)
poblacion_list = df['population'].unique().tolist()
tipo_ejercicio_list = df['type_exercise'].unique().tolist()
endpoints_list = df['endpoint'].unique().tolist()

for each_value in poblacion_list:
    english_spanish_value_dictionary[each_value] = translator.translate(each_value, src='en', dest='es').text
for each_value in tipo_ejercicio_list:
    english_spanish_value_dictionary[each_value] = translator.translate(each_value, src='en', dest='es').text
for each_value in endpoints_list:
    english_spanish_column_dictionary[each_value] = translator.translate(each_value, src='en', dest='es').text

translated_term = []
for key in english_spanish_column_dictionary:
    ##print(key + " : " + wrapper.fill(text=english_spanish_column_dictionary[key]))
    translated_term.append(english_spanish_column_dictionary[key])
    ##print("=========================================================")
    df_es['punto final'] = df_es['punto final'].replace(key, english_spanish_column_dictionary[key])

for key in english_spanish_value_dictionary:
    df_es['población'] = df_es['población'].replace(key, english_spanish_value_dictionary[key])
    ##print(key)
    df_es['tipo_ejercicio'] = df_es['tipo_ejercicio'].replace(key, english_spanish_value_dictionary[key])

endpoint = ""
## Source: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
## st.session_state.clear()

endpoint_list_spanish = df
css = """
.st-key-exercise_container {
    background-color: rgba(100, 100, 200, 0.3);
}
"""
st.html(f"<style>{css}</style>")

with st.form("profile_data_form"):
    # Add your form elements here

    # name = st.text_input("Enter your name")
    # age = st.slider("Select your age", 0, 100)
    # submitted = st.form_submit_button("Submit")

    ## Monday October 8
    if "step" not in st.session_state:
        st.session_state.step = 1

    # if "endpoint" not in st.session_state:
    #     st.session_state.endpoint = None
    #
    # if "population" not in st.session_state:
    #     st.session_state.population = None
    #
    # if "duration" not in st.session_state:
    #     st.session_state.duration = None
    #
    # if "age" not in st.session_state:
    #     st.session_state.age = None

    if "plan" not in st.session_state:
        st.session_state.plan = " "


    def display_plan(plan):
        st.session_state.plan = plan
        st.session_state.step = 2


    if st.session_state.step == 1:

        enpoint_esp = st.selectbox('Seleccionar que indicador desea optimizar',
                                   options=df_es['punto final'].unique().tolist(),
                                   index=None, placeholder="--",
                                   )
        if enpoint_esp != None:
            endpoint = translator.translate(enpoint_esp, src='es', dest='en').text
            st.session_state.endpoint = endpoint

        population_esp = st.selectbox('Seleccionar la opción que describa mejor sus estado de salúd ',
                                      options=df_es['población'].unique().tolist(), index=None, placeholder="--", )

        if population_esp != None:
            population = translator.translate(population_esp, src='es', dest='en').text
            st.session_state.population = population

        duration = st.selectbox("Seleccione que preferencia tiene de duración del ejercicio:",
                                options=df['duration'].unique().tolist(), index=None, placeholder="--", )

        if duration != None:
            duration = duration
            st.session_state.duration = duration

        # age_range = st.selectbox("Seleccione un rango de edad: ", options=['45-55', '56-61', '62-67', '67-72', '72+'],
        #                          index=None, placeholder="--",)

        age = st.number_input("Ingerese su Edad: ")
        st.session_state.age = age

        age_range = '55-65'
        if (age >= 45 and age < 55):
            age_range = '45-55'
        else:
            age_range = '55-60'

        submitted = st.form_submit_button(label="Ver mi rutina de ejercicio")

if submitted:

    if st.session_state.endpoint and st.session_state.age and st.session_state.population and st.session_state.duration:  # Ensure all fields are filled

        data = {'endpoint': [st.session_state.endpoint], 'population': [st.session_state.population],
                'age': [st.session_state.age], 'duration': [st.session_state.duration]}

        df_new_patient_dataframe = pd.DataFrame(data)

        new_data = df_new_patient_dataframe

        from sklearn.compose import make_column_selector as selector

        numerical_columns_selector = selector(dtype_exclude=object)
        categorical_columns_selector = selector(dtype_include=object)

        numerical_columns = numerical_columns_selector(new_data)
        categorical_columns = categorical_columns_selector(new_data)

        predictions = make_predictions(loaded_model, new_data)
        df_with_columns = pd.DataFrame(predictions, columns=['description'])
        prediction = df_with_columns.iloc[0, 0]
        # st.write(prediction)
        # st.write(age_range)
        # st.write(population)
        # st.write(duration)
        exercise_type = ""
        exercise_type_es = ""
        if prediction == "MICT":
            exercise_type = "For your profile we recommend a Moderate Intensity Interval Training routine. "
            exercise_type_es = translator.translate(exercise_type, src='en', dest='es').text
        else:
            exercise_type = "For your profile we recommend a High Intensity Interval Training routine. "
            exercise_type_es = translator.translate(exercise_type, src='en', dest='es').text

        st.write(prediction)

        get_details_df = get_workout_details(prediction, population, age_range, duration)
        get_details_df_en = get_details_df.iloc[0, 0]
        st.session_state.plan = translator.translate(get_details_df.iloc[0, 0], src='en', dest='es').text

        ##with st.container(border=True, key="exercise_container"):
        ##  st.write("#### Su Rutina Personalizada\n\n")
        ## st.write(get_details_df_esp)

        # placeholder = st.empty()
        #
        # placeholder.write(f"{get_details_df_esp}")

        # st.write(get_details_df['description'])

        # st.markdown('\n\n')
        # st.markdown(get_details_df.iloc[0, 0])
        # st.markdown(get_details_df_esp)

        ##st.session_state.plan = get_details_df_esp
        # st.session_state.loading = True

        st.switch_page("pages/pages.py")







    else:
        st.warning("Please fill in all fields.")







