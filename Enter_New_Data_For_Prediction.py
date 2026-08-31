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
## Write to file

PATH = '.'

filename = 'trained_model.joblib'
file = os.path.join(PATH, filename)

new_patient_file = 'New_patient_data.csv'
new_patient_data = os.path.join(PATH, new_patient_file)

loaded_pt = joblib.load('power_transformer.joblib')
loaded_le = joblib.load('label_encoder.joblib')
with open('label_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)

with open('transformers.pkl', 'rb') as f:
    loaded_transformers = pickle.load(f)
encoders = {}
categorical_features = ['endpoint', 'population']

def make_predictions(filename, new_data):
    # for c in new_data.select_dtypes(exclude='object').columns:
    #
    #     new_data[c] = loaded_transformers.transform(np.array(new_data[c]).reshape(-1, 1))



    for col in categorical_features:
        # st.write(col)
        # st.write("new_data", new_data[col])
        new_data[col] = loaded_encoders[col].transform(new_data[col])
        #le = LabelEncoder()
        #st.write(loaded_le.transform(["BMI", "VO2max"]))
        ##new_data[col] = loaded_le.transform(new_data[col])
        ##encoders[col] = le


    st.write(new_data)
    try:
        # loaded_model = joblib.load(file)
        loaded_model = joblib.load(open(filename, 'rb'))
    except FileNotFoundError:
        st.write("file not found", file)
    except Exception as e:
        st.write("an error occured when loading", file)

    predictions = loaded_model.predict(new_data)
   
    st.write(predictions)



def save_data(file, df_data):

    try:
        df_existing = pd.read_csv(file)
        st.write("existing data", df_existing)
        df_updated = pd.concat([df_existing, df_data], ignore_index=True)

        st.write("updated data", df_updated)
        df_updated.to_csv('New_patient_data.csv', index=False)
        st.success("Data saved to New_patient_data.csv!")
    except FileNotFoundError:
     # If file doesn't exist, use the new data as the first row
        df_updated = df_data
        st.wrirte("FileNotFoundError")


PATH = '.'
endpoint = ""
## Source: https://docs.streamlit.io/develop/api-reference/widgets/st.number_input
st.session_state.clear()
endpoint = st.selectbox('Seleccionar que indicador desea optimizar', options=[None, 'BMI', 'Fasting Insulin', 'VO2max', 'Diastolic Blood Pressure'])

if endpoint:
    st.write(endpoint)

population = st.selectbox("Seleccione la opción que describa mejor sus estado de salúd: ",
                          options = ['Healthy', "Metabolic Syndrome", "T2D", "Overweight/obese", 'Cardiac Rehabilitation'])

age = st.number_input("Ingerese su Edad: ")


data = {'endpoint': [endpoint], 'population': [population], 'age': [age]}



df_new_patient_dataframe = pd.DataFrame(data)
st.write(df_new_patient_dataframe)
new_data = df_new_patient_dataframe
categorical_features = ['endpoint', 'population']
model = 'trained_model.joblib'


if st.button("Save Data"):

    if endpoint and age and population: # Ensure all fields are filled
        st.write(new_data)
        make_predictions(model, new_data)
    else:
        st.warning("Please fill in all fields.")


exit(0)

#             st.warning("Please fill in all fields.")

# file = "New_patient_data.csv"
#
# ## Source: https://www.google.com/search?q=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&sca_esv=ac35dc61bdce8476&sxsrf=AE3TifNWsNirD54fICsIJwfMI746Y5RcKw%3A1752859582973&ei=voN6aKWQO-fY1sQPkp3M4QY&ved=0ahUKEwil56Lm9saOAxVnrJUCHZIOM2wQ4dUDCBA&uact=5&oq=save+to+csv+comma+separated+file+with+header+user+input+python+and+streamlit&gs_lp=Egxnd3Mtd2l6LXNlcnAiTHNhdmUgdG8gY3N2IGNvbW1hIHNlcGFyYXRlZCBmaWxlIHdpdGggaGVhZGVyIHVzZXIgaW5wdXQgcHl0aG9uIGFuZCBzdHJlYW1saXRI_CZQrwRY6yRwBHgBkAEAmAHaAaAB-w2qAQYxLjEzLjG4AQPIAQD4AQGYAgOgAvoBwgIKEAAYsAMY1gQYR8ICBxAjGLACGCeYAwCIBgGQBgSSBwMxLjKgB-0osgcDMC4yuAf1AcIHBTAuMi4xyAcH&sclient=gws-wiz-serp
# if st.button("Save Data"):
#         if endpoint and age and population: # Ensure all fields are filled
#             save_data(file, df_new_patient_dataframe)
#         else:
#             st.warning("Please fill in all fields.")
#
#
#
# load_data = os.path.join(PATH, file)
#
# new_data = pd.read_csv(load_data)
categorical_features = ['endpoint', 'population']



new_data = df_new_patient_dataframe

for c in new_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    new_data[c] =  pt.fit_transform(np.array(new_data[c]).reshape(-1, 1))

encoders = {}
for col in categorical_features:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col])
        encoders[col] = le
# encoders = {}
# for col in categorical_features:
#
#         new_data[col] = le.transform(new_data[col])
#         encoders[col] = le



st.write(new_data)



filename =  'trained_model.joblib'
file = os.path.join(PATH, filename)

try:
    #loaded_model = joblib.load(file)
    loaded_model = joblib.load(open(filename, 'rb'))
except FileNotFoundError:
    st.write("file not found", file)
except Exception as e:
    st.write("an error occured when loading", file)




predictions = loaded_model.predict(new_data)


st.write(predictions)

# file_name_clustering_data =  "clusters.csv"
# load_clusters = os.path.join(PATH, file_name_clustering_data)
#
# df_clusters = pd.read_csv(load_clusters)
#
#
#
# classes = df_clusters['target_protocol'].unique().tolist()
#
#
# class_dictionary = {index: value for index, value in enumerate(classes)}
# st.write("class dictionary", class_dictionary)
#
# st.write("new predictions:")
# for i in predictions:
#     class_column = class_dictionary[i]
#     st.write("target number", i, "target name", class_column)