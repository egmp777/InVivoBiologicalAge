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


PATH = '.'
file = "New_patient_data.csv"
load_data = os.path.join(PATH, file)

new_data = pd.read_csv(load_data)
categorical_features = ['endpoint', 'population']

for c in new_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    new_data[c] =  pt.fit_transform(np.array(new_data[c]).reshape(-1, 1))

encoders = {}
for col in categorical_features:
        le = LabelEncoder()
        new_data[col] = le.fit_transform(new_data[col])
        encoders[col] = le

st.write(new_data)



filename =  'trained_model.joblib'
file = os.path.join(PATH, filename)

try:
    #loaded_model = joblib.load(file)
    loaded_model = joblib.load(open(filename, 'rb'))
except FileNotFoundError:
    st.wrire("file not found", file)
except Exception as e:
    st.write("an error occured when loading", file)




predictions = loaded_model.predict(new_data)


st.write(predictions)

file_name_clustering_data =  "clusters.csv"
load_clusters = os.path.join(PATH, file_name_clustering_data)

df_clusters = pd.read_csv(load_clusters)



classes = df_clusters['target_protocol'].unique().tolist()


class_dictionary = {index: value for index, value in enumerate(classes)}
st.write("class dictionary", class_dictionary)

st.write("new predictions:")
for i in predictions:
    class_column = class_dictionary[i]
    st.write("target number", i, "target name", class_column)