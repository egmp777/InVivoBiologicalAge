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

PATH = '.'


file_name_clustering_data = "Data_NHANES_Demographics.csv"
load_demographic_data = os.path.join(PATH, file_name_clustering_data)


df_data = pd.read_csv(load_demographic_data)
