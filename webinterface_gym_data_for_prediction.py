import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator
import os
import streamlit as st
import pandas as pd


translator = Translator()
english_spanish_column_dictionary = {}
english_spanish_value_dictionary = {}

conn = st.connection("postgresql", type="sql")