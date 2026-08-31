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
PATH = '.'

file_name_clustering_data =  "DataForClustering.csv"
load_cluster_data = os.path.join(PATH, file_name_clustering_data)

df_clustering_data = pd.read_csv(load_cluster_data)
df_clustering_data = df_clustering_data.drop(['men_ratio', 'category_men_ratio'], axis=1)
kprot_data = df_clustering_data.copy()

elbow_scores = dict()

for c in df_clustering_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

streamlit_kprot_data = st.data_editor(kprot_data)

### categorical_indices = [0, 1, 2, 4, 5]
categorical_indices = [0, 1, 2, 3, 5, 6, 9]

K = range(2,10)
for k in K:
    kproto = KPrototypes(n_clusters=k, init='Cao', n_jobs=4)
    clusters = kproto.fit(streamlit_kprot_data, categorical=categorical_indices)
    elbow_scores[k] = clusters.cost_

df_elbow_scores = pd.DataFrame(list(elbow_scores.items()), index=['2', '3', '4', '5', '6', '7', '8', '9'])
df_elbow_scores.columns = ['Cluster', 'Value']
print(df_elbow_scores)
print(df_elbow_scores.info())



st.scatter_chart(data=df_elbow_scores, x='Cluster', y='Value', x_label='Cluster Label', y_label='Cost')
