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

file_name_clustering_data =  "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"
load_cluster_data = os.path.join(PATH, file_name_clustering_data)


df_clustering_data = pd.read_csv(load_cluster_data)



# <editor-fold desc="Removing Outliers">
df_clustering_data.drop(df_clustering_data[(df_clustering_data['Mean_HIIE'] > df_clustering_data['Mean_HIIE'].quantile(0.975)) |
                                           (df_clustering_data['Mean_HIIE'] <
                                            df_clustering_data['Mean_HIIE'].quantile(0.025))].index,inplace=True)
df_clustering_data.drop(df_clustering_data[(df_clustering_data['Mean_MICT'] > df_clustering_data['Mean_MICT'].quantile(0.975)) |
                                           (df_clustering_data['Mean_MICT'] <
                                            df_clustering_data['Mean_MICT'].quantile(0.025))].index,inplace=True)


df_clustering_data.drop(df_clustering_data[(df_clustering_data['age'] > df_clustering_data['age'].quantile(0.975)) |
                                           (df_clustering_data['age'] <
                                            df_clustering_data['age'].quantile(0.025))].index,inplace=True)
df_clustering_data.drop(df_clustering_data[(df_clustering_data['duration'] > df_clustering_data['duration'].quantile(0.975)) |
                                           (df_clustering_data['duration'] <
                                            df_clustering_data['duration'].quantile(0.025))].index,inplace=True)

# </editor-fold>
elbow_scores = dict()

kprot_data = df_clustering_data.copy()

for c in df_clustering_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))




streamlit_kprot_data = st.data_editor(kprot_data)





#### July 4 will create two new columns with the boolean value of comparing the desired effect with real effect
#### with HIIE and MICT

streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_HIIE'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)
streamlit_kprot_data['MICT_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_MICT'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_MICT'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)


# <editor-fold desc="dropping 'desired_effect', 'Mean_HIIE', 'Mean_MICT'">
#### JULY 9 2025 ####

streamlit_kprot_data = streamlit_kprot_data.drop(['desired_effect', 'Mean_HIIE', 'Mean_MICT', 'duration'], axis=1)

st.write("data without Mean_HIIE, Mean_MICT, desored_effect", streamlit_kprot_data)





# </editor-fold>

# region includng the 'MICT_success' and 'HIIE_success'] columns in the categorical test

# endregion

st.write("Here is the dataframe that will be used for clustering")
st.write(streamlit_kprot_data)




# <editor-fold desc="Description">
# region adding a new column and dropping the HIIE_Success and `MICT_success`
# streamlit_kprot_data['target_protocol'] = streamlit_kprot_data.apply(lambda row: 1 if (row['HIIE_success'] == 1
#                             and row['MICT_success'] == 0)  else (2  if row['HIIE_success'] == 0
#                             and row['MICT_success'] == 1 else 3 ), axis =1)
# </editor-fold>

def target_protocol_value(row):
    if (row['HIIE_success'] == 1
            and row['MICT_success'] == 0):
        return 0
    elif (row['HIIE_success'] == 0
          and row['MICT_success'] == 1):
        return 1
    elif (row['HIIE_success'] == 1
          and row['MICT_success'] == 1):
        return 2
    else:
        return 3


streamlit_kprot_data['target_protocol'] = streamlit_kprot_data.apply(target_protocol_value, axis=1)

streamlit_kprot_data = streamlit_kprot_data.drop(['HIIE_success', 'MICT_success'], axis=1)
st.write(streamlit_kprot_data)

streamlit_kprot_data['target_protocol']= streamlit_kprot_data['target_protocol'].astype(str) + \
                                         ' ' + streamlit_kprot_data['type_exercise'].astype(str)

streamlit_kprot_data = streamlit_kprot_data.drop(['type_exercise'], axis=1)
st.write("This is the final dataframe for clustering:", streamlit_kprot_data)

categorical_indices = [0, 1, 3]
# categorical_indices = [0, 1, 2, 4, 5 ]
# endregion

kproto = KPrototypes(n_clusters= 15, init='Cao', n_jobs = 4)
clusters = kproto.fit_predict(streamlit_kprot_data, categorical=categorical_indices)
st.write(pd.Series(clusters).value_counts())
# colors = {'30 - 50 y': 'red', '> 50 y': 'green', '< 30 y': 'blue'}
# kprot_data['color'] = [colors[group] for group in streamlit_kprot_data['category_age']]
# st.write(kprot_data['color'])
#color_list = [colors[group] for group in kprot_data['category_age']]
#st.write(kprot_data['duration'].count())
#st.write(kprot_data['category_age'].count())
#st.write(len(color_list))

#### July 10
### After Running SelectingClusters.py, we can see that the optimum number of clusters is 4
kproto = KPrototypes(n_clusters= 4, init='Cao')
clusters = kproto.fit_predict(streamlit_kprot_data, categorical=categorical_indices)
st.write("Cost of model with 4 clusters: ", kproto.cost_)

clusters_df = streamlit_kprot_data
clusters_df['Cluster'] = clusters

clusters_df.to_csv('new_data.csv', index = False)