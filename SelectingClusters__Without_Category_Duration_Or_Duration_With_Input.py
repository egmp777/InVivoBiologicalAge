import os
import streamlit as st
import pandas as pd
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
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
import plotly.express as px

PATH = '.'

PATH = '.'

## file_name_clustering_data =  "New_DataSet_Using_Gretel.csv"
file_name_clustering_data = "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"
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




K = range(2,12)
for k in K:
    kmodes = KModes(n_clusters=k, init='Cao', n_jobs=4)
    clusters = kmodes.fit(streamlit_kprot_data, categorical=categorical_indices)
    elbow_scores[k] = clusters.cost_

df_elbow_scores = pd.DataFrame(list(elbow_scores.items()), index=['2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])
df_elbow_scores.columns = ['Cluster', 'Value']
print(df_elbow_scores)
print(df_elbow_scores.info())



# <editor-fold desc="Line Chart without dots">
##st.scatter_chart(data=df_elbow_scores, x='Cluster', y='Value', x_label='Cluster Label', y_label='Cost')
##st.line_chart(data=df_elbow_scores, x='Cluster', y='Value', x_label='Cluster Label', y_label='Cost')
##st.update_traces(mode="lines+markers")
# </editor-fold>

fig = px.line(df_elbow_scores, x='Cluster', y='Value',  title='Elbow Curve')
fig.update_layout(clickmode='select')
fig.update_traces(mode="lines+markers")
event = st.plotly_chart(fig,  on_select="rerun")

### 3 clusters according to elbow chart
