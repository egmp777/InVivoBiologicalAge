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

file_name_clustering_data =  "DataForClustering_With_CategoryDuration.csv"
load_cluster_data = os.path.join(PATH, file_name_clustering_data)


df_clustering_data = pd.read_csv(load_cluster_data)


df_clustering_data = df_clustering_data.drop(['men_ratio'], axis=1)


kprot_data = df_clustering_data.copy()

for c in df_clustering_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

elbow_scores = dict()
##categorical_indices = [0, 1, 2, 4, 5]
#### Changed dataset data On July 3rd to include `study`
#### Changed dataset data On July 3rd to include `desired effect`
####
##categorical_indices = [0, 1, 2, 5, 8]





streamlit_kprot_data = st.data_editor(kprot_data)




# <editor-fold desc="dropping 'desired_effect', 'Mean_HIIE', 'Mean_MICT'">
#### JULY 9 2025 ####
streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_HIIE'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)
streamlit_kprot_data['MICT_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_MICT'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_MICT'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)

#
# streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if row['Mean_HIIE'] < 0
#                             and row['desired_effect'] == 'decrease' else 0, axis=1)
# streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
#                             and row['desired_effect'] == 'decrease') or (row['Mean_HIIE'] > 0
#                                                                          and row['desired_effect'] == 'increase') else 0, axis=1)
# streamlit_kprot_data['MICT_success'] = streamlit_kprot_data.apply(lambda row: 1 if row['Mean_MICT'] < 0
#                             and row['desired_effect'] == 'decrease' else 0, axis=1)
streamlit_kprot_data = streamlit_kprot_data.drop(['desired_effect', 'Mean_HIIE', 'Mean_MICT'], axis=1)
st.write("data without Mean_HIIE, Mean_MICT, desired_effect", streamlit_kprot_data)


# </editor-fold>

# region includng the 'MICT_success' and 'HIIE_success'] columns in the categorical set
# categorical_indices = [0, 1, 2, 4, 5,6 ]
# endregion

# <editor-fold desc="Description">
# streamlit_kprot_data['target_protocol'] = streamlit_kprot_data.apply(lambda row: 1 if (row['HIIE_success'] == 1
#                             and row['MICT_success'] == 0)  else (2  if row['HIIE_success'] == 0
#                             and row['MICT_success'] == 1 else 3 ), axis =1)
# # region adding a new column and dropping the HIIE_Success and `MICT_success`
#
# for index, row in streamlit_kprot_data.iterrows():
#     if (row['HIIE_success'] == 1
#             and row['MICT_success'] == 0):
#         row['target_protocol'] = 0
#     elif (row['HIIE_success'] == 0
#             and row['MICT_success'] == 1):
#         row['target_protocol'] = 1
#     elif (row['HIIE_success'] == 1
#             and row['MICT_success'] == 1):
#         row['target_protocol'] = 2
#     else:
#         row['target_protocol'] = 3
# </editor-fold>

def target_protocol_value(row):
    if (row['HIIE_success'] == 1
            and row['MICT_success'] == 0):
        return 0
    elif (row['HIIE_success'] == 0
          and row['MICT_success'] == 1):
        return  1
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
st.write(streamlit_kprot_data)

# categorical_indices = [0, 1, 2, 4, 5 ]
# endregion


categorical_indices = [0, 1, 2, 3, 4, 5]

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
