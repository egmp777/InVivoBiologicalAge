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
from sklearn import svm

PATH = '.'

##file_name_clustering_data =  "New_DataSet_Using_Gretel.csv"

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


kprot_data = df_clustering_data.copy()

for c in df_clustering_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

##categorical_indices = [0, 1, 2, 4, 5]
#### Changed dataset data On July 3rd to include `study`
#### Changed dataset data On July 3rd to include `desired effect`
####
##categorical_indices = [0, 1, 2, 5, 8]





streamlit_kprot_data = st.data_editor(kprot_data)

st.write(streamlit_kprot_data)


#### July 4 will create two new columns with the boolean value of comparing the desired effect with real effect
#### with HIIE and MICT

streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
                            and row['desired_effect'] == 'decrease') | (row['Mean_HIIE'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)
streamlit_kprot_data['MICT_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_MICT'] < 0
                            and row['desired_effect'] == 'decrease') | (row['Mean_MICT'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)

st.write("Number of HIIE success", streamlit_kprot_data['HIIE_success'].sum())
st.write("Number of MICT success", streamlit_kprot_data['MICT_success'].sum())

st.write(streamlit_kprot_data)


## July 21

data_for_charts = streamlit_kprot_data
# <editor-fold desc="dropping 'desired_effect', 'Mean_HIIE', 'Mean_MICT'">
#### JULY 9 2025 ####
streamlit_kprot_data = streamlit_kprot_data.drop(['desired_effect', 'Mean_HIIE', 'Mean_MICT', 'duration'], axis=1)

st.write("data without Mean_HIIE, Mean_MICT, desired_effect", streamlit_kprot_data)





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
st.write(streamlit_kprot_data)
#exit(0)


streamlit_kprot_data = streamlit_kprot_data.drop(['HIIE_success', 'MICT_success'], axis=1)
st.write(streamlit_kprot_data)

streamlit_kprot_data['target_protocol']= streamlit_kprot_data['target_protocol'].astype(str) + \
                                         ' ' + streamlit_kprot_data['type_exercise'].astype(str)

streamlit_kprot_data = streamlit_kprot_data.drop(['type_exercise'], axis=1)
st.write("This is the final dataframe for clustering:", streamlit_kprot_data)

categorical_indices = [0, 1, 3]
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
st.write("streamlit_kprot_data", streamlit_kprot_data)

#### July 10
### After Running SelectingClusters.py, we can see that the optimum number of clusters is 5
kproto = KPrototypes(n_clusters= 4, init='Cao')
clusters = kproto.fit_predict(streamlit_kprot_data, categorical=categorical_indices)
st.write("Cost of model with 4 clusters: ", kproto.cost_)

clusters_df = streamlit_kprot_data
clusters_df['Cluster'] = clusters

clusters_df.to_csv('clusters.csv', index = False)

filename = 'trained_cluster_model.pkl'
with open(filename, 'wb') as file:
        pickle.dump(kproto, file)

filename = 'trained_cluster_model.joblib'
with open(filename, 'wb') as file:
    joblib.dump(kproto, file)


f, axs = plt.subplots(1,2,figsize = (25,5))
sns.countplot(x=clusters_df['Cluster'],order=clusters_df['Cluster'].value_counts().index,hue=clusters_df['target_protocol'],ax=axs[0],palette='rainbow')
sns.countplot(x=clusters_df['population'],order=clusters_df['population'].value_counts().index,hue=clusters_df['target_protocol'],ax=axs[1],palette='rainbow')


plt.tight_layout
st.pyplot(f)

f2, axs2 = plt.subplots(figsize = (20,5))
sns.countplot(x=clusters_df['endpoint'],order=clusters_df['endpoint'].value_counts().index,hue=clusters_df['target_protocol'],ax=axs2,palette='rainbow')
plt.tight_layout
st.pyplot(f2)


f3, axs3 = plt.subplots(figsize = (20,5))
sns.countplot(x=clusters_df['endpoint'],order=clusters_df['endpoint'].value_counts().index,hue=clusters_df['Cluster'],ax=axs3,palette='rainbow')
##plt.tight_layout
st.pyplot(f3)

# <editor-fold desc="Description">
# f4, axs4 = plt.subplots(figsize = (20,5))
# sns.countplot(x=clusters_df['endpoint'],order=clusters_df['endpoint'].value_counts().index,hue=clusters_df['duration'],ax=axs4,palette='rainbow')
# ##plt.tight_layout
# st.pyplot(f4)
# </editor-fold>





centroids = kproto.cluster_centroids_
print("Cluster Centroids:")
print(centroids)
st.write(centroids)

# Cluster analysis
for cluster in range(kproto.n_clusters):
    st.write('\nCluster', cluster)
    cluster_data = clusters_df[clusters_df['Cluster'] == cluster]
    st.write(cluster_data)
    ##describe(include='all')






data_for_charts['labels'] = kproto.labels_
st.write(data_for_charts['labels'])

# streamlit_kprot_data.groupby('labels').agg(['median' ,'mean']).T
# streamlit_kprot_data.groupby('labels').agg(['count']).T

st.write("This is kprot_data", streamlit_kprot_data)



# data_for_charts = data_for_charts[data_for_charts["category_age"] != '< 30 y']
df_profiles = data_for_charts.groupby(['labels', 'endpoint', 'population']).aggregate({'duration':'mean','Mean_HIIE':'mean',

                                                                                         'Mean_MICT':'mean','endpoint':'count'})


df_profiles = data_for_charts.groupby(['labels', 'endpoint']).agg(
                        duration_mean = ('duration','mean'),
                        mean_HIIE= ('Mean_HIIE','mean'),
                        mean_MICT = ('Mean_MICT', 'mean'),
                        count_endpoint = ('endpoint', 'count'))

new_df_profiles = df_profiles.reset_index()


st.write(new_df_profiles)


fig = px.scatter(
    new_df_profiles,
    x="labels",
    y="mean_HIIE",
    color="endpoint",
    size="count_endpoint",
    hover_data=["duration_mean"],
)

event = st.plotly_chart(fig,  on_select="rerun")



event.selection


fig2 = px.scatter(
    new_df_profiles,
    x="labels",
    y="mean_MICT",
    color="endpoint",
    size="count_endpoint",
    hover_data=["duration_mean"],
)

event2 = st.plotly_chart(fig2,  on_select="rerun")
event2.selection

st.scatter_chart(
     new_df_profiles,
     x="labels",
     y="mean_HIIE",
     color="endpoint",
     size="count_endpoint"
)


# df_profiles = data_for_charts.groupby(['labels', 'endpoint', 'population']).agg(
#                         duration_mean = ('duration','mean'),
#                         mean_HIIE= ('Mean_HIIE','mean'),
#                         mean_MICT = ('Mean_MICT', 'mean'),
#                         count_study = ('study', 'count'),
#                         percent_sucess_HIIE = ('HIIE_success', lambda x: float(x.sum()/x.count())))
#
#
# new_df_profiles = df_profiles.reset_index()
#
#
# fig3 = px.scatter(
#     new_df_profiles,
#     x="labels",
#     y="mean_HIIE",
#     color="endpoint",
#     size="count_study",
#     hover_data=['population'],
# )


# event3 = st.plotly_chart(fig3,  on_select="rerun")
# event3.selection



# fig3a = px.scatter(
#     new_df_profiles,
#     x="labels",
#     y= 'mean_HIIE',
#     color="endpoint",
#     size="count_study",
#     hover_data=['percent_sucess_HIIE'],
# )
# event3a = st.plotly_chart(fig3a,  on_select="rerun")
# event3a.selection
#
#
#
#
# new_df_profiles = df_profiles.reset_index()
# fig4 = px.scatter(
#     new_df_profiles,
#     x="labels",
#     y="mean_MICT",
#     color="endpoint",
#     size="count_study",
#     hover_data=['population'],
# )
#
# event4 = st.plotly_chart(fig4,  on_select="rerun")
# event4.selection
#
#
#
clusters_and_exercise_type_duration_age = data_for_charts.filter(['duration', 'type_exercise', 'category_age', 'labels'])

clusters_and_exercise_type_duration_age['Cycling'] = data_for_charts.apply(lambda row: 1 if row['type_exercise'] == 'Cycling'
                            else 0, axis=1)
clusters_and_exercise_type_duration_age['Running'] = data_for_charts.apply(lambda row: 1 if row['type_exercise'] == 'Running'
                            else 0, axis=1)

st.write(clusters_and_exercise_type_duration_age)


df_exercise_type_per_label = clusters_and_exercise_type_duration_age.groupby(['labels']).agg(
                        count_cycling = ('Cycling', 'sum'),
                        count_running = ('Running', 'sum'),
)

flipped_df_exercise_type_per_label =      df_exercise_type_per_label.reset_index()

flipped_df_exercise_type_per_label.filter(['labels', 'count_cycling', 'count_running'])

st.bar_chart(flipped_df_exercise_type_per_label)




# c = (
#     alt.Chart(new_df_profiles)
#     .mark_point()
#     .encode(
#         x="labels",
#         y=alt.Y('mean_HIIE', scale=alt.Scale(domain=[-1, 2])),
#         size="count_endpoint",
#         color="endpoint",
#
#     )
# )
# st.altair_chart(c, use_container_width=True)

##st.scatter_chart(data=kprot_data,  x='Mean_HIIE', y='duration', x_label='Mean_HIIE', y_label='duration', color = 'color')

