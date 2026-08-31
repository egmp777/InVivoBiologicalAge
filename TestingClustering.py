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
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

PATH = '.'

file_name_clustering_data =  "DataForClustering.csv"
load_cluster_data = os.path.join(PATH, file_name_clustering_data)


df_clustering_data = pd.read_csv(load_cluster_data)

corr = df_clustering_data.corr()
cmap = sns.diverging_palette(2, 15, as_cmap=True)
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f5, axs5 = plt.subplots(figsize=(30, 10))
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.7, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});
st.pyplot(f5)

df_clustering_data = df_clustering_data.drop(['men_ratio', 'category_men_ratio'], axis=1)


kprot_data = df_clustering_data.copy()

for c in df_clustering_data.select_dtypes(exclude='object').columns:
    pt = PowerTransformer()
    kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

##categorical_indices = [0, 1, 2, 4, 5]
#### Changed dataset data On July 3rd to include `study`
#### Changed dataset data On July 3rd to include `desired effect`
categorical_indices = [0, 1, 2, 3, 5, 6, 9]




streamlit_kprot_data = st.data_editor(kprot_data)

#### July 4 will create two new columns with the boolean value of comparing the desired effect with real effect
#### with HIIE and MICT

streamlit_kprot_data['HIIE_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_HIIE'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_HIIE'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)
streamlit_kprot_data['MICT_success'] = streamlit_kprot_data.apply(lambda row: 1 if (row['Mean_MICT'] < 0
                            and row['desired_effect'] == 'decrease') or (row['Mean_MICT'] > 0
                                                                         and row['desired_effect'] == 'increase') else 0, axis=1)


####

st.write("Here is the dataframe that will be used for clustering")
st.write(streamlit_kprot_data)
kproto = KPrototypes(n_clusters= 15, init='Cao', n_jobs = 4)
clusters = kproto.fit_predict(streamlit_kprot_data, categorical=categorical_indices)
st.write(pd.Series(clusters).value_counts())
colors = {'30 - 50 y': 'red', '> 50 y': 'green', '< 30 y': 'blue'}
kprot_data['color'] = [colors[group] for group in streamlit_kprot_data['category_age']]
st.write(kprot_data['color'])
#color_list = [colors[group] for group in kprot_data['category_age']]
#st.write(kprot_data['duration'].count())
#st.write(kprot_data['category_age'].count())
#st.write(len(color_list))

#### July 1st
### After Running SelectingClusters.py, we can see that the optimum number of clusters is 6
kproto = KPrototypes(n_clusters= 6, init='Cao')
clusters = kproto.fit_predict(streamlit_kprot_data, categorical=categorical_indices)
st.write("Cost of model with 6 clusters: ", kproto.cost_)

streamlit_kprot_data['labels'] = kproto.labels_
st.write(streamlit_kprot_data['labels'])

# streamlit_kprot_data.groupby('labels').agg(['median' ,'mean']).T
# streamlit_kprot_data.groupby('labels').agg(['count']).T

st.write("This is kprot_data", streamlit_kprot_data)



streamlit_kprot_data = streamlit_kprot_data[streamlit_kprot_data["category_age"] != '< 30 y']
df_profiles = streamlit_kprot_data.groupby(['labels', 'endpoint', 'population', 'category_age']).aggregate({'duration':'mean','Mean_HIIE':'mean',

                                                                                         'Mean_MICT':'mean','endpoint':'count'})


df_profiles = streamlit_kprot_data.groupby(['labels', 'endpoint']).agg(
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


df_profiles = streamlit_kprot_data.groupby(['labels', 'endpoint', 'population']).agg(
                        duration_mean = ('duration','mean'),
                        mean_HIIE= ('Mean_HIIE','mean'),
                        mean_MICT = ('Mean_MICT', 'mean'),
                        count_study = ('study', 'count'),
                        percent_sucess_HIIE = ('HIIE_success', lambda x: float(x.sum()/x.count())))


new_df_profiles = df_profiles.reset_index()


fig3 = px.scatter(
    new_df_profiles,
    x="labels",
    y="mean_HIIE",
    color="endpoint",
    size="count_study",
    hover_data=['population'],
)


event3 = st.plotly_chart(fig3,  on_select="rerun")
event3.selection

fig3a = px.scatter(
    new_df_profiles,
    x="labels",
    y= 'mean_HIIE',
    color="endpoint",
    size="count_study",
    hover_data=['percent_sucess_HIIE'],
)
event3a = st.plotly_chart(fig3a,  on_select="rerun")
event3a.selection




new_df_profiles = df_profiles.reset_index()
fig4 = px.scatter(
    new_df_profiles,
    x="labels",
    y="mean_MICT",
    color="endpoint",
    size="count_study",
    hover_data=['population'],
)

event4 = st.plotly_chart(fig4,  on_select="rerun")
event4.selection



clusters_and_exercise_type_duration_age = streamlit_kprot_data.filter(['duration', 'type_exercise', 'category_age', 'labels'])

clusters_and_exercise_type_duration_age['Cycling'] = streamlit_kprot_data.apply(lambda row: 1 if row['type_exercise'] == 'Cycling'
                            else 0, axis=1)
clusters_and_exercise_type_duration_age['Running'] = streamlit_kprot_data.apply(lambda row: 1 if row['type_exercise'] == 'Running'
                            else 0, axis=1)

st.write(clusters_and_exercise_type_duration_age)


df_exercise_type_per_label = clusters_and_exercise_type_duration_age.groupby(['labels']).agg(
                        count_cycling = ('Cycling', 'sum'),
                        count_running = ('Running', 'sum'),
)

flipped_df_exercise_type_per_label =      df_exercise_type_per_label.reset_index()

flipped_df_exercise_type_per_label.filter(['labels', 'count_cycling', 'count_running'])

st.bar_chart(flipped_df_exercise_type_per_label)

input = st.text_input("Enter Input Data :")
st.write(input)


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

