import os
import streamlit as st
import pandas as pd
from kmodes.kprototypes import KPrototypes

import numpy as np
PATH = '.'


file_name = "features.csv"
load_metabolic = os.path.join(PATH, file_name)


file_name_clustering_data =  "DataForClustering.csv"
load_cluster_data = os.path.join(PATH, file_name_clustering_data)

df_features_metabolic = pd.read_csv(load_metabolic)

df_clustering_data = pd.read_csv(load_cluster_data)

print("Testing")


#### June 27 2025
df_clustering_data = df_clustering_data.drop(['men_ratio', 'category_men_ratio'], axis=1)
####


#### June 27 2025 stremalite coding

edited_df = st.data_editor(df_clustering_data)
st.write("Here is the dataframe that will be used for clustering")
st.write(edited_df)
categorical_indices = [0, 1, 2, 4, 5]
kproto = KPrototypes( init='Cao', verbose=1)
clusters = kproto.fit_predict(edited_df, categorical=categorical_indices)
#Print the results
st.write(clusters)
print(clusters)

##print(edited_df.info())
######

exit()

filtered_df = df_features_metabolic
#filtered_df = df_features_metabolic[(df_features_metabolic['age'] > 40  )  & (df_features_metabolic['age'] < 50)  ]
## filtered_df = df_features_metabolic[(df_features_metabolic['age'] == 50.85) ]
##filtered_df = df_features_metabolic[(df_features_metabolic['study'] == "Abdelbasset 2020") ]

Filtered_Type_Exercise_Population_Endpoint = filtered_df[(filtered_df['endpoint'] == 'Fasting Glucose') &
                                                        #(filtered_df['Mean_MICT'] < 0) &


                                                        (filtered_df['population'] == 'T2D') &
                                                         (filtered_df['category_age'] == '> 50 y')  ]
                                                            # &
                                                       #  (filtered_df['Mean_HIIE'] < 0)]
pd.options.display.max_colwidth = 100
pd.set_option('display.max_columns', None)
##print(len(filtered_df))
print(filtered_df.info())
##print(filtered_df.head(50))
print(Filtered_Type_Exercise_Population_Endpoint)

mean_effect = Filtered_Type_Exercise_Population_Endpoint.groupby('category_age')['Mean_HIIE', 'Mean_MICT'].mean()
#### June 27 2025
#### print(mean_effect)
####
new_dataframe = (mean_effect).reset_index()

#### June 27 2025 stremalite coding
#from streamlit_jupyter import StreamlitPatcher, tqdm
#sp = StreamlitPatcher()
#sp.jupyter()  # register patcher with streamlit
st_df = st.data_editor(new_dataframe)
st.write("Here is the compared affects dataframe")
st.write(st_df)
####



print(new_dataframe)





one_age_bracket_dataframe = new_dataframe.filter(['Mean_HIIE', 'Mean_MICT'])


df_melted = one_age_bracket_dataframe.melt(var_name='Type_Training', value_name='mean_effect')

print(df_melted)

# print(new_dataframe.info())
# filtered_age_bracket_effect = new_dataframe['category_age'] == '30 - 50 y'
import plotly.express as px

fig = px.bar(df_melted, x='Type_Training', y='mean_effect', title = "> 50 y Overweight/obese Compared Effect of Both "
                                                                    "Exercise Protocols on Fasting Glucose")

fig.show()
exit()

Filtered_Type_Exercise_Population_Endpoint = filtered_df[(filtered_df['endpoint'] == 'Fasting Glucose') &
                                                        # (filtered_df['Mean_MICT'] < 0) &


                                                         (filtered_df['population'] == 'Healthy') ]
                                                         #
                                                         #(filtered_df['Mean_HIIE'] < 0)]
mean_effect = Filtered_Type_Exercise_Population_Endpoint.groupby('category_age')['Mean_HIIE', 'Mean_MICT'].mean()
print(mean_effect)

new_dataframe = (mean_effect).reset_index()

fig = px.bar(new_dataframe, x='category_age', y='Mean_MICT', title = "Effects of MICT by Age Brackets on Fasting Glucose in Healthy People")
##fig = px.bar(new_dataframe, x='Mean_MICT', y='category_age', orientation= 'h')

fig.show()

fig = px.bar(new_dataframe, x='category_age', y='Mean_HIIE', title = "Effects of HIIE by Age Brackets on Fasting Glucose in Healthy People")

fig.show()

print(new_dataframe)


pd.options.display.float_format = '{:.2%}'.format
print(df_features_metabolic.isnull().sum()/len(filtered_df))

