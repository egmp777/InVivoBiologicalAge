import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import joblib

data_file = "gym_members_exercise_tracking.csv"
data_file_2 = "heart_rate_data.csv"
PATH = '.'

loaded_data = os.path.join(PATH, data_file)
loaded_data_2 = os.path.join(PATH, data_file_2)

pandas_data = pd.read_csv(loaded_data)
pandas_data2 = pd.read_csv(loaded_data_2)


pandas_data['vo2max'] = 15.3 * ( (220 - pandas_data['Age']) / pandas_data['Resting_BPM'] )
pandas_data2['vo2max1'] = 15.3 * ( (220 - pandas_data2['Age']) / pandas_data2['Resting Heart Rate Before'] )
pandas_data2['vo2max2'] = 15.3 * ( (220 - pandas_data2['Age']) / pandas_data2['Resting Heart Rate After'] )
pandas_data2['%_change_vo2max'] = (pandas_data2['vo2max2'] - pandas_data2['vo2max1'])/pandas_data2['vo2max1'] *100
st.write(pandas_data2)
pandas_data2 = pandas_data2.loc[pandas_data2['Age'] >= 45]
pandas_data2.loc[pandas_data2['Age'].between(45, 50, 'both'), 'Age Group'] = '45-50'
pandas_data2.loc[pandas_data2['Age'].between(50,55, 'right'), 'Age Group'] = '51-55'
pandas_data2.loc[pandas_data2['Age'].between(55,60 ,'right'), 'Age Group'] = '56-60'

aggegated_by_age_pandas_2 = pd.DataFrame(pandas_data2.groupby(['Age Group']).mean(), columns=['%_change_vo2max'])




##pandas_data['vo2_max'] = pandas_data['Max_BPM']/ pandas_data['Resting_BPM'] * 15.3


pandas_data = pandas_data.loc[pandas_data['Age'] >= 45]
pandas_data = pandas_data.loc[pandas_data['Workout_Type'] != 'Yoga']
pandas_data = pandas_data.loc[pandas_data['Experience_Level'] > 1]


pandas_data['duration_level'] = pandas_data.apply(lambda row: 'a' if (row['Session_Duration (hours)'] >= 0.5 and row['Session_Duration (hours)'] <= 1) else 'b', axis=1)

## Jan 8 2026
pandas_data['start_vo2max'] = pandas_data['vo2max']/(1.10)

pandas_data = pandas_data.drop(['Session_Duration (hours)'], axis = 1)
pandas_data['workout_plan'] = pandas_data['Workout_Type'] + '_' + pandas_data['duration_level'].astype(str)



pandas_data = pandas_data.drop([ 'Max_BPM', 'Workout_Type', 'duration_level', 'Water_Intake (liters)', 'Calories_Burned', 'Resting_BPM', 'BMI', 'Fat_Percentage'],  axis = 1)


st.write(pandas_data.isna().sum())
pandas_data = pandas_data.dropna()
st.write(pandas_data.isna().sum())

st.write(pandas_data)

from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(pandas_data)
categorical_columns = categorical_columns_selector(pandas_data)

target = pandas_data['workout_plan']
pandas_data = pandas_data.drop(['workout_plan'], axis = 1)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()


numerical_columns = numerical_columns_selector(pandas_data)
categorical_columns = categorical_columns_selector(pandas_data)


from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, DecisionTreeClassifier(criterion="entropy", max_depth=4))


from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    pandas_data, target, random_state=42
)

model.fit(data_train, target_train)

## model.predict(data_test)[:]

st.write(model.score(data_test, target_test))

filename = 'trained_model_decision_tree.joblib'
with open(filename, 'wb') as file:
    joblib.dump(model, file)