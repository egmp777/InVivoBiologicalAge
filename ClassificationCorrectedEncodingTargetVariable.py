from sklearn.datasets import make_classification
from sklearn import svm

# source: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn

X, y = make_classification(
    n_features=6,
    n_classes=4,
    n_samples=391,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# end source

# <editor-fold desc="Source: https://medium.com/@shuv.sdr/na%C3%AFve-bayes-classification-in-python-f869c2e0dbf1">
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pickle
import joblib

# </editor-fold>

import matplotlib.pyplot as plt
import numpy as np

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
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

label_encoder =  preprocessing.LabelEncoder()

PATH = '.'

file_name_clustering_data =  "clusters.csv"
load_clusters = os.path.join(PATH, file_name_clustering_data)

df_clusters = pd.read_csv(load_clusters)

st.write("data with clusters to be used for train and test", df_clusters)

classes = df_clusters['target_protocol'].unique().tolist()
st.write('classes', classes)

class_dictionary = {index: value for index, value in enumerate(classes)}
st.write("class dictionary", class_dictionary)

## categorical_features = ['endpoint', 'population',  'target_protocol', 'Cluster' ]
categorical_features = ['endpoint', 'population', 'target_protocol']

# filename = "trained_model.pkl"
# loaded_model = pickle.load(open(filename, 'rb'))
# df_ml = df_clusters.apply(label_encoder.fit_transform)
# st.write(df_ml.describe())
# st.write(df_ml)

# filename =  'trained_model.joblib'
# file = os.path.join(PATH, filename)
#
# try:
#     #loaded_model = joblib.load(file)
#     loaded_model = joblib.load(open(filename, 'rb'))
# except FileNotFoundError:
#     st.wrire("file not found", file)
# except Exception as e:
#     st.write("an error occured when loading", file)
le = LabelEncoder()
df_ml = df_clusters
# Normalize continuous variables
# df_ml[['age']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    #st.write(col, df_ml[col])
    df_ml[col] = le.fit_transform(df_ml[col])
    st.write(col, df_ml[col])

    encoders[col] = le




with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

##joblib.dump(le, "label_encoder.joblib")


target_name = 'target_protocol'
data_target = df_ml[target_name]
df_ml=df_ml.drop([target_name], axis=1)
df_ml=df_ml.drop(['Cluster'], axis=1)

st.write("classifiable data dataframe", df_ml)
# df_ml.to_csv("classifiable encoded data", index= False)
#
# exit(0)

X_train, X_test, y_train, y_test = train_test_split(df_ml, data_target, test_size=0.3, random_state=42)
Xtrain, Xval, Ztrain, Zval = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

## predicting with naive bayes
gnb = GaussianNB()

gnb.fit(X_train, y_train)
st.write(X_train)


# X_train.to_csv('new_data.csv', index = False)

# Predict the labels for the test set
y_pred = gnb.predict(X_test)

filename = 'trained_model.joblib'
with open(filename, 'wb') as file:
    joblib.dump(gnb, file)

## predicting with svm
## Source: https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)
filename = 'trained_svm_model.joblib'
with open(filename, 'wb') as file:
    joblib.dump(gnb, file)


st.write("X_test:", X_test)

st.write(y_pred)

st.write(y_pred_svm)


accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write('Accuracy gnb:',  accuracy)
st.write("Accuracy svm: ", accuracy_svm )
st.write("X_test", X_test)
# st.write("Prediction for first test sample:", y_pred[0])
predicted_class_list = []
# for i in X_test['target_protocol']:
#     class_column = class_dictionary[i]
#
#     st.write("target number", i, "target name", class_column)

st.write(y_pred)
st.write(class_dictionary)

# for i in y_pred:
#     class_column = class_dictionary[i]
#     st.write("target number gnb", i, "target name gnb", class_column)
#
# for i in y_pred_svm:
#     class_column = class_dictionary[i]
#     st.write("target number gnb", i, "target name gnb", class_column)

# exit(0)

# decoded_df = X_test.copy()
# encoders.__delitem__("Cluster")
# for col, le in encoders.items():
#     decoded_df[col] = le.inverse_transform(X_test[col])
# st.write("decoded df", decoded_df)



# file_name_new_data =  "new_data_points.csv"
# load_data= os.path.join(PATH, file_name_clustering_data)
#
# df_new_data = pd.read_csv(load_data)
#
# categorical_features = ['endpoint', 'population' ]
#
# df_ml = df_new_data
# encoders = {}
# for col in categorical_features:
#         le = LabelEncoder()
#         df_ml[col] = le.fit_transform(df_ml[col])
#         encoders[col] = le
#
#


filename =  'trained_model.joblib'
file = os.path.join(PATH, filename)

try:
    #loaded_model = joblib.load(file)
    loaded_model = joblib.load(open(filename, 'rb'))
except FileNotFoundError:
    st.write("file not found", file)
except Exception as e:
    st.write("an error occured when loading", file)

new_data_file_name = "New_patient_data.csv"
loaded_new_data = os.path.join(PATH, new_data_file_name)

new_data = pd.read_csv(loaded_new_data)


categorical_features = ['endpoint', 'population']

with open('label_encoders.pkl', 'rb') as f:
    loaded_encoders = pickle.load(f)

with open('transformers.pkl', 'rb') as f:
    loaded_transformers = pickle.load(f)

categorical_features = ['endpoint', 'population']



for col in categorical_features:
        # st.write(col)
        # st.write("new_data", new_data[col])
        new_data[col] = loaded_encoders[col].transform(new_data[col])
        # le = LabelEncoder()
        # st.write(loaded_le.transform(["BMI", "VO2max"]))
        ##new_data[col] = loaded_le.transform(new_data[col])
        ##encoders[col] = le







# Normalize continuous variables
new_data[['age']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

predictions = loaded_model.predict(new_data)




st.write(predictions)

figb, ax = plt.subplots()
ax.hist(predictions, bins=20)

st.pyplot(figb)
# exit(0)
# st.write("new predictions:")

# for i in predictions:
#     class_column = class_dictionary[i]
#     st.write("target number", i, "target name", class_column)