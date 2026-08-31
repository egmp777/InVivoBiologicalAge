from gretel_client.navigator_client import Gretel
gretel = Gretel(api_key="grtu3252508a87b41411a6fdfd4d6740c05c25833a979526f3cf1876ab628405799c")
from gretel_client.data_designer.columns import SamplerColumn
from gretel_client.data_designer.params import (
    GaussianSamplerParams,
    CategorySamplerParams,
    SamplerType,
    PersonSamplerParams,
    UniformSamplerParams

)
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

## Sources
## https://python.docs.gretel.ai/en/stable/data_designer.html#classical-building-column-types
## https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/data-designer-101/1-the-basics.ipynb#scrollTo=h0-c52YPgrsQ
## https://github.com/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/data-designer-101/3-seeding-with-a-dataset.ipynb
## https://colab.research.google.com/github/gretelai/gretel-blueprints/blob/main/docs/notebooks/data-designer/data-designer-101/1-the-basics.ipynb#scrollTo=sx8uVzsjPKJG

PATH = '.'

file_name_clustering_data =  "DataForClustering_Without_Study_Category_Duration_With_Age_with_Duration.csv"
load_data = os.path.join(PATH, file_name_clustering_data)



df_seed_data = pd.read_csv(load_data)

mean_age =  df_seed_data['age'].mean()
st.write("mean age from my dataset", mean_age)
std_age = df_seed_data['age'].std()
st.write("standard deviation of age in my dataset", std_age)
aidd = gretel.data_designer.new(model_suite="apache-2.0")



aidd.add_column(
    SamplerColumn(
        name="endpoint",
        type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=df_seed_data['endpoint'].unique().tolist()),

    )
)

aidd.add_column(
    SamplerColumn(
        name="population",
        type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=df_seed_data['population'].unique().tolist()),

    )
)

aidd.add_column(
    SamplerColumn(
        name="age",
        type=SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(mean=df_seed_data['age'].mean(),
                                     stddev= df_seed_data['age'].std(), low = df_seed_data['age'].min()),
        convert_to="float",
    )
)

aidd.add_column(
    SamplerColumn(
        name="duration",
        type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=df_seed_data['duration'].unique().tolist()),
        convert_to="float",
    )
)

aidd.add_column(
    SamplerColumn(
        name="type_exercise",
        type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=df_seed_data['type_exercise'].unique().tolist()),

    )
)

aidd.add_column(
    SamplerColumn(
        name="Mean_HIIE",
        type = SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(mean=-0.825941449, stddev=5.737205101, low = -46.57, high = 30.6),

    )
)

aidd.add_column(
    SamplerColumn(
        name="Mean_MICT",
        type = SamplerType.GAUSSIAN,
        params=GaussianSamplerParams(mean=-1.170247214, stddev=5.864421017, low = -71.1, high = 23.6),

    )
)


aidd.add_column(
    SamplerColumn(
        name="desired_effect",
        type=SamplerType.CATEGORY,
        params=CategorySamplerParams(values=df_seed_data['desired_effect'].unique().tolist()),

    )
)


workflow_run = aidd.create(num_records=500, name="creating_synthetic_study_data")
# aidd.add_column(
#     SamplerColumn(
#         name="person",  # This creates a nested object with all person attributes
#         type= SamplerType.PERSON,
#         params= PersonSamplerParams(
#             locale="en_US",
#             age_range=[22, 65],
#             state="CA"
#         )
#     )
# )

aidd.validate()

preview = aidd.preview()

st.write(preview.dataset.df.head())
st.write(preview.dataset.df)