import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("data/products_05_13.csv")
    df.replace("NaN", np.nan, inplace=True)
    return df

art_df = load_data()

filter_columns_config = [
    {"label": "Style", "col": "style"},
    {"label": "Category", "col": "category"},
    {"label": "Class", "col": "class"},
    {"label": "Occasion", "col": "occasion"},
    {"label": "Orientation", "col": "orientation"},
    {"label": "Color", "col": "dominant_color_hex"},
    {"label": "Country of Origin", "col": "country_of_origin"},
]

def get_filter_options(df, config):
    opts = {}
    for f in config:
        vals = df[f["col"]].dropna().unique().tolist()
        opts[f["col"]] = ["Any"] + sorted(v for v in vals if pd.notna(v))
    return opts

filter_options = get_filter_options(art_df, filter_columns_config)
