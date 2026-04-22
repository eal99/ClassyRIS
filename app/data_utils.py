import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st


def _normalize_column(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "holiday_occasion_season": "occasion",
        "family_collection": "collection_name",
        "romance_copy": "page_romance",
        "frame_color_finish": "frame_color",
    }
    rename_map = {
        src: dst for src, dst in aliases.items()
        if src in df.columns and dst not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

@st.cache_data
def load_data():
    csv_path = os.getenv("PRODUCTS_CSV", "data/products_01_07.csv")
    if not Path(csv_path).exists():
        data_dir = Path("data")
        csv_files = sorted(data_dir.glob("*.csv"))
        if csv_files:
            csv_path = str(csv_files[-1])
    df = pd.read_csv(csv_path)
    df.columns = [_normalize_column(c) for c in df.columns]
    df = _apply_column_aliases(df)
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
        col = f["col"]
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique().tolist()
        opts[col] = ["Any"] + sorted(v for v in vals if pd.notna(v))
    return opts

filter_options = get_filter_options(art_df, filter_columns_config)


def get_image_url_by_sku(sku: str) -> str | None:
    if not sku or "sku" not in art_df.columns:
        return None
    sku_val = str(sku)
    rows = art_df[art_df["sku"].astype(str) == sku_val]
    if rows.empty:
        return None
    row = rows.iloc[0]
    for col in (
        "main_image_file",
        "image_url_image_file_1",
        "image_url_lifestyle_image_1",
        "image_url_diagram_image_1",
        "image_url_3_dimensional_image_1",
        "image_1_file",
    ):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return None
