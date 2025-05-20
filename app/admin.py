import streamlit as st
from app.data_utils import art_df


def render() -> None:
    st.set_page_config(page_title="Admin", layout="wide", page_icon="⚙️")
    st.header("Admin Tools")
    st.markdown("Basic catalog info and diagnostics.")

    st.write(f"Dataset contains **{len(art_df)}** products.")
    st.write("Columns:", ", ".join(art_df.columns))
    st.dataframe(art_df.head())
