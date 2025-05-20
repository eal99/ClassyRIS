import altair as alt
import streamlit as st
import pandas as pd

from app.data_utils import art_df
from app import openai_utils


def placeholder_ai_tools():
    example_desc = art_df["description"].dropna().iloc[0]
    st.subheader("AI‑Generated Summary")
    st.write(openai_utils.summarize_description(example_desc))
    tags = openai_utils.generate_tags(example_desc)
    st.write("Suggested tags:", ", ".join(tags))


def render() -> None:
    st.set_page_config(page_title="Analytics", layout="wide", page_icon="📊")
    st.header("Analytics Dashboard")
    st.markdown("Overview of available products and pricing.")

    total_products = len(art_df)
    st.metric("Total products", total_products)

    if "ecom_price" in art_df.columns:
        prices = pd.to_numeric(art_df["ecom_price"], errors="coerce").dropna()
        if not prices.empty:
            st.metric("Average price", f"${prices.mean():.2f}")
            bins = pd.cut(prices, bins=10)
            counts = bins.value_counts().sort_index().reset_index()
            counts.columns = ["range", "count"]
            st.subheader("Price distribution")
            chart = alt.Chart(counts).mark_bar().encode(
                x="range:N", y="count:Q", tooltip=["range", "count"]
            )
            st.altair_chart(chart, use_container_width=True)

    st.subheader("Top Categories")
    top_cats = art_df["category"].value_counts().head(10)
    st.bar_chart(top_cats)

    with st.expander("AI Insights (beta)"):
        placeholder_ai_tools()
