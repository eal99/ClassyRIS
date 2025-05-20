# ui.py

import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from io import BytesIO
from PIL import Image
from app.embedding import get_image_embedding, get_text_embedding
from app.qdrant_utils import vector_search, hybrid_search

# ── Page config ──
st.set_page_config(
    page_title="Classy RIS/Text Search",
    layout="wide",
    page_icon="🎨"
)

# ── Load data ──
@st.cache_data
def load_data():
    df = pd.read_csv("data/products_05_13.csv")
    df.replace("NaN", np.nan, inplace=True)
    return df

art_df = load_data()

# ── Facet configuration ──
filter_columns_config = [
    {"label": "Style",             "col": "style"},
    {"label": "Category",          "col": "category"},
    {"label": "Class",             "col": "class"},
    {"label": "Occasion",          "col": "occasion"},
    {"label": "Orientation",       "col": "orientation"},
    {"label": "Color",             "col": "dominant_color_hex"},
    {"label": "Country of Origin", "col": "country_of_origin"},
]

def get_filter_options(df, config):
    opts = {}
    for f in config:
        vals = df[f["col"]].dropna().unique().tolist()
        opts[f["col"]] = ["Any"] + sorted(v for v in vals if pd.notna(v))
    return opts

filter_options = get_filter_options(art_df, filter_columns_config)

# ── Helpers ──
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))

def show_active_filters(filters):
    if not filters:
        return
    chips = [f"`{field}: {', '.join(map(str, vals))}`"
             for field, vals in filters.items()]
    st.markdown("**Active filters:** " + " &nbsp; ".join(chips))

def display_results(results):
    """Render search results in a grid and offer a CSV download."""
    if not results:
        st.warning(
            "No results found. Try broadening your query or removing some filters."
        )
        return

    # Convert results to a DataFrame for the download button
    df_results = pd.DataFrame(
        [
            {
                **(r.payload or {}),
                "score": getattr(r, "score", None),
            }
            for r in results
        ]
    )

    num_cols = 5
    for i in range(0, len(results), num_cols):
        cols = st.columns(num_cols)
        for idx, r in enumerate(results[i : i + num_cols]):
            pl = r.payload or {}
            img_url = pl.get("main_image_file")
            name = pl.get("product_name", "N/A")
            sku = pl.get("sku", "")
            style = pl.get("style", "")
            category = pl.get("category", "")
            sclass = pl.get("class", "")
            raw_desc = pl.get("description")
            description = str(raw_desc) if raw_desc is not None else ""
            score = getattr(r, "score", None)

            with cols[idx]:
                if img_url:
                    st.image(img_url, caption=name, use_container_width=True)
                st.markdown(f"**{name}**")
                snippet = description[:100] + (
                    "..." if len(description) > 100 else ""
                )
                st.caption(snippet)
                st.markdown(
                    f"<span style='color:#0074D9; background:#e3f2fd; "
                    f"padding:1px 8px; border-radius:8px;font-size:0.9em'>{style}</span> "
                    f"<span style='color:#388E3C; background:#f1f8e9; "
                    f"padding:1px 8px; border-radius:8px;font-size:0.9em'>{category}</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"SKU: `{sku}`  |  Class: {sclass}")
                if score is not None:
                    st.markdown(f"*Relevance: {score:.3f}*")
                with st.expander("View all details"):
                    for k, v in pl.items():
                        st.write(f"**{k}**: {v}")
                st.write("---")

    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV", csv, "results.csv", mime="text/csv"
    )

# ── Analytics ──
def analytics_dashboard():
    """Display basic analytics for the product catalog."""
    st.header("Analytics Dashboard")
    st.markdown("Overview of available products and pricing.")

    total_products = len(art_df)
    st.metric("Total products", total_products)

    if "ecom_price" in art_df.columns:
        prices = pd.to_numeric(art_df["ecom_price"], errors="coerce").dropna()
        if not prices.empty:
            st.metric("Average price", f"${prices.mean():.2f}")
            bins = pd.cut(prices, bins=10)
            counts = bins.value_counts().sort_index()
            st.subheader("Price distribution")
            st.bar_chart(counts)

    st.subheader("Top Categories")
    top_cats = art_df["category"].value_counts().head(10)
    st.bar_chart(top_cats)

# ── Main Interface ──
def search_interface():
    # ── Sidebar ──
    with st.sidebar:
        st.image("company_logo.png", width=140)
        st.title("Filters")

        # Search mode (Image/Text/Hybrid) unchanged…
        st.markdown("**Search mode**")
        search_mode = st.radio(
            "Choose", ["Image", "Text", "Hybrid"], horizontal=True,
            key="search_mode"
        )
        st.markdown("---")

        # Color toggle + picker
        use_color = st.checkbox("Filter by color", value=False, key="filter_enable")
        if use_color:
            picked_color = st.color_picker("Pick a color to filter", key="pick_color")
            tol = st.slider("Color tolerance (0–441)", 0, 441, 50, key="color_tol")
        else:
            picked_color = None
            tol = None
        st.markdown("---")

        # All your other facet multiselects…
        filter_selections = {}
        for conf in filter_columns_config:
            if conf["col"] == "dominant_color_hex":
                continue
            filter_selections[conf["col"]] = st.multiselect(
                conf["label"],
                filter_options[conf["col"]],
                default=[],
                key=f"filter_{conf['col']}"
            )

        top_k = st.slider("Number of results", 1, 100, value=5)
        st.markdown("---")

        if st.button("Reset all filters"):
            st.session_state.clear()
            st.rerun()

    # ── Build facet filter dict ──
    filters = {}

    # only apply color filtering when use_color is True
    if use_color and picked_color:
        target = hex_to_rgb(picked_color)
        tmp = art_df.copy()
        tmp["_dist"] = tmp["dominant_color_hex"].map(
            lambda h: math.dist(hex_to_rgb(h), target)
        )
        close_hexes = tmp[tmp["_dist"] <= tol]["dominant_color_hex"].unique().tolist()
        if close_hexes:
            filters["dominant_color_hex"] = close_hexes

    # the rest of your facets
    for conf in filter_columns_config:
        col = conf["col"]
        if col == "dominant_color_hex":
            continue
        sel = [v for v in filter_selections[col] if v and v != "Any"]
        if sel:
            filters[col] = sel

    # ── “Tabs” via radio ──
    st.title("🎨 Classy Reverse Image/Text Search")
    show_active_filters(filters)

    tabs = ["Image & Text Search", "Search by SKU", "Analytics"]
    default_tab = st.session_state.get("active_tab", tabs[0])

    tab = st.radio(
        "Mode",
        options=tabs,
        index=tabs.index(default_tab) if default_tab in tabs else 0,
        key="active_tab",
        horizontal=True
    )

    # Tab 1: Vector/Text/Hybrid
    if tab == "Image & Text Search":
        if search_mode == "Image":
            uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="img_uploader")
            if uploaded:
                img = Image.open(uploaded).convert("RGB")
                st.image(img, caption="Uploaded Image", width=220)
                if st.button("Search", key="img_search"):
                    with st.spinner("Searching…"):
                        emb = get_image_embedding(img)
                        results = vector_search(emb, "image", top_k, filters)
                    display_results(results)

        elif search_mode == "Text":
            text_query = st.text_input("Enter a descriptive query", key="txt_query")
            if text_query and st.button("Search", key="txt_search"):
                with st.spinner("Searching…"):
                    emb = get_text_embedding(text_query)
                    results = vector_search(emb, "text", top_k, filters)
                display_results(results)

        else:  # Hybrid
            up_img = st.file_uploader("Upload image for hybrid search", type=["jpg","jpeg","png"], key="hyb_img")
            text_query = st.text_input("Enter a descriptive query", key="hyb_txt")
            if (up_img or text_query) and st.button("Search (Hybrid)", key="hyb_search"):
                with st.spinner("Searching…"):
                    vectors = {}
                    if up_img:
                        img = Image.open(up_img).convert("RGB")
                        vectors["image"] = get_image_embedding(img)
                    if text_query:
                        vectors["text"]  = get_text_embedding(text_query)
                    results = hybrid_search(vectors, top_k, filters)
                display_results(results)

    # Tab 2: SKU Lookup & Find Similar
    elif tab == "Search by SKU":
        st.subheader("Find product by SKU")
        sku_query = st.text_input("Enter SKU", key="sku_query")
        # Sanitize SKU: capitalize all letters
        sanitized_sku_query = sku_query.upper() if sku_query else ""
        if st.button("Search SKU", key="sku_search"):
            df_hit = art_df[art_df["sku"] == sanitized_sku_query]
            st.session_state["sku_hit"] = df_hit

        if "sku_hit" in st.session_state:
            df_hit = st.session_state["sku_hit"]
            if df_hit.empty:
                st.warning(f"No product found with SKU `{sanitized_sku_query}`.")
            else:
                # display it…
                points = []
                for _, row in df_hit.iterrows():
                    class Point: pass

                    p = Point()
                    p.payload = row.dropna().to_dict()
                    p.score = None
                    points.append(p)
                display_results(points)

                # Find Similar button
                main_img = df_hit.iloc[0]["main_image_file"]
                if main_img and st.button("Find Similar", key="find_similar"):
                    img = Image.open(BytesIO(requests.get(main_img).content)).convert("RGB")
                    emb = get_image_embedding(img)
                    sim = vector_search(emb, "image", top_k, {})  # no color filter
                    display_results(sim)

    else:  # Analytics tab
        analytics_dashboard()

if __name__ == "__main__":
    search_interface()
