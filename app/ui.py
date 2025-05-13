# ui.py

import streamlit as st
import pandas as pd
import numpy as np
import math
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
    if not results:
        st.warning("No results found. Try broadening your query or removing some filters.")
        return

    num_cols = 5
    for i in range(0, len(results), num_cols):
        cols = st.columns(num_cols)
        for idx, r in enumerate(results[i:i + num_cols]):
            pl = r.payload or {}
            img_url     = pl.get("main_image_file")
            name        = pl.get("product_name", "N/A")
            sku         = pl.get("sku", "")
            style       = pl.get("style", "")
            category    = pl.get("category", "")
            sclass      = pl.get("class", "")
            raw_desc    = pl.get("description")
            description = str(raw_desc) if raw_desc is not None else ""
            score       = getattr(r, "score", None)

            with cols[idx]:
                if img_url:
                    st.image(img_url, caption=name, use_container_width=True)
                st.markdown(f"**{name}**")
                snippet = description[:100] + ("..." if len(description) > 100 else "")
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


# ── Main Interface ──
def search_interface():
    # --- Sidebar ---
    with st.sidebar:
        st.image("company_logo.png", width=140)
        st.title("Filters")

        st.markdown("**Search mode**")
        search_mode = st.radio("Choose", ["Image", "Text", "Hybrid"], horizontal=True)
        st.markdown("---")

        use_color = st.checkbox("Filter by color", value=False, key="filter_enable")
        if use_color:
            picked_color = st.color_picker("Pick a color to filter", key="pick_color")
            tol = st.slider("Color tolerance (0–441)", 0, 441, 50, key="color_tol")
        else:
            picked_color = None
            tol = None
        st.markdown("---")

        filter_selections = {}
        for conf in filter_columns_config:
            col = conf["col"]
            if col == "dominant_color_hex":
                continue
            filter_selections[col] = st.multiselect(
                conf["label"],
                filter_options[col],
                default=[],
                key=f"filter_{col}"
            )

        top_k = st.slider("Number of results", 1, 20, value=5)
        st.markdown("---")

        if st.button("Reset all filters"):
            st.session_state.clear()
            st.rerun()

        with st.expander("ℹ️ How to use"):
            st.markdown("""
                **Mode:**
                - *Image*: Find similar art by uploading an image.
                - *Text*: Enter text/keywords to search.
                - *Hybrid*: Combine image and description for best relevance.
                """)
            st.markdown("**Tip:** You can select multiple values per facet.")

    # --- Build filter dict ---
    filters = {}

    if use_color and picked_color:
        target = hex_to_rgb(picked_color)
        tmp = art_df.copy()
        tmp["_dist"] = tmp["dominant_color_hex"].map(
            lambda h: math.dist(hex_to_rgb(h), target)
        )
        close_hexes = tmp[tmp["_dist"] <= tol]["dominant_color_hex"].unique().tolist()
        if close_hexes:
            filters["dominant_color_hex"] = close_hexes

    for conf in filter_columns_config:
        col = conf["col"]
        if col == "dominant_color_hex":
            continue
        sel = [v for v in filter_selections[col] if v != "Any"]
        if sel:
            filters[col] = sel

    # --- Tabs and results pane ---
    st.title("🎨 Classy Reverse Image/Text Search")
    # show_active_filters(filters)

    tab1, tab2 = st.tabs(["Vector/Text Search", "Search by SKU"])

    # Tab 1: Vector/Text/Hybrid
    with tab1:
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
                        vectors["image"] = get_image_embedding(Image.open(up_img).convert("RGB"))
                    if text_query:
                        vectors["text"]  = get_text_embedding(text_query)
                    results = hybrid_search(vectors, top_k, filters)
                display_results(results)

    # Tab 2: SKU Lookup (local DF)
    with tab2:
        st.subheader("Find product by SKU")
        sku_query = st.text_input("Enter SKU", key="sku_query")
        if sku_query and st.button("Search SKU", key="sku_search"):
            df_hit = art_df[art_df["sku"] == sku_query]
            if df_hit.empty:
                st.warning(f"No product found with SKU `{sku_query}`.")
            else:
                # Wrap into a minimal “point” object
                results = []
                for _, row in df_hit.iterrows():
                    class Point: pass
                    p = Point()
                    p.payload = row.dropna().to_dict()
                    p.score   = None
                    results.append(p)
                display_results(results)


if __name__ == "__main__":
    search_interface()