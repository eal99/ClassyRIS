import math
import uuid
import requests
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd

from app.embedding import get_image_embedding, get_text_embedding
from app.qdrant_utils import vector_search, hybrid_search
from app.data_utils import art_df, filter_columns_config, filter_options

# small CSS tweaks for tighter layout
st.markdown(
    """
    <style>
    .stButton button {margin: 0.25rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGE_SIZE = 10


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def show_active_filters(filters: dict) -> None:
    if not filters:
        return
    chips = [f"`{field}: {', '.join(map(str, vals))}`" for field, vals in filters.items()]
    st.markdown("**Active filters:** " + " &nbsp; ".join(chips))


def display_results(results: list | None) -> None:
    if results is not None:
        st.session_state.search_results = results
        st.session_state.page = 0
    results = st.session_state.get("search_results", [])
    if not results:
        st.warning("No results found. Try broadening your query or removing some filters.")
        return

    page = st.session_state.get("page", 0)
    start = page * PAGE_SIZE
    end = start + PAGE_SIZE
    subset = results[start:end]

    df_results = pd.DataFrame([
        {**(r.payload or {}), "score": getattr(r, "score", None)} for r in results
    ])

    num_cols = 5
    for i in range(0, len(subset), num_cols):
        cols = st.columns(num_cols)
        for idx, r in enumerate(subset[i:i+num_cols]):
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
                snippet = description[:100] + ("..." if len(description) > 100 else "")
                st.caption(snippet)
                st.markdown(
                    f"<span style='color:#0074D9; background:#e3f2fd; padding:1px 8px; border-radius:8px;font-size:0.9em'>{style}</span> "
                    f"<span style='color:#388E3C; background:#f1f8e9; padding:1px 8px; border-radius:8px;font-size:0.9em'>{category}</span>",
                    unsafe_allow_html=True,
                )
                st.write(f"SKU: `{sku}`  |  Class: {sclass}")
                if score is not None:
                    st.markdown(f"*Relevance: {score:.3f}*")
                with st.expander("View all details"):
                    for k, v in pl.items():
                        st.write(f"**{k}**: {v}")
                st.write("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Prev", disabled=page == 0):
            st.session_state.page = max(page - 1, 0)
            st.experimental_rerun()
    with col2:
        total = len(results)
        st.write(f"Page {page+1} of {math.ceil(total/PAGE_SIZE)}")
    with col3:
        if st.button("Next", disabled=end >= len(results)):
            st.session_state.page = page + 1
            st.experimental_rerun()

    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        csv,
        "results.csv",
        mime="text/csv",
        key=str(uuid.uuid4()),
    )


def render() -> None:
    st.set_page_config(page_title="Classy Search", layout="wide", page_icon="🎨")

    with st.sidebar:
        st.image("company_logo.png", width=500)
        st.title("Filters")

        st.markdown("**Search mode**")
        search_mode = st.radio("Choose", ["Image", "Text", "Hybrid"], horizontal=True, key="search_mode")
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

    filters: dict[str, list[str]] = {}
    if use_color and picked_color:
        target = hex_to_rgb(picked_color)
        tmp = art_df.copy()
        tmp["_dist"] = tmp["dominant_color_hex"].map(lambda h: math.dist(hex_to_rgb(h), target))
        close_hexes = tmp[tmp["_dist"] <= tol]["dominant_color_hex"].unique().tolist()
        if close_hexes:
            filters["dominant_color_hex"] = close_hexes

    for conf in filter_columns_config:
        col = conf["col"]
        if col == "dominant_color_hex":
            continue
        sel = [v for v in filter_selections[col] if v and v != "Any"]
        if sel:
            filters[col] = sel

    st.title("🎨 Classy Reverse Image/Text Search")
    show_active_filters(filters)

    tab_names = ["Image & Text Search", "Search by SKU"]
    tabs = st.tabs(tab_names)
    tab_map = dict(zip(tab_names, tabs))

    with tab_map["Image & Text Search"]:
        if search_mode == "Image":
            uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img_uploader")
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
        else:
            up_img = st.file_uploader("Upload image for hybrid search", type=["jpg", "jpeg", "png"], key="hyb_img")
            text_query = st.text_input("Enter a descriptive query", key="hyb_txt")
            if (up_img or text_query) and st.button("Search (Hybrid)", key="hyb_search"):
                with st.spinner("Searching…"):
                    vectors = {}
                    if up_img:
                        img = Image.open(up_img).convert("RGB")
                        vectors["image"] = get_image_embedding(img)
                    if text_query:
                        vectors["text"] = get_text_embedding(text_query)
                    results = hybrid_search(vectors, top_k, filters)
                display_results(results)

    with tab_map["Search by SKU"]:
        st.subheader("Find product by SKU")
        sku_query = st.text_input("Enter SKU", key="sku_query")
        sanitized_sku_query = sku_query.upper() if sku_query else ""
        if st.button("Search SKU", key="sku_search"):
            df_hit = art_df[art_df["sku"] == sanitized_sku_query]
            st.session_state["sku_hit"] = df_hit

        if "sku_hit" in st.session_state:
            df_hit = st.session_state["sku_hit"]
            if df_hit.empty:
                st.warning(f"No product found with SKU `{sanitized_sku_query}`.")
            else:
                points = []
                for _, row in df_hit.iterrows():
                    class Point:
                        pass
                    p = Point()
                    p.payload = row.dropna().to_dict()
                    p.score = None
                    points.append(p)
                display_results(points)

                main_img = df_hit.iloc[0]["main_image_file"]
                if main_img and st.button("Find Similar", key="find_similar"):
                    img = Image.open(BytesIO(requests.get(main_img).content)).convert("RGB")
                    emb = get_image_embedding(img)
                    sim = vector_search(emb, "image", top_k, {})
                    display_results(sim)

        # end search by SKU



