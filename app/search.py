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

# --- SET PAGE CONFIG FIRST ---

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


def hex_to_rgb(h: str) -> tuple[int, ...]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def show_active_filters(filters: dict) -> None:
    if not filters:
        return
    chips = [f"`{field}: {', '.join(map(str, vals))}`" for field, vals in filters.items()]
    st.markdown("**Active filters:** " + " &nbsp; ".join(chips))


def display_results(results: list | None, key_prefix: str = "") -> None:
    if results is not None:
        st.session_state.search_results = results
        st.session_state.page = 0
        st.session_state.results_prefix = key_prefix
    results = st.session_state.get("search_results", [])
    key_prefix = st.session_state.get("results_prefix", key_prefix)
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
        if st.button("Prev", disabled=page == 0, key=f"{key_prefix}_prev"):
            st.session_state.page = max(page - 1, 0)
            st.rerun()
    with col2:
        total = len(results)
        st.write(f"Page {page+1} of {math.ceil(total/PAGE_SIZE)}")
    with col3:
        if st.button("Next", disabled=end >= len(results), key=f"{key_prefix}_next"):
            st.session_state.page = page + 1
            st.rerun()

    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        csv,
        "results.csv",
        mime="text/csv",
        key=str(uuid.uuid4()),
    )


# ──────────────────────────────  Helpers  ──────────────────────────────
def _build_sidebar() -> tuple[dict[str, list[str]], int, str]:
    """Render the sidebar and return (filters, top_k, search_mode)."""
    with st.sidebar:
        st.image("company_logo.png", width=500)
        st.title("Filters")

        search_mode = st.radio("Search mode", ["Image", "Text", "Hybrid"], horizontal=True)
        st.markdown("---")

        # --- Colour filter ----------------------------------------------------
        filters: dict[str, list[str]] = {}
        if st.checkbox("Filter by Color"):
            picked_colour = st.color_picker("Pick a Color")
            tolerance = st.slider("Colour tolerance (0–441)", 0, 441, 50)
            target_rgb = hex_to_rgb(picked_colour)
            tmp = art_df.copy()
            tmp["_dist"] = tmp["dominant_color_hex"].map(lambda h: math.dist(hex_to_rgb(h), target_rgb))
            close_hexes = tmp[tmp["_dist"] <= tolerance]["dominant_color_hex"].unique().tolist()
            if close_hexes:
                filters["dominant_color_hex"] = close_hexes
        st.markdown("---")

        # --- Multiselect filters ---------------------------------------------
        for conf in filter_columns_config:
            col_name = conf["col"]
            if col_name == "dominant_color_hex":
                continue
            selection = st.multiselect(conf["label"], filter_options[col_name], default=[])
            if selection and "Any" not in selection:
                filters[col_name] = selection

        top_k = st.slider("Number of results", 1, 100, value=5)
        st.markdown("---")

        if st.button("Reset all filters"):
            st.session_state.clear()
            st.rerun()

    return filters, top_k, search_mode


def _image_text_tab(search_mode: str, top_k: int, filters: dict) -> bool:
    """Handle the ‘Image & Text Search’ tab. Returns True if new results drawn."""
    new_results_shown = False

    if search_mode == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded and st.button("🔍  Search"):
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded image", width=220)
            with st.spinner("Searching…"):
                emb = get_image_embedding(img)
                res = vector_search(emb, "image", top_k, filters)
            display_results(res, key_prefix="img_search")
            new_results_shown = True

    elif search_mode == "Text":
        query = st.text_input("Enter a descriptive query")
        if query and st.button("🔍  Search"):
            with st.spinner("Searching…"):
                emb = get_text_embedding(query)
                res = vector_search(emb, "text", top_k, filters)
            display_results(res, key_prefix="txt_search")
            new_results_shown = True

    else:  # Hybrid
        up_img = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
        query = st.text_input("Enter a descriptive query (optional)")
        if (up_img or query) and st.button("🔍  Search"):
            vectors = {}
            if up_img:
                img = Image.open(up_img).convert("RGB")
                vectors["image"] = get_image_embedding(img)
            if query:
                vectors["text"] = get_text_embedding(query)
            with st.spinner("Searching…"):
                res = hybrid_search(vectors, top_k, filters)
            display_results(res, key_prefix="hyb_search")
            new_results_shown = True

    return new_results_shown


def _sku_tab(top_k: int) -> bool:
    """Handle the ‘Search by SKU’ tab. Returns True if new results drawn."""
    new_results_shown = False
    st.subheader("Find product by SKU")

    sku_query = st.text_input("Enter SKU").upper()
    if st.button("🔍  Search SKU"):
        hit = art_df[art_df["sku"] == sku_query]
        st.session_state["sku_hit"] = hit

    if "sku_hit" in st.session_state:
        hit = st.session_state["sku_hit"]
        if hit.empty:
            st.warning(f"No product found with SKU `{sku_query}`.")
        else:
            # Wrap DataFrame rows in dummy points expected by display_results()
            points = []
            for _, row in hit.iterrows():
                class Point:  # simple struct-like helper
                    pass
                p = Point()
                p.payload = row.dropna().to_dict()
                p.score = None
                points.append(p)
            display_results(points, key_prefix="sku_results")
            new_results_shown = True

            # Optional “find similar” feature
            main_img = hit.iloc[0]["main_image_file"]
            if main_img and st.button("Find similar items"):
                img = Image.open(BytesIO(requests.get(main_img).content)).convert("RGB")
                emb = get_image_embedding(img)
                similar = vector_search(emb, "image", top_k, {})
                display_results(similar, key_prefix="find_similar")
                new_results_shown = True

    return new_results_shown


# ───────────────────────────────  Main  ────────────────────────────────
def render() -> None:
    """Entry point for the Streamlit page."""
    # ----- sidebar & filters -------------------------------------------------
    filters, top_k, search_mode = _build_sidebar()

    # ----- header ------------------------------------------------------------
    st.title("🎨 Classy Reverse Image/Text Search")
    show_active_filters(filters)

    # ----- tabs --------------------------------------------------------------
    img_text_tab, sku_tab = st.tabs(["Image & Text Search", "Search by SKU"])

    results_shown = False
    with img_text_tab:
        results_shown |= _image_text_tab(search_mode, top_k, filters)

    with sku_tab:
        results_shown |= _sku_tab(top_k)

    # ----- fallback: redisplay previous results ------------------------------
    if not results_shown and st.session_state.get("search_results"):
        display_results(None)