import math
import uuid
from urllib.parse import urljoin

import pandas as pd
import streamlit as st
from streamlit.elements import image as st_image

from app import config
from app.data_utils import art_df, get_image_url_by_sku
from app.qdrant_utils import get_by_sku, hybrid_search, search_image, search_text

st.markdown(
    """
    <style>
    .stButton button {margin: 0.25rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

PAGE_SIZE = 10


def _normalize_filter_value(value: object, lowercase: bool = False) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower() if lowercase else text


def _build_filter_options() -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for conf in config.SEARCH_FILTERS:
        values: set[str] = set()
        for column in conf["source_columns"]:
            if column not in art_df.columns:
                continue
            for raw_value in art_df[column].dropna().tolist():
                normalized = _normalize_filter_value(
                    raw_value,
                    lowercase=conf.get("lowercase", False),
                )
                if normalized:
                    values.add(normalized)
        if values:
            options[conf["payload_key"]] = ["Any"] + sorted(values)
    return options


FILTER_OPTIONS = _build_filter_options()
IMAGE_TO_URL = getattr(st_image, "image_to_url", None)


def _absolute_media_url(media_url: str) -> str | None:
    if not media_url:
        return None
    if media_url.startswith(("http://", "https://")):
        return media_url
    if config.PUBLIC_BASE_URL:
        return urljoin(f"{config.PUBLIC_BASE_URL}/", media_url.lstrip("/"))
    headers = getattr(st.context, "headers", None)
    if not headers:
        return None
    host = headers.get("Host") or headers.get("host")
    if not host or host.startswith(("localhost", "127.0.0.1", "0.0.0.0")):
        return None
    proto = headers.get("X-Forwarded-Proto") or headers.get("x-forwarded-proto") or "https"
    return f"{proto}://{host}{media_url}"


def _uploaded_file_query_url(uploaded_file) -> str | None:
    if IMAGE_TO_URL is None or uploaded_file is None:
        return None
    upload_bytes = uploaded_file.getvalue()
    if not upload_bytes:
        return None
    try:
        media_url = IMAGE_TO_URL(
            upload_bytes,
            width=config.QDRANT_QUERY_IMAGE_MAX_DIMENSION,
            clamp=False,
            channels="RGB",
            output_format="JPEG",
            image_id=f"qdrant-upload-{uuid.uuid4()}",
        )
    except Exception:
        return None
    return _absolute_media_url(media_url)


def _get_payload_value(payload: dict, *paths: str) -> object | None:
    for path in paths:
        value: object | None = payload
        for part in path.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, [], {}):
            return value
    return None


def _get_title(payload: dict) -> str:
    title = _get_payload_value(
        payload,
        "short_description",
        "product_name",
        "searchable_text_short",
        "description",
        "sku",
    )
    return str(title) if title is not None else "Untitled product"


def _get_image_url(payload: dict, sku: str) -> str | None:
    image_url = _get_payload_value(
        payload,
        "primary_image_url",
        "media.primary_image_url",
        "main_image_file",
        "image_url_image_file_1",
        "image_url_lifestyle_image_1",
        "image_url_diagram_image_1",
        "image_url_3_dimensional_image_1",
    )
    if image_url:
        return str(image_url)
    return get_image_url_by_sku(sku)


def _get_description_snippet(payload: dict) -> str:
    description = _get_payload_value(
        payload,
        "romance_copy",
        "page_romance",
        "description",
        "searchable_text_short",
    )
    if description is None:
        return ""
    text = str(description)
    return text[:140] + ("..." if len(text) > 140 else "")


def _render_badges(payload: dict) -> None:
    badges = [
        _get_payload_value(payload, "style"),
        _get_payload_value(payload, "item_type", "item_type_facet", "category"),
        _get_payload_value(payload, "brand", "manufacturer"),
    ]
    rendered = []
    colors = [
        ("#0074D9", "#e3f2fd"),
        ("#388E3C", "#f1f8e9"),
        ("#8e24aa", "#f3e5f5"),
    ]
    for badge, (fg, bg) in zip([b for b in badges if b], colors):
        rendered.append(
            f"<span style='color:{fg}; background:{bg}; padding:1px 8px; "
            f"border-radius:8px; font-size:0.9em'>{badge}</span>"
        )
    if rendered:
        st.markdown(" ".join(rendered), unsafe_allow_html=True)


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

    df_results = pd.DataFrame(
        [{**(result.payload or {}), "score": getattr(result, "score", None)} for result in results]
    )

    num_cols = 5
    for i in range(0, len(subset), num_cols):
        cols = st.columns(num_cols)
        for idx, result in enumerate(subset[i : i + num_cols]):
            payload = result.payload or {}
            sku = str(_get_payload_value(payload, "sku") or "")
            title = _get_title(payload)
            description = _get_description_snippet(payload)
            image_url = _get_image_url(payload, sku)
            score = getattr(result, "score", None)

            with cols[idx]:
                if image_url:
                    st.image(image_url, caption=title, width="stretch")
                st.markdown(f"**{title}**")
                if description:
                    st.caption(description)
                _render_badges(payload)
                collection = _get_payload_value(payload, "family_collection", "collection_name")
                if collection:
                    st.write(f"Collection: {collection}")
                st.write(f"SKU: `{sku}`")
                if score is not None:
                    st.markdown(f"*Relevance: {score:.3f}*")
                with st.expander("View details"):
                    st.json(payload)
                st.write("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Prev", disabled=page == 0, key=f"{key_prefix}_prev"):
            st.session_state.page = max(page - 1, 0)
            st.rerun()
    with col2:
        total = len(results)
        st.write(f"Page {page + 1} of {math.ceil(total / PAGE_SIZE)}")
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


def _build_sidebar() -> tuple[dict[str, list[str]], int, str]:
    with st.sidebar:
        st.image("company_logo.png", width=500)
        st.title("Filters")

        search_mode = st.radio("Search mode", ["Image", "Text", "Hybrid"], horizontal=True)
        st.markdown("---")

        filters: dict[str, list[str]] = {}
        only_active = st.checkbox("Only active products", value=True)
        if only_active:
            filters.update(config.DEFAULT_FILTERS)

        for conf in config.SEARCH_FILTERS:
            options = FILTER_OPTIONS.get(conf["payload_key"])
            if not options:
                continue
            selection = st.multiselect(conf["label"], options, default=[])
            if selection and "Any" not in selection:
                filters[conf["payload_key"]] = selection

        top_k = st.slider("Number of results", 1, 100, value=10)
        st.markdown("---")

        if st.button("Reset all filters"):
            st.session_state.clear()
            st.rerun()

    return filters, top_k, search_mode


def _run_search(search_fn, *args, **kwargs) -> list:
    try:
        return search_fn(*args, **kwargs)
    except Exception as exc:
        st.error(f"Qdrant search failed: {exc}")
        return []


def _image_text_tab(search_mode: str, top_k: int, filters: dict) -> bool:
    new_results_shown = False

    if search_mode == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded and st.button("Search", key="image_search_button"):
            st.image(uploaded, caption="Uploaded image", width=220)
            uploaded_url = _uploaded_file_query_url(uploaded)
            with st.spinner("Searching..."):
                results = _run_search(
                    search_image,
                    top_k=top_k,
                    payload_filters=filters,
                    image_url=uploaded_url,
                    image_bytes=None if uploaded_url else uploaded.getvalue(),
                )
            display_results(results, key_prefix="img_search")
            new_results_shown = True

    elif search_mode == "Text":
        query = st.text_input("Enter a descriptive query")
        if query and st.button("Search", key="text_search_button"):
            with st.spinner("Searching..."):
                results = _run_search(
                    search_text,
                    text=query,
                    top_k=top_k,
                    payload_filters=filters,
                )
            display_results(results, key_prefix="txt_search")
            new_results_shown = True

    else:
        uploaded = st.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
        query = st.text_input("Enter a descriptive query (optional)")
        if (uploaded or query) and st.button("Search", key="hybrid_search_button"):
            uploaded_url = _uploaded_file_query_url(uploaded) if uploaded else None
            with st.spinner("Searching..."):
                results = _run_search(
                    hybrid_search,
                    top_k=top_k,
                    payload_filters=filters,
                    text_query=query,
                    image_url=uploaded_url,
                    image_bytes=(uploaded.getvalue() if uploaded and not uploaded_url else None),
                )
            display_results(results, key_prefix="hyb_search")
            new_results_shown = True

    return new_results_shown


def _sku_tab(top_k: int) -> bool:
    new_results_shown = False
    st.subheader("Find product by SKU")

    sku_query = st.text_input("Enter SKU").strip()
    show_raw = st.checkbox("Show raw Qdrant payload")
    if st.button("Search SKU"):
        st.session_state["sku_hit"] = _run_search(get_by_sku, sku_query)

    if "sku_hit" in st.session_state:
        hits = st.session_state["sku_hit"]
        if not hits:
            st.warning(f"No product found with SKU `{sku_query}`.")
        else:
            if show_raw:
                st.subheader("Raw Qdrant response (SKU)")
                st.json(
                    [
                        {
                            "id": hit.id,
                            "score": getattr(hit, "score", None),
                            "payload": hit.payload,
                        }
                        for hit in hits
                    ]
                )
            display_results(hits, key_prefix="sku_results")
            new_results_shown = True

            image_url = _get_image_url(hits[0].payload or {}, sku_query)
            if image_url and st.button("Find similar items"):
                with st.spinner("Searching for similar items..."):
                    similar = _run_search(
                        search_image,
                        top_k=top_k,
                        payload_filters={},
                        image_url=image_url,
                    )
                display_results(similar, key_prefix="find_similar")
                new_results_shown = True

    return new_results_shown


def render() -> None:
    filters, top_k, search_mode = _build_sidebar()

    st.title("Classy Reverse Image/Text Search")
    show_active_filters(filters)

    tab1, tab2 = st.tabs(["Image & Text Search", "Search by SKU"])

    results_shown = False
    with tab1:
        results_shown = _image_text_tab(search_mode, top_k, filters)
    with tab2:
        if _sku_tab(top_k):
            results_shown = True

    if not results_shown and st.session_state.get("search_results"):
        display_results(None)
