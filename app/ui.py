# ui.py
import streamlit as st
from app.embedding import get_image_embedding, get_text_embedding
from app.qdrant_utils import vector_search, hybrid_search
from PIL import Image

st.set_page_config(
    page_title="Classy RIS/Text Search",
    layout="wide",
    page_icon="🎨"
)

import os

style_options = ["Any", "Contemporary", "Photography", "Illustration", "Transitional",
                  "Traditional", "Pop Art", "Vintage", "Expressionism", "Industrial",
                  "Art Deco", "Abstract", "Fashion Illustration", "Art Nouveau", "Figurative",
                  "Watercolor", "Impressionism", "Glam", "Renaissance", "Mid-Century",
                  "Rustic", "Mixed Media", "Street Art", "Motivational", "Metal",
                  "Typography and Symbols", "Modern", "Cubism", "Illustrative", "Typography",
                  "Water Color", "Photograph", "Folk Art", "MIX", "Realism", "Wildlife Art",
                  "Farmhouse", "Botanical"]
category_options = ["Any", "Botanical", "Decorative Mirrors", "Entertainment", "Abstract",
                 "Black & White", "Coastal", "Typography and Symbols", "Wildlife", "Cityscape",
                 "Youth/Kids", "Americana", "Country/Cottage", "Portrait", "Farmhouse",
                 "Travel", "Impressionism", "Scenic", "Photography", "World Culture", "Decor",
                 "Animals", "Motivational", "Decorative Art", "Shapes", "Comics", "Home Décor",
                 "Tropical", "Urban", "Nature", "Nautical", "Contemporary", "Figurative",
                 "Religion & Spirituality", "Transportation", "Wall Mirror", "Floral",
                 "Tuscan", "Floral Art", "Places", "Wildlife Art", "Hunting", "Cabin/Lodge",
                 "Sepia", "Fashion", "Art Deco", "Industrial", "Traditional", "Architecture",
                 "Mixed Media", "Framed Art", "Landscape", "Cityscapes", "Flower Photography",
                 "Vintage", "Home Decor", "Wall Art", "Acrylic", "Artistic", "Illustrative",
                 "Water Color", "Still Life", "Maps", "Watercolor", "Ethnic", "People",
                 "Wall Decor", "Portraiture", "Animal Art", "Modern", "Sports", "Home & Hearth",
                 "Advertisements", "Framed Print", "Decorative Mirror", "Food & Beverage",
                 "Cottage", "Sports & Teams", "Oil", "Botanical Art", "Patriotic", "Religious",
                 "Modern Art", "Skyline & City Scape", "Fine Art", "Classy Art", "Decorative",
                 "Blueprint", "Nature Art", "Military", "Wine & Spirits", "Sport", "Kitchen",
                 "Sports & Outdoor", "Landscapes", "Motorcycles", "Sports & Outdoors",
                 "Urban/Cityscape", "Cuisine", "Music", "Nostalgic", "Luxury", "Pop Art",
                 "Illustration", "Performing Arts", "Bar Decor", "Historical", "Bohemian",
                 "Classic", "Kitchen/Dining", "Dance", "Family", "Nature & Landscape"]
class_options = ["Any", "22x26 Framed Print", "34x40 Mirror Frame Print", "34x40 Framed Print",
              "18x42 Mirror Frame Print", "Tempered Glass", "22x26 Mirror Frame Print",
              "28x34 Framed Print", "Mixed Media", "18x42 Framed Print"]

occasion_options = ["Any", 'Spring', 'General decoration', 'General celebration', 'Summer',
   'Fall', 'General Decoration', 'Autumn', 'Winter', 'General',
   'Tropical', 'General Celebration', "Valentine's Day", 'Birthday',
   'Travel', 'Fashion Week', 'Anniversary', 'Motivational',
   'Independence Day', 'Farm', 'Christmas', 'Family Gathering',
   'Rainy Day', 'Spring, General decoration', 'Family Celebration',
   'Wedding', 'Celebration', 'Fashion', "Children's Decor", 'General decor', 'Easter', "Mother's Day", 'Seasonal', 'Wildlife',
   'Religious', 'Halloween', 'Veterans Day', 'Culture',
   'Wildlife Conservation Day', 'Pride']

orientation_options = ["Any", 'Vertical', 'Horizontal', 'Square', 'Round']


def search_interface():
    # --- Sidebar ---
    with st.sidebar:
        st.image("company_logo.png", width=140)
        st.title("Filters")
        st.markdown("**Search mode**")
        search_mode = st.radio("Choose", ["Image", "Text", "Hybrid"], horizontal=True)

        st.markdown("---")
        st.markdown("**Facet Filters**:")
        style_filter = st.multiselect("Style", style_options)
        category_filter = st.multiselect("Category", category_options)
        class_filter = st.multiselect("Class", class_options)
        occasion_filter = st.multiselect("Occasion", occasion_options)
        orientation_filter = st.multiselect("Orientation", orientation_options)
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
            \n**Tip:** You can select multiple filter values for each facet to narrow results.
            """)

    # --- Build filters dict ---
    filters = {}
    if style_filter: filters["Style"] = style_filter
    if category_filter: filters["Category"] = category_filter
    if class_filter: filters["Class"] = class_filter
    if occasion_filter: filters["Occasion"] = occasion_filter
    if orientation_filter: filters["Orientation"] = orientation_filter

    # --- Show main search interface ---
    st.title("🎨 Classy Reverse Image/Text Search")
    show_active_filters(filters)

    # --------- Main search modes ----------
    if search_mode == "Image":
        uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"], key="img_uploader")
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=False, width=220)
            if st.button("Search", key="img_search"):
                with st.spinner("Searching, please wait..."):
                    emb = get_image_embedding(img)
                    results = vector_search(emb, "image", top_k, filters)
                display_results(results)
    elif search_mode == "Text":
        text_query = st.text_input("Enter a descriptive query", key="txt_query")
        if text_query and st.button("Search", key="txt_search"):
            with st.spinner("Searching, please wait..."):
                emb = get_text_embedding(text_query)
                results = vector_search(emb, "text", top_k, filters)
            display_results(results)
    elif search_mode == "Hybrid":
        up_img = st.file_uploader("Upload an image for hybrid search", type=["jpg", "jpeg", "png"], key="hyb_img")
        text_query = st.text_input("Enter a descriptive query", key="hyb_txt")
        if (up_img or text_query) and st.button("Search (Hybrid)", key="hyb_search"):
            with st.spinner("Searching, please wait..."):
                img_emb = get_image_embedding(Image.open(up_img)) if up_img else None
                txt_emb = get_text_embedding(text_query) if text_query else None
                vectors = {}
                if img_emb: vectors["image"] = img_emb
                if txt_emb: vectors["text"] = txt_emb
                if vectors:
                    results = hybrid_search(vectors, top_k, filters)
                    display_results(results)
                else:
                    st.warning("Supply at least an image or a text query.")


def show_active_filters(filters):
    # Display currently selected filters as chips
    if filters:
        chips = []
        for field, vals in filters.items():
            chips.append(f"`{field}: {', '.join(str(v) for v in vals)}`")
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
            img_url = pl.get("Cloudinary_1") or pl.get("Main Image File")
            name = pl.get("Product Name", "N/A")
            sku = pl.get("SKU", "")
            style = pl.get("Style", "")
            category = pl.get("Category", "")
            sclass = pl.get("Class", "")
            description = pl.get("Description", "")
            score = getattr(r, 'score', None)
            with cols[idx]:
                with st.container():
                    if img_url:
                        st.image(img_url, caption=name, use_container_width=True)
                    st.markdown(f"**{name}**", unsafe_allow_html=True)
                    if description:
                        st.caption(description[:100] + ("..." if len(description) > 100 else ""))
                    else:
                        st.caption("No description available.")
                    st.markdown(
                        f"<span style='color: #0074D9; background:#e3f2fd; padding:1px 8px; "
                        f"border-radius:8px;font-size:0.9em'>{style}</span> "
                        f"<span style='color: #388E3C; background:#f1f8e9; padding:1px 8px; "
                        f"border-radius:8px;font-size:0.9em'>{category}</span> ",
                        unsafe_allow_html=True,
                    )
                    st.write(f"SKU: `{sku}`  |  Class: {sclass}")
                    if score: st.markdown(f"*Relevance: {score:.3f}*")
                    with st.expander("View all details"):
                        for k, v in pl.items():
                            st.write(f"**{k}**: {v}")
                    st.write("---")