# ui.py
import streamlit as st
from app.embedding import get_image_embedding, get_text_embedding
from app.qdrant_utils import vector_search, hybrid_search
from PIL import Image

def search_interface():
    st.title("Classy Reverse Image/Text Search")

    search_mode = st.radio("Search by", ["Image", "Text", "Hybrid"])
    # Define your filter options here (ideally, load these from your product catalog or a helper)
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

    # Add UI multiselects
    style_filter = st.multiselect("Style", style_options)
    category_filter = st.multiselect("Category", category_options)
    class_filter = st.multiselect("Class", class_options)
    occasion_filter = st.multiselect("Occasion", occasion_options)
    orientation_filter = st.multiselect("Orientation", orientation_options)

    # Build filters dictionary dynamically, sending only applied/selected fields
    filters = {}
    if style_filter:
        filters["Style"] = style_filter
    if category_filter:
        filters["Category"] = category_filter
    if class_filter:
        filters["Class"] = class_filter
    if occasion_filter:
        filters["Occasion"] = occasion_filter
    if orientation_filter:
        filters["Orientation"] = orientation_filter

    top_k = st.slider("Number of results", 1, 20, value=5)

    if search_mode == "Image":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img)
            emb = get_image_embedding(img)
            if st.button("Search"):
                results = vector_search(emb, "image", top_k, filters)
                display_results(results)
    elif search_mode == "Text":
        query = st.text_input("Enter text query")
        if query and st.button("Search"):
            emb = get_text_embedding(query)
            results = vector_search(emb, "text", top_k, filters)
            display_results(results)
    elif search_mode == "Hybrid":
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        query = st.text_input("Enter text query (optional)")
        if uploaded or query:
            image_emb = get_image_embedding(Image.open(uploaded)) if uploaded else None
            text_emb = get_text_embedding(query) if query else None
            vectors = {}
            if image_emb: vectors["image"] = image_emb
            if text_emb: vectors["text"] = text_emb
            if vectors and st.button("Hybrid Search"):
                results = hybrid_search(vectors, top_k, filters)
                display_results(results)

def display_results(results):
    if not results:
        st.warning("No results found.")
        return

    num_cols = 5  # Number of result columns per row (can tune)

    for i in range(0, len(results), num_cols):
        cols = st.columns(num_cols)
        for idx, r in enumerate(results[i:i+num_cols]):
            pl = r.payload or {}
            img_url = pl.get("Cloudinary_1") or pl.get("Main Image File")
            name = pl.get("Product Name", "N/A")
            sku = pl.get("SKU", "")
            style = pl.get("Style", "")
            category = pl.get("Category", "")
            sclass = pl.get("Class", "")
            price = pl.get("MAP Price", "")
            description = pl.get("Description", "")
            score = r.score if hasattr(r, "score") else None

            with cols[idx]:
                if img_url:
                    st.image(img_url, use_container_width=True)
                st.markdown(f"**{name}**")
                st.markdown(f"SKU: `{sku}`")
                st.markdown(f"Style: <span style='color:#2196F3;'>{style}</span>", unsafe_allow_html=True)
                st.markdown(f"Category: <span style='color:#43A047;'>{category}</span>", unsafe_allow_html=True)
                st.markdown(f"Class: <span style='color:#F9A825;'>{sclass}</span>", unsafe_allow_html=True)
                if price:
                    st.markdown(f"Price: ${price}")
                if score:
                    st.markdown(f"*Relevance Score: {score:.3f}*")
                if description:
                    st.caption(description[:90] + ("..." if len(description) > 90 else ""))
                st.write("---")