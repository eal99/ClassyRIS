import os

PAGE_TITLE = "Classy RIS/Text Search"
PAGE_LAYOUT = "wide"
PAGE_SIDEBAR_STATE = "expanded"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://8a5d6688-43c7-453b-9744-8c25e746fd04.us-east-1-0.aws.cloud.qdrant.io",
)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "classyliving_products_current")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
QDRANT_CLOUD_INFERENCE = _get_bool("QDRANT_CLOUD_INFERENCE", True)

QDRANT_TEXT_DENSE_VECTOR = os.getenv("QDRANT_TEXT_DENSE_VECTOR", "text_dense")
QDRANT_TEXT_SPARSE_VECTOR = os.getenv("QDRANT_TEXT_SPARSE_VECTOR", "text_sparse")
QDRANT_IMAGE_VECTOR = os.getenv("QDRANT_IMAGE_VECTOR", "image_dense")

QDRANT_TEXT_DENSE_MODEL = os.getenv(
    "QDRANT_TEXT_DENSE_MODEL", "openai/text-embedding-3-large"
)
QDRANT_TEXT_SPARSE_MODEL = os.getenv("QDRANT_TEXT_SPARSE_MODEL", "Qdrant/bm25")
QDRANT_IMAGE_MODEL = os.getenv(
    "QDRANT_IMAGE_MODEL", "qdrant/clip-vit-b-32-vision"
)

SEARCH_PREFETCH_LIMIT = int(os.getenv("SEARCH_PREFETCH_LIMIT", "50"))

SEARCH_PAYLOAD_FIELDS = [
    "sku",
    "parent_sku",
    "brand",
    "manufacturer",
    "status",
    "status_facet",
    "item_type",
    "item_type_facet",
    "family_collection",
    "short_description",
    "searchable_text_short",
    "romance_copy",
    "page_romance",
    "description",
    "style",
    "style_facets",
    "subject",
    "subject_facets",
    "orientation",
    "country_of_origin",
    "ship_type",
    "material",
    "material_list",
    "material_facets",
    "room_type_list",
    "room_type_facets",
    "color_category_list",
    "color_category_facets",
    "primary_image_url",
    "main_image_file",
    "image_url_image_file_1",
    "image_url_lifestyle_image_1",
    "image_url_diagram_image_1",
    "image_url_3_dimensional_image_1",
    "media",
    "width",
    "height",
    "depth",
    "weight",
    "wholesale_price",
    "map_price",
    "msrp",
]

SEARCH_FILTERS = [
    {
        "label": "Style",
        "payload_key": "style_facets",
        "source_columns": ["style"],
        "lowercase": True,
    },
    {
        "label": "Collection",
        "payload_key": "family_collection",
        "source_columns": ["family_collection", "collection_name"],
    },
    {
        "label": "Subject",
        "payload_key": "subject_facets",
        "source_columns": ["subject"],
        "lowercase": True,
    },
    {
        "label": "Country of Origin",
        "payload_key": "country_of_origin",
        "source_columns": ["country_of_origin"],
    },
]

DEFAULT_FILTERS = {
    "status_facet": ["active"],
}
