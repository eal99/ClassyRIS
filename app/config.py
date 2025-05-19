# Basic Streamlit configuration
import os

PAGE_TITLE = "Classy RIS/Text Search"
PAGE_LAYOUT = "wide"
PAGE_SIDEBAR_STATE = "expanded"

# Allow the collection name to be configured via an environment variable.  This
# is useful when deploying to platforms like Heroku.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Classy_Art")

