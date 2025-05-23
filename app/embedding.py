# embedding.py
import os
from PIL import Image
from openai import OpenAI
import streamlit as st
from app.clip_utils import generate_image_embedding
import tempfile

# Only initialize the OpenAI client if an API key is available.  Importing this
# module shouldn't fail just because the environment variable is missing.
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None

def get_image_embedding(image: Image.Image) -> list[float]:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name)
        return generate_image_embedding(tmp.name)

def get_text_embedding(text: str) -> list[float]:
    model_name = "text-embedding-3-large"
    if client is None:
        st.error("OPENAI_API_KEY not set; text search is unavailable.")
        return []
    try:
        response = client.embeddings.create(input=[text], model=model_name)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding from OpenAI: {e}")
        return []
