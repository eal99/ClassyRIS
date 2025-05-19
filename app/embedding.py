# embedding.py
import os
from PIL import Image
from openai import OpenAI
import streamlit as st
from app.clip_utils import generate_image_embedding
import tempfile

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_image_embedding(image: Image.Image) -> list[float]:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name)
        return generate_image_embedding(tmp.name)

def get_text_embedding(text: str) -> list[float]:
    model_name = "text-embedding-3-large"
    try:
        response = client.embeddings.create(input=[text], model=model_name)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding from OpenAI: {e}")
        return []
