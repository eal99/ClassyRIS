# embedding.py
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
import streamlit as st

# API key can be provided via the OPENAI_API_KEY environment variable which is
# the typical approach when deploying to Heroku.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_clip_model():
    if not hasattr(get_clip_model, "model"):
        # load once
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)

        # prefer CUDA, else CPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # move safely
        try:
            model = model.to(device)
        except NotImplementedError:
            device = torch.device("cpu")
            model = model.to(device)

        get_clip_model.model     = model
        get_clip_model.processor = processor
        get_clip_model.device    = device

    return get_clip_model.model, get_clip_model.processor, get_clip_model.device

def get_image_embedding(image: Image.Image) -> list[float]:
    model, processor, device = get_clip_model()
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().cpu().tolist()

def get_text_embedding(text: str) -> list[float]:
    model_name = "text-embedding-3-large"
    try:
        response = client.embeddings.create(input=[text], model=model_name)
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding from OpenAI: {e}")
        return []
