"""Utility functions for future OpenAI-powered features."""

from openai import OpenAI
import os
import streamlit as st

# Initialize only if API key is present so importing doesn't fail in
# environments without OpenAI credentials.
_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None


def summarize_description(description: str) -> str:
    """Return a short summary of a product description using OpenAI."""
    if not description:
        return ""
    if client is None:
        st.warning("OPENAI_API_KEY not set; summarization disabled.")
        return ""
    try:
        prompt = f"Summarize the following product description in one sentence:\n{description}"
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"OpenAI summarization failed: {e}")
        return ""


def generate_tags(description: str) -> list[str]:
    """Generate simple comma-separated tags from a description."""
    if not description:
        return []
    if client is None:
        st.warning("OPENAI_API_KEY not set; tag generation disabled.")
        return []
    try:
        prompt = (
            "Create up to five short tags describing this product, separated by commas:\n"
            + description
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        text = resp.choices[0].message.content
        return [t.strip() for t in text.split(",") if t.strip()]
    except Exception as e:
        st.warning(f"OpenAI tag generation failed: {e}")
        return []

