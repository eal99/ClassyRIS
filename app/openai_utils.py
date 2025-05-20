"""Utility functions for future OpenAI-powered features."""

from openai import OpenAI
import os
import streamlit as st

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_description(description: str) -> str:
    """Return a short summary of a product description using OpenAI."""
    if not description:
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

