# main.py

import torch
# ── Prevent Streamlit’s watcher from poking into torch.classes ──
if hasattr(torch, "classes"):
    torch.classes.__path__ = []

import logging
# ── Silence the watcher’s tracebacks ──
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

import streamlit as st
from app.ui import search_interface

if __name__ == "__main__":
    search_interface()