import streamlit as st
st.set_page_config(page_title="Classy Search", layout="wide", page_icon="🎨")

from app import search

if __name__ == "__main__":
    search.render()
