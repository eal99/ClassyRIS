# qdrant_utils.py
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "Classy_Art_RIS"

def get_client():
    if not hasattr(get_client, 'instance'):
        get_client.instance = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return get_client.instance

def build_filter(payload_filters: dict):
    must_conditions = []
    for field, values in payload_filters.items():
        if values:
            if isinstance(values, (list, tuple)):
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=field,
                        match=(qmodels.MatchAny(any_of=values)
                               if len(values) > 1
                               else qmodels.MatchValue(value=values[0]))
                    )
                )
            else:
                must_conditions.append(
                    qmodels.FieldCondition(
                        key=field,
                        match=qmodels.MatchValue(value=values)
                    )
                )
    return qmodels.Filter(must=must_conditions) if must_conditions else None

def vector_search(vector: list[float], vector_name: str, top_k: int, payload_filters: dict = None):
    client = get_client()
    q_filter = build_filter(payload_filters or {})
    # MODERN WAY: Use query_points instead of search
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        using=vector_name,
        limit=top_k,
        query_filter=q_filter,
        with_payload=True
    )
    return result.points

def hybrid_search(vectors: dict, top_k: int, payload_filters: dict = None):
    client = get_client()
    prefetch = [
        qmodels.Prefetch(query=emb, using=field, limit=top_k)
        for field, emb in vectors.items()
    ]
    q_filter = build_filter(payload_filters or {})
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        query_filter=q_filter,
        with_payload=True
    )
    return result.points