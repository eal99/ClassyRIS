from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class SearchResult(BaseModel):
    payload: Dict[str, Any]
    score: Optional[float]

class VectorSearchRequest(BaseModel):
    vector: List[float]
    vector_name: str
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]]

class HybridSearchRequest(BaseModel):
    vectors: Dict[str, List[float]]
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]]

class ChatRequest(BaseModel):
    session_id: str
    message: str


class TextSearchRequest(BaseModel):
    """Search using a natural language query."""
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]] = None


class ImageSearchRequest(BaseModel):
    """Search using a base64 encoded image."""
    image_base64: str
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]] = None
