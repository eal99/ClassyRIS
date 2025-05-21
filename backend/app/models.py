from __future__ import annotations
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class SearchResult(BaseModel):
    payload: Dict[str, Any]
    score: Optional[float] = None


class VectorSearchRequest(BaseModel):
    vector: List[float]
    vector_name: str
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]] = None


class HybridSearchRequest(BaseModel):
    vectors: Dict[str, List[float]]
    top_k: int = 5
    filters: Optional[Dict[str, List[str]]] = None
