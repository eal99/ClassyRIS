from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, UploadFile, File

# allow importing modules from repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    # Redirect local "app" package to the repository root so imports like
    # ``from app import qdrant_utils`` resolve correctly when running from this
    # directory.
    pkg = sys.modules.get("app")
    if pkg and Path(getattr(pkg, "__file__", "")).parent == Path(__file__).parent:
        pkg.__path__.insert(0, str(ROOT_DIR / "app"))

from app import qdrant_utils, embedding, data_utils, openai_utils
from shared_types.api_models import (
    VectorSearchRequest,
    HybridSearchRequest,
    ChatRequest,
    TextSearchRequest,
)

import redis
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL)

app = FastAPI(title="ClassyRIS API")




@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/search/vector")
def search_vector(req: VectorSearchRequest):
    try:
        results = qdrant_utils.vector_search(
            req.vector, req.vector_name, req.top_k, req.filters or {}
        )
        return [r.model_dump() if hasattr(r, "model_dump") else r.payload for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/text")
def search_text(req: TextSearchRequest):
    """Compute a text embedding and perform vector search."""
    try:
        emb = embedding.get_text_embedding(req.query)
        results = qdrant_utils.vector_search(
            emb, "text", req.top_k, req.filters or {}
        )
        return [r.model_dump() if hasattr(r, "model_dump") else r.payload for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    """Search using an uploaded image."""
    try:
        from PIL import Image
        img = Image.open(file.file).convert("RGB")
        emb = embedding.get_image_embedding(img)
        results = qdrant_utils.vector_search(emb, "image", top_k, {})
        return [r.model_dump() if hasattr(r, "model_dump") else r.payload for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid")
def search_hybrid(req: HybridSearchRequest):
    try:
        results = qdrant_utils.hybrid_search(req.vectors, req.top_k, req.filters or {})
        return [r.model_dump() if hasattr(r, "model_dump") else r.payload for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/summary")
def analytics_summary():
    df = data_utils.art_df
    total = len(df)
    response = {"total_products": total}
    if "ecom_price" in df.columns:
        prices = df["ecom_price"].astype(float)
        response["average_price"] = prices.mean()
    return response


@app.post("/chat")
def chat(req: ChatRequest):
    key = f"chat:{req.session_id}"
    history = redis_client.lrange(key, 0, -1)
    messages = [m.decode() for m in history]
    messages.append(req.message)
    redis_client.rpush(key, req.message)
    # In a real implementation, you'd call an LLM here with the conversation
    # For now we'll echo back the last message
    reply = openai_utils.summarize_description(req.message)
    redis_client.rpush(key, reply)
    return {"reply": reply, "history": messages + [reply]}
