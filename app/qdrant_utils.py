import os
from io import BytesIO
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

from dotenv import load_dotenv
from PIL import Image, ImageOps
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config

load_dotenv()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def get_client() -> QdrantClient:
    if not hasattr(get_client, "instance"):
        get_client.instance = QdrantClient(
            url=config.QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT,
            cloud_inference=config.QDRANT_CLOUD_INFERENCE,
            check_compatibility=False,
        )
    return get_client.instance


def get_local_inference_client() -> QdrantClient:
    if not hasattr(get_local_inference_client, "instance"):
        get_local_inference_client.instance = QdrantClient(
            url=config.QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=config.QDRANT_TIMEOUT,
            cloud_inference=False,
            check_compatibility=False,
        )
    return get_local_inference_client.instance


def build_filter(payload_filters: dict | None) -> qmodels.Filter | None:
    must_conditions = []
    for field, values in (payload_filters or {}).items():
        if not values:
            continue
        vals = values if isinstance(values, (list, tuple, set)) else [values]
        matcher = (
            qmodels.MatchAny(any=list(vals))
            if len(vals) > 1
            else qmodels.MatchValue(value=list(vals)[0])
        )
        must_conditions.append(qmodels.FieldCondition(key=field, match=matcher))
    return qmodels.Filter(must=must_conditions) if must_conditions else None


def _payload_selector(include_full_payload: bool) -> bool | list[str]:
    return True if include_full_payload else config.SEARCH_PAYLOAD_FIELDS


def _prefetch_limit(top_k: int) -> int:
    return max(top_k, config.SEARCH_PREFETCH_LIMIT)


def _text_dense_query(text: str) -> qmodels.Document:
    return qmodels.Document(
        text=text,
        model=config.QDRANT_TEXT_DENSE_MODEL,
        options=_provider_options_for_model(config.QDRANT_TEXT_DENSE_MODEL),
    )


def _text_sparse_query(text: str) -> qmodels.Document:
    return qmodels.Document(text=text, model=config.QDRANT_TEXT_SPARSE_MODEL)


def _provider_options_for_model(model_name: str) -> dict[str, str] | None:
    if model_name.startswith("openai/"):
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is required for Qdrant inference with OpenAI models."
            )
        return {"openai-api-key": OPENAI_API_KEY}
    return None


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, (255, 255, 255))
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        background.paste(image.convert("RGB"), mask=alpha)
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _resize_to_max_dimension(image: Image.Image, max_dimension: int) -> Image.Image:
    longest_side = max(image.size)
    if longest_side <= max_dimension:
        return image
    scale = max_dimension / float(longest_side)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _save_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
    )
    return buffer.getvalue()


def _prepare_query_image_bytes(image_bytes: bytes) -> bytes:
    with Image.open(BytesIO(image_bytes)) as raw_image:
        image = _flatten_to_rgb(raw_image)
    resized = _resize_to_max_dimension(image, config.QDRANT_QUERY_IMAGE_MAX_DIMENSION)
    return _save_jpeg_bytes(resized, quality=config.QDRANT_QUERY_IMAGE_JPEG_QUALITY)


@contextmanager
def _temporary_query_image(image_bytes: bytes):
    prepared = _prepare_query_image_bytes(image_bytes)
    tmp = NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        tmp.write(prepared)
        tmp.flush()
        tmp.close()
        yield tmp.name
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass


def _search_uploaded_image_locally(
    top_k: int,
    payload_filters: dict | None = None,
    include_full_payload: bool = False,
    image_bytes: bytes | None = None,
) -> list[qmodels.ScoredPoint]:
    if not image_bytes:
        return []
    try:
        client = get_local_inference_client()
        with _temporary_query_image(image_bytes) as image_path:
            resp = client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=qmodels.Image(
                    image=image_path,
                    model=config.QDRANT_LOCAL_IMAGE_MODEL,
                ),
                using=config.QDRANT_IMAGE_VECTOR,
                limit=top_k,
                query_filter=build_filter(payload_filters),
                with_payload=_payload_selector(include_full_payload),
            )
        return resp.points
    except Exception as exc:
        raise RuntimeError(
            "Uploaded-file image search uses local FastEmbed because Qdrant Cloud "
            "image inference rejects inline uploads for this collection. Install "
            "`qdrant-client[fastembed]` in the app environment to enable it."
        ) from exc


def _fuse_scored_points_rrf(
    result_sets: list[list[qmodels.ScoredPoint]],
    top_k: int,
    rrf_k: int = 60,
) -> list[SimpleNamespace]:
    combined_scores: dict[str, float] = {}
    best_points: dict[str, qmodels.ScoredPoint] = {}
    for results in result_sets:
        for rank, point in enumerate(results, start=1):
            point_id = str(getattr(point, "id", ""))
            if not point_id:
                continue
            combined_scores[point_id] = combined_scores.get(point_id, 0.0) + 1.0 / (rrf_k + rank)
            best_points.setdefault(point_id, point)

    ordered = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[SimpleNamespace] = []
    for point_id, fused_score in ordered[:top_k]:
        point = best_points[point_id]
        fused.append(
            SimpleNamespace(
                id=getattr(point, "id", point_id),
                payload=getattr(point, "payload", None),
                score=fused_score,
            )
        )
    return fused


def build_image_vector(image_url: str | None = None, image_bytes: bytes | None = None) -> qmodels.Image:
    if image_bytes:
        raise ValueError(
            "Inline image bytes are not supported for Qdrant Cloud inference in this app. "
            "Use a public image URL or the local upload search path."
        )
    if image_url:
        return qmodels.Image(image=image_url, model=config.QDRANT_IMAGE_MODEL)
    raise ValueError("Either image_url or image_bytes must be provided.")


def search_text(
    text: str,
    top_k: int,
    payload_filters: dict | None = None,
    include_full_payload: bool = False,
) -> list[qmodels.ScoredPoint]:
    query_text = text.strip()
    if not query_text:
        return []
    client = get_client()
    resp = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            qmodels.Prefetch(
                query=_text_dense_query(query_text),
                using=config.QDRANT_TEXT_DENSE_VECTOR,
                limit=_prefetch_limit(top_k),
            ),
            qmodels.Prefetch(
                query=_text_sparse_query(query_text),
                using=config.QDRANT_TEXT_SPARSE_VECTOR,
                limit=_prefetch_limit(top_k),
            ),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        query_filter=build_filter(payload_filters),
        with_payload=_payload_selector(include_full_payload),
    )
    return resp.points


def search_image(
    top_k: int,
    payload_filters: dict | None = None,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
    include_full_payload: bool = False,
) -> list[qmodels.ScoredPoint]:
    if image_bytes and not image_url:
        return _search_uploaded_image_locally(
            top_k=top_k,
            payload_filters=payload_filters,
            include_full_payload=include_full_payload,
            image_bytes=image_bytes,
        )
    client = get_client()
    resp = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=build_image_vector(image_url=image_url, image_bytes=image_bytes),
        using=config.QDRANT_IMAGE_VECTOR,
        limit=top_k,
        query_filter=build_filter(payload_filters),
        with_payload=_payload_selector(include_full_payload),
    )
    return resp.points


def hybrid_search(
    top_k: int,
    payload_filters: dict | None = None,
    text_query: str | None = None,
    image_url: str | None = None,
    image_bytes: bytes | None = None,
    include_full_payload: bool = False,
) -> list[qmodels.ScoredPoint]:
    prefetch: list[qmodels.Prefetch] = []
    clean_text = (text_query or "").strip()

    if image_bytes and not image_url:
        image_results = _search_uploaded_image_locally(
            top_k=_prefetch_limit(top_k),
            payload_filters=payload_filters,
            include_full_payload=include_full_payload,
            image_bytes=image_bytes,
        )
        if not clean_text:
            return image_results[:top_k]
        text_results = search_text(
            clean_text,
            top_k=_prefetch_limit(top_k),
            payload_filters=payload_filters,
            include_full_payload=include_full_payload,
        )
        return _fuse_scored_points_rrf([text_results, image_results], top_k=top_k)

    if clean_text:
        prefetch.extend(
            [
                qmodels.Prefetch(
                    query=_text_dense_query(clean_text),
                    using=config.QDRANT_TEXT_DENSE_VECTOR,
                    limit=_prefetch_limit(top_k),
                ),
                qmodels.Prefetch(
                    query=_text_sparse_query(clean_text),
                    using=config.QDRANT_TEXT_SPARSE_VECTOR,
                    limit=_prefetch_limit(top_k),
                ),
            ]
        )

    if image_url or image_bytes:
        prefetch.append(
            qmodels.Prefetch(
                query=build_image_vector(image_url=image_url, image_bytes=image_bytes),
                using=config.QDRANT_IMAGE_VECTOR,
                limit=_prefetch_limit(top_k),
            )
        )

    if not prefetch:
        return []

    client = get_client()
    if len(prefetch) == 1:
        query = prefetch[0].query
        using = prefetch[0].using
        resp = client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query,
            using=using,
            limit=top_k,
            query_filter=build_filter(payload_filters),
            with_payload=_payload_selector(include_full_payload),
        )
        return resp.points

    resp = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetch,
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        query_filter=build_filter(payload_filters),
        with_payload=_payload_selector(include_full_payload),
    )
    return resp.points


def vector_search(
    vector: list[float],
    vector_name: str,
    top_k: int,
    payload_filters: dict | None = None,
    include_full_payload: bool = False,
) -> list[qmodels.ScoredPoint]:
    client = get_client()
    resp = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        using=vector_name,
        limit=top_k,
        query_filter=build_filter(payload_filters),
        with_payload=_payload_selector(include_full_payload),
    )
    return resp.points


def get_by_sku(sku: str) -> list[qmodels.ScoredPoint]:
    client = get_client()
    resp = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=None,
        limit=10,
        query_filter=build_filter({"sku": [sku.strip()]}),
        with_payload=True,
    )
    return resp.points


def get_all_skus(batch_size: int = 1000) -> set[str]:
    client = get_client()
    skus: set[str] = set()
    offset = None
    while True:
        resp = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_payload=["sku"],
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        if isinstance(resp, tuple):
            points, offset = resp
        else:
            points = getattr(resp, "points", resp)
            offset = getattr(resp, "next_page_offset", None)
        if not points:
            break
        for point in points:
            payload = getattr(point, "payload", None) or {}
            sku = payload.get("sku")
            if sku is not None:
                skus.add(str(sku))
        if offset is None:
            break
    return skus


def get_sku_id_map(batch_size: int = 1000) -> dict[str, str]:
    client = get_client()
    sku_id_map: dict[str, str] = {}
    offset = None
    while True:
        resp = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_payload=["sku"],
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        if isinstance(resp, tuple):
            points, offset = resp
        else:
            points = getattr(resp, "points", resp)
            offset = getattr(resp, "next_page_offset", None)
        if not points:
            break
        for point in points:
            payload = getattr(point, "payload", None) or {}
            sku = payload.get("sku")
            if sku is None:
                continue
            sku_id_map[str(sku)] = str(getattr(point, "id", sku))
        if offset is None:
            break
    return sku_id_map


def set_payload_by_id(point_id: str, payload: dict) -> None:
    client = get_client()
    client.set_payload(
        collection_name=config.QDRANT_COLLECTION,
        payload=payload,
        points=[point_id],
        wait=True,
    )


def _vector_is_missing(vector: object) -> bool:
    if vector is None:
        return True
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is not None or values is not None:
        return not indices or not values
    if isinstance(vector, dict) and {"indices", "values"} <= set(vector.keys()):
        return not vector["indices"] or not vector["values"]
    try:
        return len(vector) == 0
    except TypeError:
        return False


def _vector_dimension(vector: object) -> int | None:
    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)
    if indices is not None or values is not None:
        return None
    if isinstance(vector, dict) and {"indices", "values"} <= set(vector.keys()):
        return None
    try:
        return len(vector)
    except TypeError:
        return None


def get_missing_vector_points(
    vector_name: str,
    batch_size: int = 200,
    limit: int | None = 500,
) -> list[dict]:
    client = get_client()
    missing: list[dict] = []
    offset = None
    while True:
        resp = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=[vector_name],
            limit=batch_size,
            offset=offset,
        )
        if isinstance(resp, tuple):
            points, offset = resp
        else:
            points = getattr(resp, "points", resp)
            offset = getattr(resp, "next_page_offset", None)
        if not points:
            break
        for point in points:
            vectors = getattr(point, "vector", None) or {}
            vector = vectors.get(vector_name) if isinstance(vectors, dict) else None
            if _vector_is_missing(vector):
                missing.append(
                    {
                        "id": str(getattr(point, "id", "")),
                        "payload": getattr(point, "payload", None) or {},
                    }
                )
                if limit is not None and len(missing) >= limit:
                    return missing
        if offset is None:
            break
    return missing


def update_vectors_by_id(point_id: str, vectors: dict[str, object]) -> None:
    client = get_client()
    client.update_vectors(
        collection_name=config.QDRANT_COLLECTION,
        points=[qmodels.PointVectors(id=point_id, vector=vectors)],
        wait=True,
    )


def update_vectors_batch(points: list[dict[str, object]]) -> None:
    if not points:
        return
    client = get_client()
    payload = [
        qmodels.PointVectors(id=str(point["id"]), vector=point["vector"])
        for point in points
        if "id" in point and "vector" in point
    ]
    if not payload:
        return
    client.update_vectors(
        collection_name=config.QDRANT_COLLECTION,
        points=payload,
        wait=True,
    )


def audit_collection(
    required_payload_fields: list[str],
    required_vectors: list[str],
    sample_size: int | None = 1000,
    batch_size: int = 200,
    expected_vector_dims: dict[str, int] | None = None,
) -> dict:
    client = get_client()
    totals = {"checked": 0}
    payload_missing = {field: 0 for field in required_payload_fields}
    vector_missing = {vec: 0 for vec in required_vectors}
    vector_dim_mismatch = {vec: 0 for vec in (expected_vector_dims or {})}
    missing_examples: dict[str, list[str]] = {
        **{f"payload:{field}": [] for field in required_payload_fields},
        **{f"vector:{vec}": [] for vec in required_vectors},
    }

    offset = None
    while True:
        resp = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=required_vectors,
            limit=batch_size,
            offset=offset,
        )
        if isinstance(resp, tuple):
            points, offset = resp
        else:
            points = getattr(resp, "points", resp)
            offset = getattr(resp, "next_page_offset", None)
        if not points:
            break

        for point in points:
            totals["checked"] += 1
            payload = getattr(point, "payload", None) or {}
            sku = payload.get("sku")
            sku_label = str(sku) if sku is not None else str(getattr(point, "id", "unknown"))

            for field in required_payload_fields:
                value = payload.get(field)
                if value is None or (isinstance(value, str) and not value.strip()):
                    payload_missing[field] += 1
                    key = f"payload:{field}"
                    if len(missing_examples[key]) < 5:
                        missing_examples[key].append(sku_label)

            vectors = getattr(point, "vector", None) or {}
            for vec in required_vectors:
                vector = vectors.get(vec)
                if _vector_is_missing(vector):
                    vector_missing[vec] += 1
                    key = f"vector:{vec}"
                    if len(missing_examples[key]) < 5:
                        missing_examples[key].append(sku_label)
                    continue

                if expected_vector_dims and vec in expected_vector_dims:
                    actual_dim = _vector_dimension(vector)
                    expected_dim = expected_vector_dims[vec]
                    if actual_dim is not None and actual_dim != expected_dim:
                        vector_dim_mismatch[vec] += 1

            if sample_size is not None and totals["checked"] >= sample_size:
                return {
                    "totals": totals,
                    "payload_missing": payload_missing,
                    "vector_missing": vector_missing,
                    "vector_dim_mismatch": vector_dim_mismatch,
                    "missing_examples": missing_examples,
                }

        if offset is None:
            break

    return {
        "totals": totals,
        "payload_missing": payload_missing,
        "vector_missing": vector_missing,
        "vector_dim_mismatch": vector_dim_mismatch,
        "missing_examples": missing_examples,
    }
