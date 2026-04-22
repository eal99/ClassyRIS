import numpy as np
import pandas as pd
import streamlit as st

from app import config
from app.data_utils import art_df, get_image_url_by_sku
from app.qdrant_utils import (
    audit_collection,
    build_image_vector,
    get_all_skus,
    get_missing_vector_points,
    get_sku_id_map,
    set_payload_by_id,
    update_vectors_batch,
)


def _row_to_payload(row: pd.Series, include_empty: bool) -> dict:
    payload: dict = {}
    for col, val in row.items():
        if pd.isna(val):
            continue
        if isinstance(val, str):
            cleaned = val.strip()
            if not cleaned and not include_empty:
                continue
            payload[col] = cleaned
            continue
        if isinstance(val, np.generic):
            payload[col] = val.item()
            continue
        payload[col] = val
    return payload


def render() -> None:
    st.set_page_config(page_title="Admin", layout="wide", page_icon="⚙️")
    st.header("Admin Tools")
    st.markdown("Catalog diagnostics and Qdrant maintenance helpers.")

    st.write(f"Dataset contains **{len(art_df)}** products.")
    st.write("Columns:", ", ".join(art_df.columns))
    st.dataframe(art_df.head())

    st.subheader("CSV vs Qdrant SKU check")
    if "sku" not in art_df.columns:
        st.warning("CSV is missing a `sku` column after normalization.")
        return

    if st.button("Compare CSV SKUs to Qdrant"):
        with st.spinner("Fetching SKUs from Qdrant..."):
            qdrant_skus = get_all_skus()
        csv_skus = set(art_df["sku"].dropna().astype(str))
        missing = sorted(csv_skus - qdrant_skus)
        st.write(f"CSV SKUs: {len(csv_skus):,}")
        st.write(f"Qdrant SKUs: {len(qdrant_skus):,}")
        st.write(f"Missing in Qdrant: {len(missing):,}")
        if missing:
            st.dataframe(pd.DataFrame({"missing_sku": missing}).head(200))
            csv = pd.DataFrame({"missing_sku": missing}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download missing SKUs",
                csv,
                "missing_skus.csv",
                mime="text/csv",
            )

    st.subheader("Qdrant audit")
    st.markdown("Checks indexed payload fields and the new named vectors for missing data.")

    required_payload = [
        "sku",
        "short_description",
        "family_collection",
        "status_facet",
        "primary_image_url",
    ]
    required_vectors = [
        config.QDRANT_IMAGE_VECTOR,
        config.QDRANT_TEXT_DENSE_VECTOR,
        config.QDRANT_TEXT_SPARSE_VECTOR,
    ]
    expected_dims = {
        config.QDRANT_IMAGE_VECTOR: 512,
        config.QDRANT_TEXT_DENSE_VECTOR: 3072,
    }

    sample_size = st.slider("Sample size", 100, 5000, 1000, step=100)
    full_scan = st.checkbox("Scan entire collection (slow)")
    st.caption("Expected dense vector dimensions")
    expected_dims[config.QDRANT_IMAGE_VECTOR] = st.number_input(
        f"{config.QDRANT_IMAGE_VECTOR} dim",
        value=expected_dims[config.QDRANT_IMAGE_VECTOR],
        step=1,
    )
    expected_dims[config.QDRANT_TEXT_DENSE_VECTOR] = st.number_input(
        f"{config.QDRANT_TEXT_DENSE_VECTOR} dim",
        value=expected_dims[config.QDRANT_TEXT_DENSE_VECTOR],
        step=1,
    )

    if st.button("Run Qdrant audit"):
        with st.spinner("Auditing Qdrant collection..."):
            audit = audit_collection(
                required_payload_fields=required_payload,
                required_vectors=required_vectors,
                sample_size=None if full_scan else sample_size,
                expected_vector_dims=expected_dims,
            )
        checked = audit["totals"]["checked"]
        st.write(f"Points checked: {checked:,}")

        st.subheader("Missing payload fields")
        payload_rows = [
            {
                "field": field,
                "missing_count": count,
                "missing_pct": (count / checked * 100.0) if checked else 0.0,
                "examples": ", ".join(audit["missing_examples"].get(f"payload:{field}", [])),
            }
            for field, count in audit["payload_missing"].items()
        ]
        st.dataframe(pd.DataFrame(payload_rows))

        st.subheader("Missing vectors")
        vector_rows = [
            {
                "vector": vec,
                "missing_count": count,
                "missing_pct": (count / checked * 100.0) if checked else 0.0,
                "examples": ", ".join(audit["missing_examples"].get(f"vector:{vec}", [])),
            }
            for vec, count in audit["vector_missing"].items()
        ]
        st.dataframe(pd.DataFrame(vector_rows))

        st.subheader("Vector dimension mismatches")
        dim_rows = [
            {
                "vector": vec,
                "expected_dim": expected_dims.get(vec),
                "mismatch_count": count,
                "mismatch_pct": (count / checked * 100.0) if checked else 0.0,
            }
            for vec, count in audit["vector_dim_mismatch"].items()
        ]
        st.dataframe(pd.DataFrame(dim_rows))

    st.subheader("Sync CSV payloads to Qdrant (merge)")
    st.markdown("Uses `sku` to match points and merges non-empty CSV fields into the existing payload.")
    include_empty = st.checkbox("Include empty string values from CSV", value=False)
    if st.button("Sync payloads from CSV"):
        with st.spinner("Building SKU → point ID map..."):
            sku_id_map = get_sku_id_map()

        missing = []
        updated = 0
        total = len(art_df)
        progress = st.progress(0)
        for idx, row in art_df.iterrows():
            sku = row.get("sku")
            if pd.isna(sku):
                continue
            sku_key = str(sku).strip()
            point_id = sku_id_map.get(sku_key)
            if not point_id:
                missing.append(sku_key)
                continue
            payload = _row_to_payload(row, include_empty=include_empty)
            if not payload:
                continue
            set_payload_by_id(point_id, payload)
            updated += 1
            if updated % 50 == 0 or idx == total - 1:
                progress.progress(min((idx + 1) / total, 1.0))

        st.write(f"Updated payloads: {updated:,}")
        st.write(f"Missing in Qdrant: {len(missing):,}")
        if missing:
            st.dataframe(pd.DataFrame({"missing_sku": missing}).head(200))
            csv = pd.DataFrame({"missing_sku": missing}).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download missing SKUs",
                csv,
                "missing_skus_payload_sync.csv",
                mime="text/csv",
            )

    st.subheader("Backfill missing image vectors")
    st.markdown(
        "Backfills missing image vectors using Qdrant Cloud inference and the point image URL."
    )
    max_points = st.number_input("Max points to process", min_value=1, max_value=2000, value=200, step=50)
    batch_size = st.slider("Qdrant batch size", 1, 100, 25)
    if st.button("Backfill image vectors"):
        with st.spinner("Finding points missing image vectors..."):
            missing = get_missing_vector_points(config.QDRANT_IMAGE_VECTOR, limit=int(max_points))

        if not missing:
            st.success("No missing image vectors found.")
            return

        st.write(f"Missing image vectors found: {len(missing):,}")
        progress = st.progress(0)
        updated = 0
        skipped = 0
        failed: list[dict[str, str]] = []
        batch: list[dict[str, object]] = []

        for idx, item in enumerate(missing, start=1):
            payload = item.get("payload", {})
            sku = str(payload.get("sku") or "")
            image_url = payload.get("primary_image_url") or payload.get("main_image_file")
            if not image_url:
                image_url = get_image_url_by_sku(sku)

            if not image_url:
                skipped += 1
            else:
                try:
                    batch.append(
                        {
                            "id": item.get("id"),
                            "vector": {
                                config.QDRANT_IMAGE_VECTOR: build_image_vector(image_url=str(image_url))
                            },
                        }
                    )
                    if len(batch) >= int(batch_size):
                        update_vectors_batch(batch)
                        updated += len(batch)
                        batch = []
                except Exception as exc:
                    failed.append({"sku": sku, "error": str(exc)})

            progress.progress(min(idx / len(missing), 1.0))

        if batch:
            update_vectors_batch(batch)
            updated += len(batch)

        st.write(f"Updated image vectors: {updated:,}")
        st.write(f"Skipped (no image URL): {skipped:,}")
        st.write(f"Failed: {len(failed):,}")
        if failed:
            st.dataframe(pd.DataFrame(failed).head(50))
