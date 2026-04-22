# ClassyRIS

ClassyRIS is a Streamlit application for catalog search against Qdrant. The app now targets the `classyliving_products_current` alias by default and uses Qdrant Cloud inference for text and image queries instead of generating search embeddings locally.

Uploaded file image search is handled differently from URL-based image search: on deployed/public Streamlit runs, uploads are exposed through a temporary `/media/...` URL and sent to Qdrant Cloud inference as an HTTPS image URL. Set `PUBLIC_BASE_URL` to the public app origin so the app can always turn those relative media paths into absolute URLs. On local or private hosts where that URL would not be publicly reachable, the app falls back to optional local FastEmbed image inference using `Qdrant/clip-ViT-B-32-vision`.

## Defaults

- Qdrant URL: `https://8a5d6688-43c7-453b-9744-8c25e746fd04.us-east-1-0.aws.cloud.qdrant.io`
- Collection alias: `classyliving_products_current`
- Text dense vector: `text_dense`
- Text sparse vector: `text_sparse`
- Image vector: `image_dense`

## Required Environment Variables

- `QDRANT_API_KEY` – required for Qdrant Cloud access

## Optional Environment Variables

- `QDRANT_URL` – override the default Qdrant Cloud URL
- `QDRANT_COLLECTION` – override the default alias
- `QDRANT_TIMEOUT` – HTTP timeout in seconds
- `QDRANT_CLOUD_INFERENCE` – set to `false` to disable Qdrant inference objects
- `PUBLIC_BASE_URL` – public app origin used to turn Streamlit `/media/...` paths into absolute URLs for uploaded-image search
- `QDRANT_TEXT_DENSE_MODEL` – defaults to `openai/text-embedding-3-large`
- `QDRANT_TEXT_SPARSE_MODEL` – defaults to `Qdrant/bm25`
- `QDRANT_IMAGE_MODEL` – defaults to `qdrant/clip-vit-b-32-vision`
- `PRODUCTS_CSV` – override the CSV used for filter options and payload sync

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run Search.py
```

The admin tools are available from the Streamlit multipage navigation under `Admin`. They include:

- CSV vs Qdrant SKU comparison
- Qdrant audit for payload/vector completeness
- CSV payload sync into Qdrant by SKU
- Image vector backfill using Qdrant Cloud inference
