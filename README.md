# ClassyRIS Modernized

This repository now contains a starter monorepo for a modern rewrite of the original Streamlit app.  The goal is to provide a scalable architecture with a TypeScript/React frontend and a FastAPI backend.

```
./frontend       Next.js application (TypeScript)
./backend        FastAPI service with REST/async APIs
./shared-types   Shared API models
```

The legacy Streamlit prototype still lives in the root `app/` directory for reference.

## Local Development

1. Create a `.env` file with the following variables:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY` (optional if your Qdrant instance is open)
   - `QDRANT_COLLECTION` (defaults to `Classy_Art`)
2. Install dependencies for the frontend and backend (optional when using Docker).
   A setup script is provided to install everything in one step:

```bash
./.codex/setup.sh
```

3. Build and start all services via Docker Compose:

```bash
docker compose up --build
```

The frontend will be available on `http://localhost:3000` and the backend on `http://localhost:8000`.
Hot reloading is enabled for both services when using the default compose configuration.

## Backend

The FastAPI app exposes search, analytics and chat endpoints and reuses the existing Qdrant/OpenAI utilities.  See `backend/app/main.py` for the implementation.

Run locally with **one** of the following commands:

From the repository root:

```bash
uvicorn backend.app.main:app --reload
```

Or from inside the `backend` directory:

```bash
PYTHONPATH=.. uvicorn app.main:app --reload
```

Running from any other directory may import the wrong `app` package because of
the legacy `app/` folder in the repo root.

Main API endpoints:

```
POST /search/text   - search using a text query
POST /search/image  - search using an uploaded image
POST /search/vector - search using a raw embedding vector
POST /search/hybrid - combine multiple vectors using RRF
GET  /analytics/summary
POST /chat
```

## Frontend

A minimal Next.js project with route based pages (`/search`, `/analytics`, `/chat`, `/imagesearch`, `/hub`).  Each page calls the backend APIs and is styled with Tailwind CSS and Material‑UI components.

## Notes

This rewrite is a starting point only.  It does not yet include production authentication, monitoring or the full analytics database.  Those pieces can be added on top of this scaffolding.

## Next Steps

* Connect the backend to a persistent database and add authentication.
* Flesh out the React components for a richer user experience.
* Add automated tests for the API and frontend.
