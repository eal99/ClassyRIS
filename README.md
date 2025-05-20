# ClassyRIS

ClassyRIS is a Streamlit-based application for searching an artwork catalog using both text and images. It relies on OpenAI embeddings and a Qdrant vector database for retrieval.

## Installation

1. Clone the repository.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

The app expects several environment variables to be set:

- `OPENAI_API_KEY` – API key for accessing OpenAI embeddings.
- `QDRANT_URL` – URL for your Qdrant instance.
- `QDRANT_API_KEY` – API key for Qdrant (if required).
- `QDRANT_COLLECTION` – Name of the Qdrant collection (defaults to `Classy_Art`).

You can place these in a `.env` file or set them in your shell before running the app.

## Running the App

Start the Streamlit server:

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`.

A `company_logo.png` image is included and appears in the user interface. Feel free to replace it with your own branding.

## Screenshot

Below is a placeholder screenshot of the running UI (replace with your own if desired):

![Screenshot](company_logo.png)

