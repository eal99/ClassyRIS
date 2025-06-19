# app/clip_utils.py

import numpy as np
from gradio_client import Client, handle_file
import logging

HUGGING_FACE_URL = "elev802/CLIP-Large-Image-Search"

def generate_image_embedding(file_path):
    """
    Generate image embedding using the Hugging Face Space with the CLIP model.
    :param file_path: Path to the image file.
    :return: The embedding vector for the image as a list.
    """
    try:
        client = Client(HUGGING_FACE_URL)
        result = client.predict(
            image=handle_file(file_path),
            api_name="/predict",
        )
        print("Raw result:", result)
        # If string, parse properly
        if isinstance(result, str):
            embedding = np.fromstring(result.strip('[]'), sep=',')
        elif isinstance(result, (list, np.ndarray)):
            embedding = np.array(result)
            if embedding.ndim == 2 and embedding.shape[0] == 1:
                embedding = embedding[0]
        else:
            raise ValueError("Unexpected response format from Hugging Face API")
        embedding = embedding.tolist()
        assert len(embedding) == 768, f"Embedding length is {len(embedding)}, should be 768"
        return embedding
    except Exception as e:
        logging.error(f"Error in generate_image_embedding: {e}")
        raise