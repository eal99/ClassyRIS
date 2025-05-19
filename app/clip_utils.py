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
        # Initialize the Gradio client
        client = Client(HUGGING_FACE_URL)

        # Send the file using handle_file
        result = client.predict(
            image=handle_file(file_path),
            api_name="/predict",
        )

        # Ensure the result is a numpy array or similar
        if isinstance(result, (np.ndarray, list)):
            embedding = result
        elif isinstance(result, str):
            # Convert string response to list (if stringified numpy array)
            embedding = np.fromstring(result.strip('[]'), sep=' ')
        else:
            raise ValueError("Unexpected response format from Hugging Face API")

        # Ensure the embedding is a plain Python list of floats
        return [float(x) for x in embedding]

    except Exception as e:
        logging.error(f"Error in generate_image_embedding: {e}")
        raise