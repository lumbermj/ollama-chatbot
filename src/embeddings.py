"""
Module: embeddings

Utility for generating embeddings via the Ollama API.
Creates vector representations of text for use in retrieval and RAG.
"""

import ollama
from src import config


def create_embedding(text: str) -> list[float] | None:
    """
    Generate an embedding vector for the given text input using Ollama.

    Args:
        text (str): The text to embed.

    Returns:
        list[float] | None: The embedding vector if successful, otherwise None.
    """
    try:
        # Send the text to Ollama's embed endpoint with the configured model
        response = ollama.embed(
            model=config.EMBEDDING_MODEL,
            input=text
        )
        # Extract the first embedding from the response
        embedding = response['embeddings'][0]
        return embedding

    except Exception as e:
        # Log the error and return None on failure
        print(f"Error creating embedding: {e}")
        return None
