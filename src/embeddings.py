import ollama
from src import config

def create_embedding(text):
    """Create embedding for a given text."""
    try:
        embedding = ollama.embed(model=config.EMBEDDING_MODEL, input=text)['embeddings'][0]
        return embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None