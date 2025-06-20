# config.py
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Accessing environment variables
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME")
META_COLLECTION = os.getenv("META_COLLECTION")
EMBEDDINGS_COLLECTION = os.getenv("EMBEDDINGS_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL")

HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_DATASET_SPLIT = os.getenv("HF_DATASET_SPLIT")