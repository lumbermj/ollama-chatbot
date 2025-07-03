# config.py
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Accessing environment variables
# MONGO_URI = os.getenv("MONGO_URI")
# DATABASE_NAME = os.getenv("DATABASE_NAME")
# META_COLLECTION = os.getenv("META_COLLECTION")
# EMBEDDINGS_COLLECTION = os.getenv("EMBEDDINGS_COLLECTION")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# LANGUAGE_MODEL = os.getenv("LANGUAGE_MODEL")
#
# HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
# HF_DATASET_SPLIT = os.getenv("HF_DATASET_SPLIT")

MONGO_URI = 'mongodb+srv://lumbermj74:QAZwsx121212@oc1.kuxifga.mongodb.net/?retryWrites=true&w=majority&appName=oc1'
DATABASE_NAME = 'mj'
META_COLLECTION = 'customer-support-meta'
EMBEDDINGS_COLLECTION = 'customer-support-embeddings'
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

HF_DATASET_NAME = 'bitext/Bitext-customer-support-llm-chatbot-training-dataset'
HF_DATASET_SPLIT = 'train'