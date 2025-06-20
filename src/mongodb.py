import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import datetime
from src import config
import ollama

from src import utils
from src.models import response_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB Configuration
# client = MongoClient(config.MONGO_URI)
# db = client[config.DATABASE_NAME]
# meta_collection = db[config.META_COLLECTION]
# embeddings_collection = db[config.EMBEDDINGS_COLLECTION]

global client, db, meta_collection, embeddings_collection

def connect_to_mongo():
    """Test MongoDB connection."""
    try:
        client.admin.command('ping')
        print(f"Successfully connected to MongoDB: {config.DATABASE_NAME}")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        exit(1)

def create_indexes():
    """Create necessary indexes in MongoDB collections."""
    try:
        meta_collection.create_index("chunk_id")
        meta_collection.create_index("intent")
        meta_collection.create_index("category")

        embeddings_collection.create_index("chunk_id")
        print("Indexes created successfully!")
    except Exception as e:
        print(f"Warning: Could not create indexes: {e}")

def add_chunk_to_database(chunk_id, text, intent=None, category=None, chunk_type="qa"):
    """Add a text chunk and its embedding to MongoDB."""
    try:
        # Create embedding
        embedding = ollama.embed(model=config.EMBEDDING_MODEL, input=text)['embeddings'][0]

        # Store metadata
        meta_doc = {
            "chunk_id": chunk_id,
            "text": text,
            "intent": intent,
            "category": category,
            "chunk_type": chunk_type,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
            "text_length": len(text)
        }
        meta_collection.insert_one(meta_doc)

        # Store embedding
        embedding_doc = {
            "chunk_id": chunk_id,
            "embedding": embedding,
            "embedding_model": config.EMBEDDING_MODEL,
            "created_at": datetime.datetime.now(datetime.timezone.utc)
        }
        embeddings_collection.insert_one(embedding_doc)
        return True
    except Exception as e:
        print(f"Error adding chunk {chunk_id} to database: {e}")
        return False


async def retrieve_from_mongodb(query: str, top_n: int = 3) -> List[response_model.RetrievedChunk]:
    """Retrieve most similar chunks from MongoDB"""
    try:
        logger.info(f"Starting retrieval for query: '{query[:50]}...'")

        # Create query embedding - this might be slow
        logger.info("Creating query embedding...")
        query_embedding = ollama.embed(model=config.EMBEDDING_MODEL, input=query)['embeddings'][0]
        logger.info(f"Query embedding created successfully (dim: {len(query_embedding)})")

        # Get embeddings from MongoDB with limit to avoid memory issues
        logger.info("Fetching embeddings from MongoDB...")
        embedding_docs = list(embeddings_collection.find({}).limit(1000))  # Limit for testing
        logger.info(f"Retrieved {len(embedding_docs)} embedding documents")

        if not embedding_docs:
            logger.warning("No embeddings found in database!")
            return []

        similarities = []
        logger.info("Calculating similarities...")

        for i, emb_doc in enumerate(embedding_docs):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(embedding_docs)} embeddings")

            chunk_id = emb_doc['chunk_id']
            embedding = emb_doc['embedding']

            # Get corresponding metadata
            meta_doc = meta_collection.find_one({"chunk_id": chunk_id})
            if meta_doc:
                similarity = utils.cosine_similarity(query_embedding, embedding)
                similarities.append(response_model.RetrievedChunk(
                    chunk_id=chunk_id,
                    text=meta_doc['text'],
                    similarity=similarity,
                    intent=meta_doc.get('intent'),
                    category=meta_doc.get('category')
                ))

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        logger.info(f"Similarity calculation complete. Returning top {top_n} results")
        return similarities[:top_n]

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

# MongoDB connection management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    await connect_to_mongodb()
    yield
    # Shutdown
    await close_mongodb_connection()


async def connect_to_mongodb():
    """Initialize MongoDB connection"""

    global client, db, meta_collection, embeddings_collection

    try:
        client = MongoClient(config.MONGO_URI)
        db = client[config.DATABASE_NAME]
        meta_collection = db[config.META_COLLECTION]
        embeddings_collection = db[config.EMBEDDINGS_COLLECTION]

        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")

        # Log collection stats
        meta_count = meta_collection.count_documents({})
        embeddings_count = embeddings_collection.count_documents({})
        logger.info(f"Collections ready - Meta: {meta_count}, Embeddings: {embeddings_count}")

    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")