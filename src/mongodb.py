import logging
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import List

from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
import datetime

from pymongo.server_api import ServerApi

from src import config, main
import ollama

from src.models import response_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result
    return wrapper

global client, db, meta_collection, embeddings_collection

def add_chunk_to_database(chunk_id, text, intent=None, category=None, chunk_type="qa"):
    """Add a text chunk and its embedding to MongoDB."""
    try:
        # Create embedding
        embedding = main.ollama_client.embed(model=config.EMBEDDING_MODEL, input=text)['embeddings'][0]

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

@timing_decorator
async def retrieve_from_mongodb(query: str, top_n: int = 3) -> List[response_model.RetrievedChunk]:
    """Ultra-fast version with minimal candidates and direct lookup"""
    try:
        # await benchmark_retrieval(query, top_n)

        logger.info(f"Starting fast retrieval for query: '{query[:50]}...'")

        # Create query embedding (this is the main bottleneck)
        query_embedding = main.ollama_client.embed(model=config.EMBEDDING_MODEL, input=query)['embeddings'][0]

        # Minimal vector search - just get the IDs
        vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(10, top_n * 2),  # Minimal candidates
                    "limit": top_n
                }
            },
            {
                "$project": {
                    "chunk_id": 1,
                    "similarity": {"$meta": "vectorSearchScore"},
                    "_id": 0
                }
            }
        ]

        # Execute vector search
        vector_results = list(embeddings_collection.aggregate(vector_pipeline))
        logger.info(f"Vector search found {len(vector_results)} results")

        if not vector_results:
            return []

        # Direct lookup in metadata collection
        chunk_ids = [doc['chunk_id'] for doc in vector_results]
        metadata_cursor = meta_collection.find(
            {"chunk_id": {"$in": chunk_ids}},
            {"chunk_id": 1, "text": 1, "intent": 1, "category": 1, "_id": 0}
        )

        # Create lookup dictionary
        metadata_dict = {doc['chunk_id']: doc for doc in metadata_cursor}

        # Combine results
        retrieved_chunks = []
        for vector_doc in vector_results:
            chunk_id = vector_doc['chunk_id']
            metadata = metadata_dict.get(chunk_id)

            if metadata:
                retrieved_chunks.append(
                    response_model.RetrievedChunk(
                        chunk_id=chunk_id,
                        text=metadata.get('text', ''),
                        similarity=vector_doc['similarity'],
                        intent=metadata.get('intent'),
                        category=metadata.get('category')
                    )
                )

        logger.info(f"Fast retrieval completed. Returning {len(retrieved_chunks)} chunks")
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error during fast retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Fast retrieval error: {str(e)}")

async def benchmark_retrieval(
    query: str,
    top_n: int = 3
) -> None:
    """
    Benchmark individual stages of retrieval:
    - Embedding generation
    - Vector search
    - Metadata lookup
    Prints timings for each to console.
    """
    # 1. Embedding creation timing
    start = time.time()
    q_emb = ollama.embed(model=config.EMBEDDING_MODEL, input=query)['embeddings'][0]
    print(f"Embedding creation: {time.time() - start:.2f}s")

    # 2. Vector search timing
    start = time.time()
    pipeline = [
        {"$vectorSearch": {"index": "vector_index", "path": "embedding",
                             "queryVector": q_emb, "numCandidates": 10, "limit": top_n}},
        {"$project": {"chunk_id": 1, "similarity": {"$meta": "vectorSearchScore"}}}
    ]
    vr = list(embeddings_collection.aggregate(pipeline))
    print(f"Vector search: {time.time() - start:.2f}s")

    # 3. Metadata lookup timing
    start = time.time()
    m_ids = [d['chunk_id'] for d in vr]
    list(meta_collection.find({"chunk_id": {"$in": m_ids}}))
    print(f"Metadata lookup: {time.time() - start:.2f}s")

async def warm_up_model() -> None:
    """
    Perform a dummy retrieval and generation to preload models
    and connections, improving first-call latency.
    """
    try:
        # Retrieve a few dummy chunks
        chunks = await retrieve_from_mongodb('hello', 3)
        if not chunks:
            raise HTTPException(404, "No data for warm-up retrieval")
        # Generate a synthetic chat response
        await main.generate_chat_response('hello', chunks)
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")

# MongoDB connection management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context:
    - Connects to MongoDB on startup
    - Warms up models
    - Closes connection on shutdown
    """
    # Startup actions
    await connect_to_mongodb()
    await warm_up_model()
    yield
    # Shutdown actions
    await close_mongodb_connection()


async def connect_to_mongodb() -> None:
    """
    Initialize and verify MongoDB connection and collections.
    """
    global client, db, meta_collection, embeddings_collection
    try:
        # Establish client with server API version
        client = MongoClient(config.MONGO_URI, server_api=ServerApi('1'))
        db = client[config.DATABASE_NAME]
        meta_collection = db[config.META_COLLECTION]
        embeddings_collection = db[config.EMBEDDINGS_COLLECTION]

        # Ping to confirm connectivity
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB!")

        # Log counts for monitoring
        mcount = meta_collection.count_documents({})
        e_count = embeddings_collection.count_documents({})
        logger.info(f"Collections ready - Meta: {mcount}, Embeddings: {e_count}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection() -> None:
    """
    Gracefully close the global MongoDB client connection.
    """
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")
