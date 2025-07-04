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
# async def retrieve_from_mongodb(query: str, top_n: int = 3) -> List[response_model.RetrievedChunk]:
#     """Your existing function signature - just replace the body"""
#     try:
#         logger.info(f"Starting vector search for query: '{query[:50]}...'")
#
#         # Create query embedding
#         query_embedding = ollama.embed(model=config.EMBEDDING_MODEL, input=query)['embeddings'][0]
#
#         # Vector search pipeline
#         pipeline = [
#             {
#                 "$vectorSearch": {
#                     "index": "vector_index",
#                     "path": "embedding",
#                     "queryVector": query_embedding,
#                     "numCandidates": top_n * 5,
#                     "limit": top_n * 5
#                 }
#             },
#             {
#                 "$addFields": {
#                     "similarity": {"$meta": "vectorSearchScore"}
#                 }
#             },
#             {
#                 "$lookup": {
#                     "from": "customer-support-meta",  # Your metadata collection name
#                     "localField": "chunk_id",
#                     "foreignField": "chunk_id",
#                     "as": "metadata"
#                 }
#             },
#             {
#                 "$match": {"metadata": {"$ne": []}}
#             },
#             {
#                 "$project": {
#                     "chunk_id": 1,
#                     "similarity": 1,
#                     "text": {"$arrayElemAt": ["$metadata.text", 0]},
#                     "intent": {"$arrayElemAt": ["$metadata.intent", 0]},
#                     "category": {"$arrayElemAt": ["$metadata.category", 0]},
#                     "_id": 0
#                 }
#             },
#             {"$sort": {"similarity": -1}},
#             {"$limit": top_n}
#         ]
#
#         results = list(embeddings_collection.aggregate(pipeline))
#
#         return [
#             response_model.RetrievedChunk(
#                 chunk_id=doc['chunk_id'],
#                 text=doc['text'],
#                 similarity=doc['similarity'],
#                 intent=doc.get('intent'),
#                 category=doc.get('category')
#             )
#             for doc in results
#         ]
#
#     except Exception as e:
#         logger.error(f"Error during vector search: {e}")
#         raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")
#


async def benchmark_retrieval(query: str, top_n: int = 3):
    """Benchmark different parts of retrieval"""

    # Benchmark embedding creation
    start = time.time()
    query_embedding = ollama.embed(model=config.EMBEDDING_MODEL, input=query)['embeddings'][0]
    embedding_time = time.time() - start
    print(f"Embedding creation: {embedding_time:.2f}s")

    # Benchmark vector search only
    start = time.time()
    vector_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 10,
                "limit": top_n
            }
        },
        {"$project": {"chunk_id": 1, "similarity": {"$meta": "vectorSearchScore"}}}
    ]
    vector_results = list(embeddings_collection.aggregate(vector_pipeline))
    vector_time = time.time() - start
    print(f"Vector search: {vector_time:.2f}s")

    # Benchmark metadata lookup
    start = time.time()
    chunk_ids = [doc['chunk_id'] for doc in vector_results]
    metadata_docs = list(meta_collection.find({"chunk_id": {"$in": chunk_ids}}))
    lookup_time = time.time() - start
    print(f"Metadata lookup: {lookup_time:.2f}s")

    print(f"Total estimated: {embedding_time + vector_time + lookup_time:.2f}s")

async def warm_up_model():
    try:
        retrieved_chunks = await retrieve_from_mongodb('hello', 3)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base"
            )

        # Generate response
        response_text = await main.generate_chat_response('hello', retrieved_chunks)
        logger.info("Model pre-warmed successfully")
    except Exception as e:
        logger.error(f"Model warm-up failed: {e}")

# MongoDB connection management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    await connect_to_mongodb()
    await warm_up_model()
    yield
    # Shutdown
    await close_mongodb_connection()


async def connect_to_mongodb():
    """Initialize MongoDB connection"""

    global client, db, meta_collection, embeddings_collection

    try:
        client = MongoClient(config.MONGO_URI, server_api=ServerApi('1'))
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