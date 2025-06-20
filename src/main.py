import datetime

import logging
from typing import List, Optional

import ollama
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config
# from src import .response_model
from src import mongodb
from src.models import response_model, health_model, query_model, database_model
# from src.mongodb import retrieve_from_mongodb, client, meta_collection, embeddings_collection, lifespan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Support RAG API",
    description="A RAG-powered customer support chatbot API using MongoDB and Ollama",
    version="1.0.0",
    lifespan=mongodb.lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello World"}

# @app.get("/", tags=["Root"])
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Customer Support RAG API",
#         "version": "1.0.0",
#         "endpoints": {
#             "/chat": "POST - Chat with the RAG system",
#             "/retrieve": "POST - Just retrieve similar chunks",
#             "/health": "GET - Health check",
#             "/stats": "GET - Database statistics"
#         }
#     }


@app.post("/chat", response_model=response_model.ChatResponse, tags=["Chat"])
async def chat_endpoint(request: query_model.QueryRequest):
    """
    Main chat endpoint that retrieves relevant context and generates a response
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)

    try:
        # Retrieve relevant chunks
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base"
            )

        # Generate response
        response_text = await generate_chat_response(request.query, retrieved_chunks)

        # Calculate processing time
        processing_time = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        # Filter metadata if requested
        if not request.include_metadata:
            for chunk in retrieved_chunks:
                chunk.intent = None
                chunk.category = None

        return response_model.ChatResponse(
            query=request.query,
            response=response_text,
            retrieved_chunks=retrieved_chunks,
            processing_time=processing_time,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/retrieve", response_model=List[response_model.RetrievedChunk], tags=["Retrieval"])
async def retrieve_endpoint(request: query_model.QueryRequest):
    """
    Retrieve similar chunks without generating a chat response
    """
    try:
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base"
            )

        # Filter metadata if requested
        if not request.include_metadata:
            for chunk in retrieved_chunks:
                chunk.intent = None
                chunk.category = None

        return retrieved_chunks

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in retrieve endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health", response_model=health_model.HealthStatus, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    """
    try:
        # Check MongoDB
        mongodb_connected = False
        total_chunks = 0
        try:
            mongodb.client.admin.command('ping')
            mongodb_connected = True
            total_chunks = mongodb.meta_collection.count_documents({})
        except:
            pass

        # Check Ollama
        ollama_available = False
        try:
            ollama.list()  # Simple test to see if Ollama is responsive
            ollama_available = True
        except:
            pass

        status = "healthy" if mongodb_connected and ollama_available else "degraded"

        return health_model.HealthStatus(
            status=status,
            mongodb_connected=mongodb_connected,
            ollama_available=ollama_available,
            total_chunks=total_chunks,
            embedding_model=config.EMBEDDING_MODEL,
            language_model=config.LANGUAGE_MODEL
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/stats", response_model=database_model.DatabaseStats, tags=["Statistics"])
async def get_database_stats():
    """
    Get database statistics
    """
    try:
        meta_count = mongodb.meta_collection.count_documents({})
        embeddings_count = mongodb.embeddings_collection.count_documents({})

        # Get intent distribution
        intent_pipeline = [
            {"$group": {"_id": "$intent", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10}
        ]
        top_intents = list(mongodb.meta_collection.aggregate(intent_pipeline))

        # Get category distribution
        category_pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        categories = list(mongodb.meta_collection.aggregate(category_pipeline))

        return database_model.DatabaseStats(
            total_chunks=meta_count,
            total_embeddings=embeddings_count,
            top_intents=top_intents,
            categories=categories,
            last_updated=datetime.datetime.now(datetime.timezone.utc)
        )

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database statistics")


# Additional utility endpoints
@app.get("/search", tags=["Search"])
async def search_by_intent_or_category(
        intent: Optional[str] = Query(None, description="Filter by intent"),
        category: Optional[str] = Query(None, description="Filter by category"),
        limit: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    """
    Search chunks by intent or category
    """
    try:
        query_filter = {}
        if intent:
            query_filter["intent"] = intent
        if category:
            query_filter["category"] = category

        results = list(mongodb.meta_collection.find(query_filter).limit(limit))

        # Convert ObjectId to string for JSON serialization
        for result in results:
            result["_id"] = str(result["_id"])

        return {
            "filter": query_filter,
            "count": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


async def generate_chat_response(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> str:
    """Generate chat response using Ollama"""
    try:
        # Prepare context
        context_chunks = [chunk.text for chunk in retrieved_chunks]

        instruction_prompt = f'''You are a helpful customer support chatbot.
Use only the following pieces of context to answer the customer's question. The context contains previous customer queries and support responses.
Don't make up any new information - only use what's provided:

{chr(10).join([f' - {chunk}' for chunk in context_chunks])}

Provide a helpful, professional response based on this context.'''

        # Generate response
        response = ollama.chat(
            model=config.LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': query},
            ],
            stream=False,  # For API, we don't stream
        )

        return response['message']['content']

    except Exception as e:

        logger.error(f"Error generating chat response: {e}")
        # raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=retrieved_chunks)

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "src.main:app",  # Adjust if your file is named differently
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )