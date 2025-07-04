import datetime
import logging
import time
import asyncio
import json
from functools import wraps
from typing import List, Optional, AsyncGenerator

import aiohttp
import ollama
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src import config
from src import mongodb
from src.models import response_model, health_model, query_model, database_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global clients
ollama_client = ollama.Client(host='http://127.0.0.1:11434')
http_session = None

app = FastAPI(
    title="Customer Support RAG API",
    description="A RAG-powered customer support chatbot API using MongoDB and Ollama",
    version="1.0.0",
    lifespan=mongodb.lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_http_session():
    """Get or create HTTP session"""
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()
    return http_session


def timing_decorator(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f}s")
        return result

    return wrapper


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/chat", response_model=response_model.ChatResponse, tags=["Chat"])
async def chat_endpoint(request: query_model.QueryRequest):
    """
    Main chat endpoint that retrieves relevant context and generates a response
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)

    try:
        # Run retrieval and response generation concurrently where possible
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base"
            )

        # Generate response with optimized async call
        response_text = await generate_chat_response_async(request.query, retrieved_chunks)

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


@timing_decorator
async def generate_chat_response_async(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> str:
    """Generate chat response using async HTTP calls to Ollama"""
    try:
        # Prepare context (optimized)
        context_chunks = [chunk.text for chunk in retrieved_chunks]
        max_context_length = 2000

        # More efficient string joining
        context_text = '\n - '.join(context_chunks)
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."

        # Optimized prompt
        system_prompt = f"""You are a helpful customer support chatbot. Use only the following context to answer the customer's question:

{context_text}

Provide a helpful, professional response based on this context."""

        # Async HTTP request to Ollama
        session = await get_http_session()

        payload = {
            "model": config.LANGUAGE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False,
            "keep_alive": "5m",  # Keep model loaded longer
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512  # Limit response length for speed
            }
        }

        async with session.post(
                "http://127.0.0.1:11434/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["message"]["content"]
            else:
                error_text = await response.text()
                raise HTTPException(
                    status_code=500,
                    detail=f"Ollama API error: {response.status} - {error_text}"
                )

    except asyncio.TimeoutError:
        logger.error("Ollama request timed out")
        raise HTTPException(status_code=504, detail="Response generation timed out")
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")


# Fallback sync version for compatibility
@timing_decorator
async def generate_chat_response(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> str:
    """Original sync version - kept for backward compatibility"""
    try:
        # Use thread pool for sync ollama calls
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _generate_sync_response, query, retrieved_chunks)
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        raise HTTPException(status_code=500, detail="Chat generation failed")


def _generate_sync_response(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> str:
    """Synchronous response generation"""
    context_chunks = [chunk.text for chunk in retrieved_chunks]
    max_context_length = 2000
    context_text = '\n - '.join(context_chunks)

    if len(context_text) > max_context_length:
        context_text = context_text[:max_context_length] + "..."

    instruction_prompt = f'''You are a helpful customer support chatbot.
Use only the following pieces of context to answer the customer's question:

{context_text}

Provide a helpful, professional response based on this context.'''

    response = ollama_client.chat(
        model=config.LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': query},
        ],
        stream=False,
        keep_alive="5m",
        options={
            "temperature": 0.7,
            "num_predict": 512
        }
    )

    return response['message']['content']


@app.get("/health", response_model=health_model.HealthStatus, tags=["Health"])
async def health_check():
    """Health check endpoint"""
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
            ollama.list()
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
    """Get database statistics"""
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


@app.get("/search", tags=["Search"])
async def search_by_intent_or_category(
        intent: Optional[str] = Query(None, description="Filter by intent"),
        category: Optional[str] = Query(None, description="Filter by category"),
        limit: int = Query(10, ge=1, le=100, description="Number of results to return")
):
    """Search chunks by intent or category"""
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


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream_endpoint(request: query_model.QueryRequest):
    """
    Streaming chat endpoint for real-time response generation
    """
    try:
        # First, retrieve relevant chunks
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            # Return error as SSE event
            async def error_stream():
                yield f"data: {json.dumps({'error': 'No relevant information found'})}\n\n"

            return StreamingResponse(error_stream(), media_type="text/plain")

        # Stream the response
        return StreamingResponse(
            stream_chat_response(request.query, retrieved_chunks),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {e}")

        async def error_stream():
            yield f"data: {json.dumps({'error': 'Internal server error'})}\n\n"

        return StreamingResponse(error_stream(), media_type="text/plain")


async def stream_chat_response(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> AsyncGenerator[
    str, None]:
    """
    Generate streaming chat response using async HTTP calls to Ollama
    """
    try:
        # Prepare context
        context_chunks = [chunk.text for chunk in retrieved_chunks]
        max_context_length = 2000

        context_text = '\n - '.join(context_chunks)
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."

        system_prompt = f"""You are a helpful customer support chatbot. Use only the following context to answer the customer's question:

{context_text}

Provide a helpful, professional response based on this context."""

        # Send initial metadata
        metadata = {
            "type": "metadata",
            "retrieved_chunks": len(retrieved_chunks),
            "query": query[:100] + "..." if len(query) > 100 else query
        }
        yield f"data: {json.dumps(metadata)}\n\n"

        # Prepare streaming request to Ollama
        session = await get_http_session()

        payload = {
            "model": config.LANGUAGE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": True,  # Enable streaming
            "keep_alive": "5m",
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512
            }
        }

        # Stream response from Ollama
        async with session.post(
                "http://127.0.0.1:11434/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
        ) as response:

            if response.status != 200:
                error_text = await response.text()
                error_data = {
                    "type": "error",
                    "message": f"Ollama API error: {response.status} - {error_text}"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # Process streaming response
            full_response = ""
            async for line in response.content:
                line = line.decode('utf-8').strip()

                if not line:
                    continue

                try:
                    # Parse JSON response from Ollama
                    chunk_data = json.loads(line)

                    if "message" in chunk_data and "content" in chunk_data["message"]:
                        content = chunk_data["message"]["content"]
                        full_response += content

                        # Send content chunk
                        chunk_response = {
                            "type": "content",
                            "content": content,
                            "full_response": full_response
                        }
                        yield f"data: {json.dumps(chunk_response)}\n\n"

                    # Check if generation is complete
                    if chunk_data.get("done", False):
                        # Send completion message
                        completion_data = {
                            "type": "complete",
                            "full_response": full_response,
                            "total_duration": chunk_data.get("total_duration", 0),
                            "load_duration": chunk_data.get("load_duration", 0),
                            "prompt_eval_count": chunk_data.get("prompt_eval_count", 0),
                            "eval_count": chunk_data.get("eval_count", 0)
                        }
                        yield f"data: {json.dumps(completion_data)}\n\n"
                        break

                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                except Exception as e:
                    logger.error(f"Error processing stream chunk: {e}")
                    continue

    except asyncio.TimeoutError:
        error_data = {
            "type": "error",
            "message": "Response generation timed out"
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    except Exception as e:
        logger.error(f"Error in stream_chat_response: {e}")
        error_data = {
            "type": "error",
            "message": f"Streaming error: {str(e)}"
        }
        yield f"data: {json.dumps(error_data)}\n\n"


# Alternative: Simple text streaming version
async def stream_chat_response_simple(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> \
AsyncGenerator[str, None]:
    """
    Simplified streaming version that just streams text content
    """
    try:
        context_chunks = [chunk.text for chunk in retrieved_chunks]
        max_context_length = 2000

        context_text = '\n - '.join(context_chunks)
        if len(context_text) > max_context_length:
            context_text = context_text[:max_context_length] + "..."

        system_prompt = f"""You are a helpful customer support chatbot. Use only the following context to answer the customer's question:

{context_text}

Provide a helpful, professional response based on this context."""

        session = await get_http_session()

        payload = {
            "model": config.LANGUAGE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": True,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": 512
            }
        }

        async with session.post(
                "http://127.0.0.1:11434/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
        ) as response:

            if response.status != 200:
                yield f"Error: Failed to get response from AI model\n"
                return

            async for line in response.content:
                line = line.decode('utf-8').strip()

                if not line:
                    continue

                try:
                    chunk_data = json.loads(line)

                    if "message" in chunk_data and "content" in chunk_data["message"]:
                        content = chunk_data["message"]["content"]
                        yield content

                    if chunk_data.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing stream chunk: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error in simple stream: {e}")
        yield f"Error: {str(e)}\n"


@app.post("/chat/stream-simple", tags=["Chat"])
async def chat_stream_simple_endpoint(request: query_model.QueryRequest):
    """
    Simple streaming endpoint that just returns text content
    """
    try:
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            async def error_stream():
                yield "Error: No relevant information found in the knowledge base.\n"

            return StreamingResponse(error_stream(), media_type="text/plain")

        return StreamingResponse(
            stream_chat_response_simple(request.query, retrieved_chunks),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Error in simple streaming endpoint: {e}")

        async def error_stream():
            yield f"Error: Internal server error\n"

        return StreamingResponse(error_stream(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 3000))

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
# Cleanup on shutdown
async def cleanup():
    global http_session
    if http_session:
        await http_session.close()

app.add_event_handler("shutdown", cleanup)