import datetime
import logging
from typing import List, Optional, Dict, Any
import uuid

import ollama
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src import mongodb
from src.models import response_model, health_model, query_model, database_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory conversation storage (for production, consider Redis or database)
conversation_store: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="Customer Support RAG API",
    description="A RAG-powered customer support chatbot API with conversation memory using MongoDB and Ollama",
    version="1.1.0",
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


# Enhanced models for conversation context
class ConversationExchange:
    def __init__(self, question: str, answer: str, timestamp: datetime.datetime = None):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp or datetime.datetime.now(datetime.timezone.utc)


class ConversationContext:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[ConversationExchange] = []
        self.customer_context: Dict[str, Any] = {}
        self.created_at = datetime.datetime.now(datetime.timezone.utc)
        self.last_accessed = datetime.datetime.now(datetime.timezone.utc)
        self.max_history_length = 5

    def add_exchange(self, question: str, answer: str):
        """Add a Q&A exchange to conversation history"""
        self.conversation_history.append(ConversationExchange(question, answer))
        self.last_accessed = datetime.datetime.now(datetime.timezone.utc)

        # Keep only recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def update_customer_context(self, updates: Dict[str, Any]):
        """Update customer-specific context"""
        self.customer_context.update(updates)
        self.last_accessed = datetime.datetime.now(datetime.timezone.utc)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        return {
            "session_id": self.session_id,
            "total_exchanges": len(self.conversation_history),
            "customer_context": self.customer_context,
            "last_accessed": self.last_accessed.isoformat(),
            "recent_questions": [ex.question for ex in self.conversation_history[-3:]]
        }


def get_or_create_conversation(session_id: str) -> ConversationContext:
    """Get existing conversation or create new one"""
    if session_id not in conversation_store:
        conversation_store[session_id] = ConversationContext(session_id)

    conversation_store[session_id].last_accessed = datetime.datetime.now(datetime.timezone.utc)
    return conversation_store[session_id]


def cleanup_old_conversations():
    """Clean up conversations older than 24 hours"""
    cutoff_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
    expired_sessions = [
        session_id for session_id, conv in conversation_store.items()
        if conv.last_accessed < cutoff_time
    ]
    for session_id in expired_sessions:
        del conversation_store[session_id]

    if expired_sessions:
        logger.info(f"Cleaned up {len(expired_sessions)} expired conversation sessions")


def build_contextual_prompt(query: str, context_chunks: List[str], conversation: ConversationContext) -> str:
    """Build enhanced prompt with conversation context"""

    # Format conversation history
    conversation_context = ""
    if conversation.conversation_history:
        conversation_context = "\n\nCONVERSATION HISTORY (for context):\n"
        # Use last 3 exchanges to keep prompt manageable
        recent_history = conversation.conversation_history[-3:]
        for i, exchange in enumerate(recent_history, 1):
            conversation_context += f"Previous Q{i}: {exchange.question}\n"
            # Truncate long answers to keep prompt size reasonable
            answer_preview = exchange.answer[:200] + "..." if len(exchange.answer) > 200 else exchange.answer
            conversation_context += f"Previous A{i}: {answer_preview}\n"

    # Format customer context
    customer_info = ""
    if conversation.customer_context:
        customer_info = f"\n\nCUSTOMER CONTEXT:\n"
        for key, value in conversation.customer_context.items():
            customer_info += f"- {key}: {value}\n"

    instruction_prompt = f'''You are a professional customer support assistant. Your role is to provide helpful, accurate, and contextually aware responses.

INSTRUCTIONS:
1. Use ONLY the provided knowledge base and conversation history to answer questions
2. If the current question relates to previous questions in the conversation, acknowledge the connection explicitly
3. Maintain consistency with your previous responses in this conversation
4. Reference previous parts of the conversation when relevant (e.g., "As I mentioned earlier...")
5. If you cannot answer based on the provided context, clearly state this limitation
6. Be concise but thorough, maintaining a professional and helpful tone
7. If this appears to be a follow-up question, provide context-aware responses

KNOWLEDGE BASE CONTEXT:
{chr(10).join([f'- {chunk}' for chunk in context_chunks])}
{conversation_context}
{customer_info}

CURRENT QUESTION: {query}

Provide a helpful response that considers both the knowledge base and conversation context. If this question builds on previous questions, make clear connections and provide a coherent answer.'''

    return instruction_prompt


@app.get("/")
def read_root():
    return {
        "message": "Enhanced Customer Support RAG API with Conversation Memory",
        "version": "1.1.0",
        "features": ["RAG", "Conversation Memory", "Context Awareness", "Session Management"]
    }


@app.post("/chat", response_model=response_model.ChatResponse, tags=["Chat"])
async def chat_endpoint(request: query_model.QueryRequest):
    """
    Enhanced chat endpoint with conversation memory and context awareness
    """
    start_time = datetime.datetime.now(datetime.timezone.utc)

    # Clean up old conversations periodically
    cleanup_old_conversations()

    try:
        # Get or create conversation session
        session_id = getattr(request, 'session_id', None) or str(uuid.uuid4())
        conversation = get_or_create_conversation(session_id)

        # Retrieve relevant chunks
        retrieved_chunks = await mongodb.retrieve_from_mongodb(request.query, request.top_n)

        if not retrieved_chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant information found in the knowledge base"
            )

        # Generate contextual response
        response_text = await generate_contextual_chat_response(
            request.query,
            retrieved_chunks,
            conversation
        )

        # Add exchange to conversation history
        conversation.add_exchange(request.query, response_text)

        # Calculate processing time
        processing_time = (datetime.datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        # Filter metadata if requested
        if not request.include_metadata:
            for chunk in retrieved_chunks:
                chunk.intent = None
                chunk.category = None

        # Enhanced response with conversation context
        response = response_model.ChatResponse(
            query=request.query,
            response=response_text,
            retrieved_chunks=retrieved_chunks,
            processing_time=processing_time,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
        )

        # Add session info to response (if your response model supports it)
        response.session_id = session_id
        response.conversation_context = {
            "total_exchanges": len(conversation.conversation_history),
            "has_context": len(conversation.conversation_history) > 1
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/chat/context", tags=["Chat"])
async def update_customer_context(
        session_id: str,
        context_updates: Dict[str, Any]
):
    """
    Update customer context for a conversation session
    """
    try:
        conversation = get_or_create_conversation(session_id)
        conversation.update_customer_context(context_updates)

        return {
            "message": "Customer context updated successfully",
            "session_id": session_id,
            "updated_context": conversation.customer_context
        }

    except Exception as e:
        logger.error(f"Error updating customer context: {e}")
        raise HTTPException(status_code=500, detail="Failed to update customer context")


@app.get("/chat/sessions/{session_id}", tags=["Chat"])
async def get_conversation_summary(session_id: str):
    """
    Get conversation summary for a session
    """
    try:
        if session_id not in conversation_store:
            raise HTTPException(status_code=404, detail="Conversation session not found")

        conversation = conversation_store[session_id]
        return conversation.get_conversation_summary()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation summary")


@app.delete("/chat/sessions/{session_id}", tags=["Chat"])
async def clear_conversation_session(session_id: str):
    """
    Clear/delete a conversation session
    """
    try:
        if session_id in conversation_store:
            del conversation_store[session_id]
            return {"message": f"Conversation session {session_id} cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation session not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation session: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation session")


@app.get("/chat/sessions", tags=["Chat"])
async def list_active_sessions():
    """
    List all active conversation sessions
    """
    try:
        cleanup_old_conversations()  # Clean up before listing

        sessions = []
        for session_id, conversation in conversation_store.items():
            sessions.append({
                "session_id": session_id,
                "total_exchanges": len(conversation.conversation_history),
                "last_accessed": conversation.last_accessed.isoformat(),
                "has_customer_context": bool(conversation.customer_context)
            })

        return {
            "active_sessions": len(sessions),
            "sessions": sessions
        }

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


# Keep all your existing endpoints unchanged
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

        health_response = health_model.HealthStatus(
            status=status,
            mongodb_connected=mongodb_connected,
            ollama_available=ollama_available,
            total_chunks=total_chunks,
            embedding_model=config.EMBEDDING_MODEL,
            language_model=config.LANGUAGE_MODEL
        )

        # Add conversation memory status
        health_response.active_conversations = len(conversation_store)

        return health_response

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

        stats = database_model.DatabaseStats(
            total_chunks=meta_count,
            total_embeddings=embeddings_count,
            top_intents=top_intents,
            categories=categories,
            last_updated=datetime.datetime.now(datetime.timezone.utc)
        )

        # Add conversation stats
        stats.active_conversations = len(conversation_store)
        total_exchanges = sum(len(conv.conversation_history) for conv in conversation_store.values())
        stats.total_conversation_exchanges = total_exchanges

        return stats

    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database statistics")


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


async def generate_contextual_chat_response(
        query: str,
        retrieved_chunks: List[response_model.RetrievedChunk],
        conversation: ConversationContext
) -> str:
    """Generate contextual chat response using Ollama with conversation memory"""
    try:
        # Prepare context
        context_chunks = [chunk.text for chunk in retrieved_chunks]

        # Build contextual prompt
        instruction_prompt = build_contextual_prompt(query, context_chunks, conversation)

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
        logger.error(f"Error generating contextual chat response: {e}")
        raise HTTPException(status_code=500, detail=f"Chat generation error: {str(e)}")


# Legacy function for backward compatibility
async def generate_chat_response(query: str, retrieved_chunks: List[response_model.RetrievedChunk]) -> str:
    """Legacy generate chat response function for backward compatibility"""
    # Create a temporary conversation context for non-contextual calls
    temp_conversation = ConversationContext("temp")
    return await generate_contextual_chat_response(query, retrieved_chunks, temp_conversation)


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