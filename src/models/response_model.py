from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    similarity: float
    intent: Optional[str] = None
    category: Optional[str] = None

class ChatResponse(BaseModel):
    query: str
    response: str
    retrieved_chunks: List[RetrievedChunk]
    processing_time: float
    timestamp: datetime