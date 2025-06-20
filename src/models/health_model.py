from pydantic import BaseModel


class HealthStatus(BaseModel):
    status: str
    mongodb_connected: bool
    ollama_available: bool
    total_chunks: int
    embedding_model: str
    language_model: str