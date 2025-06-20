from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel


class DatabaseStats(BaseModel):
    total_chunks: int
    total_embeddings: int
    top_intents: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    last_updated: Optional[datetime] = None