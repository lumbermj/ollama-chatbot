from typing import Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question or query")
    top_n: Optional[int] = Field(default=3, ge=1, le=10, description="Number of similar chunks to retrieve")
    # session_id: str = Field(..., description="The user's session_id")
    include_metadata: Optional[bool] = Field(default=True, description="Include intent and category metadata")