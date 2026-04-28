from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

class RequestInput(BaseModel):
    user_query: Optional[str] = Field(None, description= "User Request sent to the llm")

class RequestOutput(BaseModel):
    response: Optional[str] = Field(None, description="LLM response to the user")

class SearchSchema(BaseModel):
    query: Optional[str]