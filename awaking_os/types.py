"""Cross-package primitives."""

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentType(StrEnum):
    SEMANTIC = "semantic"
    BIOTIC = "biotic"
    EXECUTIVE = "executive"
    RESEARCH = "research"


class TokenBudget(BaseModel):
    max_input_tokens: int = Field(default=100_000, ge=1)
    max_output_tokens: int = Field(default=4_096, ge=1)
