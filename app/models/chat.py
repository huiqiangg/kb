from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class KnowledgeRange(BaseModel):
    knowledge_base_id: str = Field(min_length=1)
    doc_range: list[str] = Field(default_factory=list)

    @field_validator("doc_range", mode="before")
    @classmethod
    def accept_single_document_id(cls, value: object) -> list[str]:
        return [value] if isinstance(value, str) else value or []


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    ranges: list[KnowledgeRange] = Field(min_length=1)
