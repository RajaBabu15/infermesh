"""OpenAI-compatible response models used by the gateway's own endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "infermesh"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
