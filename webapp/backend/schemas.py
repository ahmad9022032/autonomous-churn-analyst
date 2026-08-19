"""Request/response models for the ChurnSight API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)


class WhatIfRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=32)
    changes: dict[str, Any] = Field(min_length=1)


class HypotheticalRequest(BaseModel):
    attributes: dict[str, Any] = Field(min_length=1)
