"""Schemas shared across endpoints."""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
