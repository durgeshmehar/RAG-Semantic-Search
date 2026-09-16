"""Request and response shapes for the search endpoint."""

from pydantic import BaseModel, Field

from ..core import config


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Natural-language query. Matching is by meaning, not "
        "keyword overlap.",
        examples=["database connectivity problems"],
    )
    top_k: int = Field(
        default=config.DEFAULT_TOP_K,
        ge=1,
        le=config.MAX_TOP_K,
        description="How many sections to return.",
    )


class SearchHit(BaseModel):
    text: str = Field(..., description="The matching section, read from the file.")
    start_byte: int
    end_byte: int
    score: float = Field(
        ..., description="Cosine similarity in [-1, 1]; higher is closer."
    )
    sequence: int = Field(..., description="Position of this passage in the file.")


class SearchResponse(BaseModel):
    file_id: str
    query: str
    total_hits: int
    results: list[SearchHit]
