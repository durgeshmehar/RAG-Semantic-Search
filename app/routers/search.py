"""Semantic search over an uploaded file.

The handler only parses the request, calls the service, and shapes the
response -- the actual matching (embed query, ask the vector store, resolve
byte ranges to text) lives in app/services/search_service.py.
"""

from fastapi import APIRouter, Depends

from .. import models
from ..core.identity import get_user_id
from ..services import search_service

router = APIRouter(tags=["search"])


@router.post(
    "/files/{file_id}/search",
    response_model=models.SearchResponse,
    summary="Search a file in natural language",
    responses={
        404: {"model": models.ErrorResponse, "description": "Unknown file"},
        409: {"model": models.ErrorResponse, "description": "Nothing indexed yet"},
    },
)
def search_file(
    file_id: str,
    payload: models.SearchRequest,
    user_id: str = Depends(get_user_id),
) -> models.SearchResponse:
    """Return the sections whose meaning is closest to the query.

    Searching is allowed while the upload is still in progress -- whatever has
    been indexed so far is queryable.
    """
    hits = search_service.search(file_id, user_id, payload.query, payload.top_k)

    return models.SearchResponse(
        file_id=file_id,
        query=payload.query,
        total_hits=len(hits),
        results=[
            models.SearchHit(
                text=h.text,
                start_byte=h.start_byte,
                end_byte=h.end_byte,
                score=h.score,
                sequence=h.sequence,
            )
            for h in hits
        ],
    )
