"""Upload endpoints: create, send chunks, complete, check status, list.

Handlers only parse the request, call the service layer, and shape the
response -- every rule (offset validation, size limits, binary detection,
state transitions) lives in app/services/upload_service.py. Domain errors
raised there (app/core/exceptions.py) are translated to HTTP responses by the
exception handlers registered in app/main.py, not here.

Every route requires an X-User-Id header (see app/pipeline/identity.py) and
scopes reads/writes to that caller -- one user cannot see, search, or delete
another user's files.
"""

from fastapi import APIRouter, Depends, Query, Request, status

from ...core import config
from ...repositories.file_repository import FileRecord
from ...schemas import upload as schemas
from ...schemas.common import ErrorResponse
from ...services.upload import upload_service
from ..deps import get_user_id

router = APIRouter(tags=["uploads"])


def _to_status(record: FileRecord) -> schemas.FileStatus:
    settled = record.chunks_indexed + record.chunks_failed
    return schemas.FileStatus(
        file_id=record.file_id,
        owner_id=record.owner_id,
        filename=record.filename,
        total_size=record.total_size,
        bytes_received=record.bytes_received,
        upload_status=record.upload_status,
        # Clamped: total_size is a client-supplied hint that actual bytes can
        # exceed, and progress is reported as a fraction in [0, 1].
        upload_progress=min(record.bytes_received / record.total_size, 1.0)
        if record.total_size
        else 1.0,
        processing_status=record.processing_status,
        # Measured against passages discovered so far, so it climbs during the
        # upload instead of sitting at zero until the last byte lands.
        processing_progress=(settled / record.chunks_total) if record.chunks_total else 0.0,
        chunks_total=record.chunks_total,
        chunks_indexed=record.chunks_indexed,
        chunks_failed=record.chunks_failed,
        searchable=record.chunks_indexed > 0,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


@router.post(
    "/files",
    response_model=schemas.CreateUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new upload",
)
def create_upload(
    payload: schemas.CreateUploadRequest,
    user_id: str = Depends(get_user_id),
) -> schemas.CreateUploadResponse:
    """Reserve a file_id. Send the bytes with PUT /files/{file_id}/chunk."""
    file_id = upload_service.create_upload(user_id, payload.filename, payload.total_size)
    return schemas.CreateUploadResponse(
        file_id=file_id,
        filename=payload.filename,
        total_size=payload.total_size,
        chunk_size=config.MAX_CHUNK_BYTES,
        upload_status="pending",
    )


@router.put(
    "/files/{file_id}/chunk",
    response_model=schemas.ChunkUploadResponse,
    summary="Upload one chunk at a byte offset",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown file"},
        409: {"model": ErrorResponse, "description": "Offset mismatch"},
        413: {"model": ErrorResponse, "description": "Chunk too large"},
        415: {
            "model": ErrorResponse,
            "description": "Content looks like binary, not text",
        },
        503: {
            "model": ErrorResponse,
            "description": "Too many uploads in progress; retry shortly",
        },
    },
)
async def upload_chunk(
    file_id: str,
    request: Request,
    offset: int = Query(
        ...,
        ge=0,
        description="Byte offset this chunk starts at. Must equal the server's "
        "current bytes_received.",
    ),
    user_id: str = Depends(get_user_id),
) -> schemas.ChunkUploadResponse:
    """Append one chunk.

    A fixed number of concurrent chunk uploads may hold their body in memory
    at once (see app.services.upload_service.acquire_upload_slot) -- past that
    limit this raises TooManyConcurrentUploads, which app/main.py maps to 503
    with Retry-After.

    The body is read via the raw Request rather than a typed `Body(...)`
    parameter deliberately: FastAPI 0.115's request-body dispatch still
    attempts to JSON-decode a scalar `bytes = Body(media_type=...)` parameter
    before honoring its declared media type, so it 422s on real clients whose
    default Content-Type isn't exactly what FastAPI expects (curl's
    --data-binary defaults to application/x-www-form-urlencoded; httpx sends
    none) -- i.e. it fails on the very clients this endpoint has to accept
    bytes from. Reading the raw body sidesteps that dispatch entirely; the
    OpenAPI schema for this endpoint is filled in by hand in app/main.py
    instead, so Swagger's "Try it out" still shows a working upload field.
    """
    with upload_service.acquire_upload_slot():
        body = await request.body()
        record = upload_service.append_chunk(file_id, user_id, offset, body)

    return schemas.ChunkUploadResponse(
        file_id=file_id,
        bytes_received=record.bytes_received,
        total_size=record.total_size,
        upload_status=record.upload_status,
        chunks_enqueued=record.next_sequence,
    )


@router.post(
    "/files/{file_id}/complete",
    response_model=schemas.FileStatus,
    summary="Mark an upload finished",
    responses={
        404: {"model": ErrorResponse, "description": "Unknown file"},
        409: {"model": ErrorResponse, "description": "No bytes received yet"},
    },
)
def complete_upload(file_id: str, user_id: str = Depends(get_user_id)) -> schemas.FileStatus:
    """Finalize the upload once the client has sent every chunk.

    This, not a byte count, is what marks the upload done -- see
    app.services.upload_service.complete_upload for why total_size alone
    can't be trusted as the completion signal.
    """
    record = upload_service.complete_upload(file_id, user_id)
    return _to_status(record)


@router.get(
    "/files/{file_id}/status",
    response_model=schemas.FileStatus,
    summary="Upload and processing progress",
)
def get_status(file_id: str, user_id: str = Depends(get_user_id)) -> schemas.FileStatus:
    """Report both states. Also doubles as the resume endpoint: send the next
    chunk from `bytes_received`."""
    record = upload_service.get_status(file_id, user_id)
    return _to_status(record)


@router.get(
    "/files",
    response_model=schemas.FileListResponse,
    summary="List your uploads",
)
def list_files(
    limit: int = Query(default=100, ge=1, le=1000),
    user_id: str = Depends(get_user_id),
) -> schemas.FileListResponse:
    """Files owned by the caller, not every file on the service."""
    records = upload_service.list_uploads(user_id, limit)
    return schemas.FileListResponse(files=[_to_status(r) for r in records])


@router.delete(
    "/files/{file_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a file and its index",
)
def delete_file(file_id: str, user_id: str = Depends(get_user_id)) -> None:
    upload_service.delete_upload(file_id, user_id)
