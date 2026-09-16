"""Maps domain exceptions (app/core/exceptions.py) to HTTP responses.

Domain exceptions are raised by the service layer with no knowledge of HTTP;
this is the single place that maps each one to a status code and response
body, instead of every raise site in every router constructing its own
HTTPException.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from .exceptions import (
    ChunkTooLarge,
    FileNotFound,
    FileTooLarge,
    NoBytesReceivedYet,
    NotTextFile,
    NothingIndexedYet,
    OffsetMismatch,
    TooManyConcurrentUploads,
    UploadAlreadyDone,
)


def _error_response(status_code: int, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(FileNotFound)
    def _handle_file_not_found(request: Request, exc: FileNotFound) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, str(exc))

    @app.exception_handler(UploadAlreadyDone)
    def _handle_upload_already_done(request: Request, exc: UploadAlreadyDone) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(OffsetMismatch)
    def _handle_offset_mismatch(request: Request, exc: OffsetMismatch) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(NoBytesReceivedYet)
    def _handle_no_bytes_received(request: Request, exc: NoBytesReceivedYet) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(ChunkTooLarge)
    def _handle_chunk_too_large(request: Request, exc: ChunkTooLarge) -> JSONResponse:
        return _error_response(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, str(exc))

    @app.exception_handler(FileTooLarge)
    def _handle_file_too_large(request: Request, exc: FileTooLarge) -> JSONResponse:
        return _error_response(status.HTTP_400_BAD_REQUEST, str(exc))

    @app.exception_handler(NotTextFile)
    def _handle_not_text_file(request: Request, exc: NotTextFile) -> JSONResponse:
        return _error_response(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc))

    @app.exception_handler(NothingIndexedYet)
    def _handle_nothing_indexed(request: Request, exc: NothingIndexedYet) -> JSONResponse:
        return _error_response(status.HTTP_409_CONFLICT, str(exc))

    @app.exception_handler(TooManyConcurrentUploads)
    def _handle_too_many_uploads(request: Request, exc: TooManyConcurrentUploads) -> JSONResponse:
        return _error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            str(exc),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )
