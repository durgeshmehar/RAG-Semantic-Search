"""Business rules for creating, appending to, completing, and deleting uploads.

The orchestrator for everything upload-related, and the only module
api/v1/upload.py imports -- validation (upload/validators.py) and the
concurrent-upload cap (upload/concurrency.py) are this service's supporting
files, kept in a sibling folder because they're each a distinct rule rather
than orchestration, but callers outside this file go through upload_service,
not around it. Every service lives flat at the top of services/ like this
one; only a service's extra supporting files go in a same-named subfolder.

Every rule that used to live inline in the router is here: offset validation,
size limits, binary-content rejection, the disk/DB reconciliation on a
resumed upload, and the state transitions an upload moves through. The router
calls these functions and translates the exceptions raised
(app/core/exceptions.py) into HTTP responses; nothing here imports FastAPI or
raises HTTPException, so these rules are callable and testable with no HTTP
framework involved.
"""

import time
import uuid

from .upload import validators
from .upload.concurrency import acquire as acquire_upload_slot
from .. import storage
from ..core import config
from ..core.exceptions import (
    ChunkTooLarge,
    FileTooLarge,
    NoBytesReceivedYet,
    NotTextFile,
    OffsetMismatch,
    UploadAlreadyDone,
)
from ..db import db
from ..rag.ingestion.chunker import LineBuffer
from ..repositories import file_repository, vector_repository
from ..repositories.file_repository import FileRecord
from ..tasks import job_queue

__all__ = [
    "acquire_upload_slot",
    "create_upload",
    "append_chunk",
    "complete_upload",
    "get_status",
    "list_uploads",
    "delete_upload",
]


def create_upload(owner_id: str, filename: str, total_size: int) -> str:
    """Reserve a file_id for a new upload. Returns that id.

    `total_size` is the client's stated size, used only to size passages and
    report progress -- it is a hint, not what decides completion (see
    complete_upload). A client that mis-declares it must not leave the
    upload stuck waiting for bytes that will never arrive.
    """
    file_id = uuid.uuid4().hex
    with db.transaction() as conn:
        file_repository.create(conn, file_id, owner_id, filename, total_size)
    return file_id


def append_chunk(file_id: str, owner_id: str, offset: int, body: bytes) -> FileRecord:
    """Append one chunk, enqueue its passages, and return the updated record.

    The read-check-append-write sequence runs inside one SQLite transaction so
    two concurrent requests for the same file (a client retry racing the
    original attempt) can't both pass the offset check against the same
    starting value: BEGIN IMMEDIATE takes the database's write lock up front,
    so the second request blocks until the first commits, then sees the
    already-advanced bytes_received and correctly fails its own check rather
    than double-appending.
    """
    if len(body) > config.MAX_CHUNK_BYTES:
        raise ChunkTooLarge(len(body), config.MAX_CHUNK_BYTES)

    with db.transaction() as conn:
        record = file_repository.get_owned(conn, file_id, owner_id)

        if record.upload_status in ("completed", "finalizing"):
            raise UploadAlreadyDone(record.upload_status)

        expected = record.bytes_received
        if offset != expected:
            # Tell the client exactly where to resume rather than just refusing.
            raise OffsetMismatch(expected=expected, got=offset)

        # total_size is a hint, not a hard ceiling -- a client may legitimately
        # exceed its own earlier estimate. MAX_FILE_BYTES is the real cap.
        if expected + len(body) > config.MAX_FILE_BYTES:
            raise FileTooLarge(config.MAX_FILE_BYTES)

        # Checked only on the first chunk of a fresh upload: nothing on disk or
        # in the queue yet to unwind, and binary content reveals itself in the
        # first few KB (a PDF header, a PNG signature, a zip's local file
        # header). Rejecting here avoids indexing a file that can only ever
        # return decode-noise from search.
        if offset == 0 and validators.looks_like_binary(body):
            raise NotTextFile()

        # The file on disk is the source of truth for how much we really have.
        # If a previous request died between the write and the commit, the
        # file may be ahead of the database; trim it back so appends stay
        # aligned. Still inside the transaction: another request cannot have
        # advanced bytes_received without first taking this same write lock.
        on_disk = storage.current_size(file_id)
        if on_disk != expected:
            storage.truncate_to(file_id, expected)

        storage.append_chunk(file_id, body)
        new_size = expected + len(body)

        # Split into passages from the bytes already in memory -- the uploaded
        # file is never re-read to build the index. Passage size is chosen
        # from the declared total_size purely to pick a sensible target; it
        # has no bearing on when the upload is considered done.
        target, maximum, overlap = config.passage_size_for(record.total_size)
        buffer = LineBuffer(
            start_offset=expected,
            pending_tail=record.pending_tail,
            target_bytes=target,
            max_bytes=maximum,
            overlap_bytes=overlap,
        )
        passages = buffer.feed(body)

        next_sequence = job_queue.enqueue(conn, file_id, passages, record.next_sequence)
        file_repository.update_after_chunk(
            conn,
            file_id,
            bytes_received=new_size,
            chunks_added=len(passages),
            next_sequence=next_sequence,
            pending_tail=buffer.pending_tail,
        )

        return file_repository.get(conn, file_id)


def complete_upload(file_id: str, owner_id: str) -> FileRecord:
    """Finalize an upload once the client has sent every chunk.

    Completion is an explicit client action rather than inferred from
    `bytes_received >= total_size`: `total_size` is client-declared and
    unverified, so trusting it as the sole completion signal means a client
    that over-states its file's size leaves the upload stuck in `uploading`
    forever, with no bytes left to send and no way to finish.

    The `finalizing` status closes the same race append_chunk guards against:
    it is set inside this transaction before the rename, so a concurrent
    chunk append sees it and is rejected rather than appending to a file this
    call is mid-way through renaming.
    """
    with db.transaction() as conn:
        record = file_repository.get_owned(conn, file_id, owner_id)

        if record.upload_status == "completed":
            return record

        if record.bytes_received == 0:
            raise NoBytesReceivedYet()

        # Flush whatever's left in the line buffer -- the final line usually
        # has no trailing newline, so it would otherwise never be emitted.
        buffer = LineBuffer(start_offset=record.bytes_received, pending_tail=record.pending_tail)
        passages = buffer.flush()
        next_sequence = job_queue.enqueue(conn, file_id, passages, record.next_sequence)

        file_repository.mark_finalizing(
            conn,
            file_id,
            chunks_added=len(passages),
            next_sequence=next_sequence,
            total_size=record.bytes_received,
        )

    # The rename happens outside the transaction (it's a filesystem call, not
    # a database one) but after upload_status is already 'finalizing', so a
    # concurrent chunk append sees that status and is rejected before it can
    # touch a file this call is in the middle of renaming.
    storage.finalize(file_id)

    with db.transaction() as conn:
        file_repository.mark_completed(conn, file_id)
        return file_repository.get(conn, file_id)


def get_status(file_id: str, owner_id: str) -> FileRecord:
    conn = db.get_connection()
    return file_repository.get_owned(conn, file_id, owner_id)


def list_uploads(owner_id: str, limit: int) -> list[FileRecord]:
    conn = db.get_connection()
    return file_repository.list_owned(conn, owner_id, limit)


def delete_upload(file_id: str, owner_id: str) -> None:
    with db.transaction() as conn:
        file_repository.get_owned(conn, file_id, owner_id)  # raises FileNotFound if not owned
        file_repository.delete(conn, file_id)
    storage.delete_all(file_id)
    vector_repository.drop(file_id)
