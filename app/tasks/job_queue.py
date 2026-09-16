r"""Durable job queue built on the `chunks` and `files` tables.

Why SQLite rather than an in-memory queue: work already enqueued must survive a
process crash. A `queue.Queue` loses it. Why not SQS/Redis: they'd add a
service dependency to something that has to run locally, and SQLite is already
here holding the metadata.

The lifecycle of a chunk row:

    pending -> processing -> indexed
                         \-> pending (retry, under MAX_RETRIES)
                         \-> failed  (retries exhausted)

Rows left in `processing` by a crash are reset to `pending` on startup, so the
only cost of an unclean shutdown is re-embedding a handful of passages.

The interface here is deliberately small -- claim / complete / fail / recover --
so swapping in SQS or Redis at scale is a contained change.

This module owns queue *semantics* (batching, retry policy, keeping `files`
and `chunks` in sync within one transaction) and holds no raw SQL itself --
that lives in app.repositories.sql_repository (its `chunks` and `files`
members), the same split as everywhere else in the codebase.
"""

import time
from dataclasses import dataclass

from ..core import config
from ..db import db
from ..repositories import sql_repository


@dataclass(frozen=True)
class ChunkJob:
    """One passage awaiting embedding."""

    chunk_id: int
    file_id: str
    sequence: int
    start_byte: int
    end_byte: int
    retry_count: int


def enqueue(
    conn,
    file_id: str,
    passages: list,
    start_sequence: int,
) -> int:
    """Insert passage ranges as pending jobs. Returns the next free sequence.

    Called from the upload path, so it must stay cheap: one executemany, no
    text, no embedding.
    """
    if not passages:
        return start_sequence

    rows = [
        (start_sequence + offset, passage.start_byte, passage.end_byte)
        for offset, passage in enumerate(passages)
    ]
    sql_repository.chunks.insert_pending(conn, file_id, rows)
    return start_sequence + len(passages)


def claim_batch(limit: int | None = None) -> list[ChunkJob]:
    """Atomically take up to `limit` pending jobs for this worker.

    Grouped by file so a batch shares one Qdrant collection, and ordered by
    sequence so vector positions stay monotonic within a file.

    The SELECT and UPDATE run inside one IMMEDIATE transaction; without that,
    two workers could read the same rows and embed them twice.
    """
    limit = limit or config.CLAIM_BATCH_SIZE

    with db.transaction() as conn:
        # Pick the file with the oldest pending work, then take a run of its
        # chunks -- keeping a batch to a single index.
        file_id = sql_repository.chunks.oldest_pending_file(conn)
        if file_id is None:
            return []

        rows = sql_repository.chunks.select_pending_for_file(conn, file_id, limit)
        if not rows:
            return []

        now = time.time()
        sql_repository.chunks.mark_processing(conn, [row.chunk_id for row in rows])
        sql_repository.files.mark_processing_started(conn, file_id, now)

        return [
            ChunkJob(
                chunk_id=row.chunk_id,
                file_id=row.file_id,
                sequence=row.sequence,
                start_byte=row.start_byte,
                end_byte=row.end_byte,
                retry_count=row.retry_count,
            )
            for row in rows
        ]


def complete_batch(jobs: list[ChunkJob]) -> None:
    """Mark jobs indexed.

    Byte ranges are looked up by (file_id, sequence) rather than a stored
    vector position: Qdrant carries start_byte/end_byte as point payload, so a
    search result maps directly to a byte range without a join back here.
    """
    if not jobs:
        return

    now = time.time()
    with db.transaction() as conn:
        sql_repository.chunks.mark_indexed(conn, [job.chunk_id for job in jobs])

        file_id = jobs[0].file_id
        sql_repository.files.record_indexed(conn, file_id, count=len(jobs), now=now)
        _refresh_processing_status(conn, file_id, now)


def fail_batch(jobs: list[ChunkJob], error: str) -> None:
    """Return jobs to the queue, or mark them failed once retries run out.

    A permanently failed passage doesn't fail the file: the rest stays
    searchable, and the error is recorded for the status endpoint.
    """
    if not jobs:
        return

    now = time.time()
    truncated = error[:500]
    retryable = [j for j in jobs if j.retry_count + 1 < config.MAX_RETRIES]
    exhausted = [j for j in jobs if j.retry_count + 1 >= config.MAX_RETRIES]

    with db.transaction() as conn:
        sql_repository.chunks.mark_pending_retry(conn, [j.chunk_id for j in retryable], truncated)
        sql_repository.chunks.mark_failed(conn, [j.chunk_id for j in exhausted], truncated)

        file_id = jobs[0].file_id
        if exhausted:
            sql_repository.files.record_failed(conn, file_id, count=len(exhausted), error=truncated, now=now)
        _refresh_processing_status(conn, file_id, now)


def _refresh_processing_status(conn, file_id: str, now: float) -> None:
    """Move a file to a terminal processing state once nothing is outstanding.

    Only meaningful after the upload itself finishes -- while bytes are still
    arriving, an empty queue just means indexing has caught up.
    """
    progress = sql_repository.files.get_processing_progress(conn, file_id)
    if progress is None:
        return
    upload_status, chunks_total, chunks_indexed, chunks_failed = progress
    if upload_status != "completed":
        return

    settled = chunks_indexed + chunks_failed
    if settled < chunks_total:
        return

    status = "completed" if chunks_failed == 0 else "failed"
    sql_repository.files.set_processing_status(conn, file_id, status, now)


def recover_stuck_jobs() -> int:
    """Reset rows and files abandoned mid-flight by a crash. Called on startup.

    Safe to run unconditionally here specifically because it runs during
    FastAPI's lifespan startup, before the server accepts any connections (see
    app/main.py) -- so "upload_status = 'uploading'" at this exact moment
    cannot mean a live client is mid-request, only that the previous process
    died holding that state. That precondition would not hold if this were
    ever called from a live request path or a multi-replica deployment sharing
    this database; at that scale, ownership of "is this upload actually still
    active" needs a heartbeat or lease rather than inference from a status
    string (see README section 6).
    """
    now = time.time()
    with db.transaction() as conn:
        recovered = sql_repository.chunks.reset_stuck_processing(conn)

        # An upload interrupted mid-flight is no longer being written to by any
        # live connection; mark it so the client knows to resume.
        sql_repository.files.mark_interrupted_uploads(conn, now)

        # A crash between marking 'finalizing' and completing the rename
        # leaves the file mid-way through POST /complete. Since chunk PUTs
        # already refuse a 'finalizing' file, the only way forward is to
        # finish what complete() was doing, so put it back in 'uploading' and
        # let the client call /complete again -- it's idempotent past the
        # rename (storage.finalize() is a no-op if .dat already exists).
        sql_repository.files.revert_stuck_finalizing(conn, now)
    return recovered


def pending_count(file_id: str | None = None) -> int:
    """Jobs still queued, for tests and the status endpoint."""
    conn = db.get_connection()
    return sql_repository.chunks.pending_count(conn, file_id)
