"""Data access for the `files` table.

Only SQL and row<->object translation live here -- no business rules (offset
validation, size limits, binary detection all belong to the service layer)
and no HTTP awareness (raises app.errors.FileNotFound, never HTTPException).
Swapping SQLite for Postgres later means changing this file and
app/infra/db.py; the service layer that calls it should not need to change.
"""

import time
from dataclasses import dataclass

from ..errors import FileNotFound
from ..infra import db


@dataclass(frozen=True)
class FileRecord:
    """A `files` row, typed instead of passed around as a raw sqlite3.Row."""

    file_id: str
    owner_id: str
    filename: str
    total_size: int
    bytes_received: int
    upload_status: str
    processing_status: str
    chunks_total: int
    chunks_indexed: int
    chunks_failed: int
    indexed_watermark: int
    pending_tail: bytes
    next_sequence: int
    error_message: str | None
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row) -> "FileRecord":
        return cls(
            file_id=row["file_id"],
            owner_id=row["owner_id"],
            filename=row["filename"],
            total_size=row["total_size"],
            bytes_received=row["bytes_received"],
            upload_status=row["upload_status"],
            processing_status=row["processing_status"],
            chunks_total=row["chunks_total"],
            chunks_indexed=row["chunks_indexed"],
            chunks_failed=row["chunks_failed"],
            indexed_watermark=row["indexed_watermark"],
            pending_tail=bytes(row["pending_tail"]),
            next_sequence=row["next_sequence"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def create(conn, file_id: str, owner_id: str, filename: str, total_size: int) -> None:
    now = time.time()
    conn.execute(
        """
        INSERT INTO files
            (file_id, owner_id, filename, total_size, upload_status,
             processing_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'pending', 'pending', ?, ?)
        """,
        (file_id, owner_id, filename, total_size, now, now),
    )


def get(conn, file_id: str) -> FileRecord | None:
    """Raw lookup, no ownership check -- most callers want `get_owned` instead."""
    row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
    return FileRecord.from_row(row) if row is not None else None


def get_owned(conn, file_id: str, owner_id: str) -> FileRecord:
    """Look up a file, enforcing that the caller owns it.

    Raises FileNotFound for both "doesn't exist" and "belongs to someone
    else" -- deliberately the same error for both, so a client can't
    distinguish the two by probing ids (object-level access control, the same
    reasoning most APIs use for returning 404 rather than 403 here).
    """
    record = get(conn, file_id)
    if record is None or record.owner_id != owner_id:
        raise FileNotFound(file_id)
    return record


def list_owned(conn, owner_id: str, limit: int) -> list[FileRecord]:
    rows = conn.execute(
        "SELECT * FROM files WHERE owner_id = ? ORDER BY created_at DESC LIMIT ?",
        (owner_id, limit),
    ).fetchall()
    return [FileRecord.from_row(row) for row in rows]


def update_after_chunk(
    conn,
    file_id: str,
    *,
    bytes_received: int,
    chunks_added: int,
    next_sequence: int,
    pending_tail: bytes,
) -> None:
    conn.execute(
        """
        UPDATE files
           SET bytes_received = ?,
               upload_status = 'uploading',
               chunks_total = chunks_total + ?,
               next_sequence = ?,
               pending_tail = ?,
               updated_at = ?
         WHERE file_id = ?
        """,
        (bytes_received, chunks_added, next_sequence, pending_tail, time.time(), file_id),
    )


def mark_finalizing(
    conn, file_id: str, *, chunks_added: int, next_sequence: int, total_size: int
) -> None:
    conn.execute(
        """
        UPDATE files
           SET upload_status = 'finalizing',
               chunks_total = chunks_total + ?,
               next_sequence = ?,
               pending_tail = x'',
               total_size = ?,
               updated_at = ?
         WHERE file_id = ?
        """,
        (chunks_added, next_sequence, total_size, time.time(), file_id),
    )


def mark_completed(conn, file_id: str) -> None:
    conn.execute(
        "UPDATE files SET upload_status = 'completed', updated_at = ? WHERE file_id = ?",
        (time.time(), file_id),
    )


def delete(conn, file_id: str) -> None:
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
