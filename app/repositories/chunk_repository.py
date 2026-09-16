"""Data access for the `chunks` table.

Only SQL and row<->object translation live here -- queue semantics (retry
policy, batching, cross-table coordination with `files`) belong to
app.tasks.job_queue, which calls this module rather than touching the table
directly. Split out because job_queue.py had grown to mix both concerns;
here it's SQL only, same as file_repository.py.
"""

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkRow:
    """One `chunks` row, typed instead of passed around as a raw sqlite3.Row."""

    chunk_id: int
    file_id: str
    sequence: int
    start_byte: int
    end_byte: int
    retry_count: int

    @classmethod
    def from_row(cls, row) -> "ChunkRow":
        return cls(
            chunk_id=row["chunk_id"],
            file_id=row["file_id"],
            sequence=row["sequence"],
            start_byte=row["start_byte"],
            end_byte=row["end_byte"],
            retry_count=row["retry_count"],
        )


def insert_pending(conn, file_id: str, rows: list[tuple[int, int, int]]) -> None:
    """Insert (sequence, start_byte, end_byte) triples as pending chunks.

    ON CONFLICT DO NOTHING makes re-enqueueing the same sequence (a retried
    upload chunk) a no-op rather than a uniqueness error.
    """
    if not rows:
        return

    now = time.time()
    conn.executemany(
        """
        INSERT INTO chunks
            (file_id, sequence, start_byte, end_byte, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id, sequence) DO NOTHING
        """,
        [(file_id, sequence, start_byte, end_byte, now, now) for sequence, start_byte, end_byte in rows],
    )


def oldest_pending_file(conn) -> str | None:
    """The file with the oldest still-pending chunk, or None if the queue is empty."""
    row = conn.execute(
        """
        SELECT file_id
          FROM chunks
         WHERE status = 'pending'
         ORDER BY chunk_id
         LIMIT 1
        """
    ).fetchone()
    return row["file_id"] if row is not None else None


def select_pending_for_file(conn, file_id: str, limit: int) -> list[ChunkRow]:
    rows = conn.execute(
        """
        SELECT chunk_id, file_id, sequence, start_byte, end_byte, retry_count
          FROM chunks
         WHERE status = 'pending' AND file_id = ?
         ORDER BY sequence
         LIMIT ?
        """,
        (file_id, limit),
    ).fetchall()
    return [ChunkRow.from_row(row) for row in rows]


def mark_processing(conn, chunk_ids: list[int]) -> None:
    if not chunk_ids:
        return
    placeholders = ",".join("?" * len(chunk_ids))
    conn.execute(
        f"""
        UPDATE chunks
           SET status = 'processing', updated_at = ?
         WHERE chunk_id IN ({placeholders})
        """,
        [time.time(), *chunk_ids],
    )


def mark_indexed(conn, chunk_ids: list[int]) -> None:
    if not chunk_ids:
        return
    now = time.time()
    conn.executemany(
        "UPDATE chunks SET status = 'indexed', updated_at = ? WHERE chunk_id = ?",
        [(now, chunk_id) for chunk_id in chunk_ids],
    )


def mark_pending_retry(conn, chunk_ids: list[int], error: str) -> None:
    if not chunk_ids:
        return
    now = time.time()
    conn.executemany(
        """
        UPDATE chunks
           SET status = 'pending',
               retry_count = retry_count + 1,
               error_message = ?,
               updated_at = ?
         WHERE chunk_id = ?
        """,
        [(error, now, chunk_id) for chunk_id in chunk_ids],
    )


def mark_failed(conn, chunk_ids: list[int], error: str) -> None:
    if not chunk_ids:
        return
    now = time.time()
    conn.executemany(
        """
        UPDATE chunks
           SET status = 'failed',
               retry_count = retry_count + 1,
               error_message = ?,
               updated_at = ?
         WHERE chunk_id = ?
        """,
        [(error, now, chunk_id) for chunk_id in chunk_ids],
    )


def reset_stuck_processing(conn) -> int:
    """Reset rows left in `processing` by a crash back to `pending`. Returns the count."""
    cursor = conn.execute(
        "UPDATE chunks SET status = 'pending', updated_at = ? WHERE status = 'processing'",
        (time.time(),),
    )
    return cursor.rowcount


def pending_count(conn, file_id: str | None = None) -> int:
    if file_id is None:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE status IN ('pending', 'processing')"
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT COUNT(*) AS n
              FROM chunks
             WHERE status IN ('pending', 'processing') AND file_id = ?
            """,
            (file_id,),
        ).fetchone()
    return row["n"]
