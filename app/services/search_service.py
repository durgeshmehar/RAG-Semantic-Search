"""Business logic for natural-language search over an uploaded file.

Embeds the query, asks the vector store for the nearest passages, and resolves
each hit's byte range back to real text. Qdrant carries the byte range as
point payload, so this never joins back to SQLite to locate a hit -- only to
confirm the file exists and is owned by the caller.
"""

from ..core.exceptions import NothingIndexedYet
from ..core.helper import storage
from ..db import db
from ..rag.ingestion import embedder
from ..rag.retrieval import retriever
from ..repositories import file_repository


class SearchHitResult:
    """A single ranked passage, already resolved to real text."""

    def __init__(self, text: str, start_byte: int, end_byte: int, score: float, sequence: int):
        self.text = text
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.score = score
        self.sequence = sequence


def search(file_id: str, owner_id: str, query: str, top_k: int) -> list[SearchHitResult]:
    """Return the sections whose meaning is closest to `query`.

    Searching is allowed while the upload is still in progress -- whatever has
    been indexed so far is queryable.
    """
    conn = db.get_connection()
    record = file_repository.get_owned(conn, file_id, owner_id)

    if record.chunks_indexed == 0:
        raise NothingIndexedYet()

    query_vector = embedder.embed_query(query)
    hits = retriever.search(file_id, query_vector, top_k)

    results = []
    for hit in hits:
        text = storage.read_range(file_id, hit["start_byte"], hit["end_byte"])
        if not text:
            # A genuinely empty read only happens if the file's gone; a
            # whitespace-only passage is legitimate content and must not be
            # dropped from results just because it strips to nothing.
            continue
        results.append(
            SearchHitResult(
                text=text,
                start_byte=hit["start_byte"],
                end_byte=hit["end_byte"],
                score=hit["score"],
                sequence=hit["sequence"],
            )
        )

    return results
