"""Ties ingestion together: embed a batch of already-chunked passages and
store the resulting vectors.

Chunking itself (rag/ingestion/chunker.py) runs earlier, inline in the upload
request path (see app/services/upload_service.py) -- passages are cut from
bytes already in memory as they arrive, not here. This module is the
embed-and-store half: given passage text and the coordinates identifying each
passage, call the embedding model and upsert the resulting vectors.

Deliberately a pure function with no knowledge of the job queue, retries, or
threading -- those are app/tasks/worker.py's job. This keeps "how ingestion
turns text into stored vectors" testable and readable independent of "how
work gets claimed and retried."

There is no generation stage: retrieval (app/rag/retrieval/retriever.py)
returns the matched source passages directly, since the assignment asks for
the relevant sections themselves, not an LLM-generated answer over them.
"""

from dataclasses import dataclass

from .ingestion import embedder
from ..repositories import vector_repository


@dataclass(frozen=True)
class IngestedPassage:
    """One passage's coordinates paired with its already-read text.

    The caller (app/tasks/worker.py) reads the text from disk via
    app/storage.py before calling ingest_batch -- this module never touches
    the filesystem, only the embedding model and the vector store.
    """

    sequence: int
    start_byte: int
    end_byte: int
    text: str


def ingest_batch(file_id: str, passages: list[IngestedPassage]) -> None:
    """Embed each passage's text and upsert the resulting vectors.

    Point IDs in the vector store are derived from (file_id, sequence), so
    calling this again for a passage already ingested (a worker retrying
    after a crash mid-batch) overwrites the same point rather than creating a
    duplicate -- see vector_repository.add_vectors.
    """
    if not passages:
        return

    vectors = embedder.embed_texts([p.text for p in passages])

    vector_repository.add_vectors(
        file_id,
        vectors,
        sequences=[p.sequence for p in passages],
        start_bytes=[p.start_byte for p in passages],
        end_bytes=[p.end_byte for p in passages],
    )
