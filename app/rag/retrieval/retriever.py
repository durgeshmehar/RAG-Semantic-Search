"""Nearest-neighbor retrieval over a file's embedded passages.

The write/ownership side of the vector store (creating a file's collection,
upserting vectors, deleting it) is app/repositories/vector_repository.py; this
module only reads. Both sides import the same client and collection-naming
helpers from the repository so they agree on which collection a file_id maps
to, rather than each computing that independently.
"""

import numpy as np

from ...repositories import vector_repository


def search(file_id: str, query_vector: np.ndarray, top_k: int) -> list[dict]:
    """Return the closest passages as dicts with byte ranges and scores.

    Vectors are cosine-configured on the collection, so `score` is already
    cosine similarity -- no separate normalisation step needed here.
    """
    client = vector_repository.get_client()
    collection = vector_repository.collection_name(file_id)
    if not client.collection_exists(collection):
        return []

    hits = client.query_points(
        collection_name=collection,
        query=query_vector.reshape(-1).tolist(),
        limit=top_k,
    ).points

    return [
        {
            "sequence": hit.payload["sequence"],
            "start_byte": hit.payload["start_byte"],
            "end_byte": hit.payload["end_byte"],
            "score": float(hit.score),
        }
        for hit in hits
    ]
