"""Aggregator for all SQL-backed repositories.

Individual table repositories live in repositories/sql/, one file per
table (same convention as vector_repository.py and
file_storage_repository.py -- one file per kind of persistence). This
module just gives callers one import path so they don't need to know the
internal table-per-file split, and a single place to see how many SQL
repositories exist.

Usage:
    from app.repositories import sql_repository
    sql_repository.files.get_owned(conn, file_id, owner_id)
    sql_repository.chunks.mark_indexed(conn, chunk_ids)
"""

from .sql import chunk_repository as chunks
from .sql import file_repository as files

__all__ = ["files", "chunks"]
