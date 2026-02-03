"""Modular vector store with pluggable embeddings and backends."""

from lha.vectorstore.store import (
    DEFAULT_COLLECTION,
    DEFAULT_K,
    add_documents,
    create_chroma_store,
    delete_by_source,
    index_to_chroma,
    query,
    query_with_scores,
)

__all__ = [
    "DEFAULT_COLLECTION",
    "DEFAULT_K",
    # VectorStore factories
    "create_chroma_store",
    "index_to_chroma",
    # Incremental indexing
    "delete_by_source",
    "add_documents",
    # Generic operations
    "query",
    "query_with_scores",
]
