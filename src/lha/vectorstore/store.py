"""Modular vector store with pluggable embeddings and backends."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

DEFAULT_COLLECTION = "langgraph_docs"
DEFAULT_K = 5


# --- VectorStore Factories ---


def create_chroma_store(
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
) -> VectorStore:
    """Create or load ChromaDB vector store."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )


CHROMA_BATCH_SIZE = 5000  # ChromaDB max is ~5461


def index_to_chroma(
    documents: list[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = CHROMA_BATCH_SIZE,
) -> VectorStore:
    """Index documents into ChromaDB with batching.

    Args:
        documents: Documents to index.
        embeddings: Embedding function.
        persist_dir: ChromaDB persist directory.
        collection_name: Collection name.
        batch_size: Max documents per batch (ChromaDB limit ~5461).

    Returns:
        VectorStore instance.
    """
    from langchain_chroma import Chroma

    if not documents:
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
        )

    # First batch creates the collection
    first_batch = documents[:batch_size]
    store = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    # Add remaining batches
    for i in range(batch_size, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        store.add_documents(batch)

    return store


# --- Generic Operations ---


def query(
    store: VectorStore,
    text: str,
    k: int = DEFAULT_K,
    score_threshold: float | None = None,
) -> list[Document]:
    """Query any vector store for relevant documents.

    Args:
        store: Vector store to query.
        text: Query text.
        k: Maximum number of documents to return.
        score_threshold: Minimum similarity score (0-1). Only return docs above this.
                        Note: ChromaDB returns distance (lower=better), so we convert.

    Returns:
        List of relevant documents.
    """
    if score_threshold is None:
        return store.similarity_search(text, k=k)

    # Get results with scores
    results = store.similarity_search_with_relevance_scores(text, k=k)

    # Filter by threshold
    return [doc for doc, score in results if score >= score_threshold]


def query_with_scores(
    store: VectorStore,
    text: str,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """Query vector store and return documents with relevance scores.

    Args:
        store: Vector store to query.
        text: Query text.
        k: Maximum number of documents to return.

    Returns:
        List of (document, score) tuples, sorted by score descending.
    """
    return store.similarity_search_with_relevance_scores(text, k=k)


# --- Incremental Indexing ---


def delete_by_source(
    persist_dir: Path, source: str, collection_name: str = DEFAULT_COLLECTION
) -> int:
    """Delete all documents with a specific source from ChromaDB.

    Args:
        persist_dir: ChromaDB persist directory.
        source: Source value to match in metadata.
        collection_name: Collection name.

    Returns:
        Number of documents deleted.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return 0  # Collection doesn't exist

    # Get IDs of documents with this source
    results = collection.get(where={"source": source}, include=[])
    ids = results.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)


def add_documents(
    documents: list[Document],
    embeddings: Embeddings,
    persist_dir: Path,
    collection_name: str = DEFAULT_COLLECTION,
    batch_size: int = CHROMA_BATCH_SIZE,
) -> int:
    """Add documents to existing ChromaDB collection with batching.

    Args:
        documents: Documents to add.
        embeddings: Embedding function.
        persist_dir: ChromaDB persist directory.
        collection_name: Collection name.
        batch_size: Max documents per batch.

    Returns:
        Number of documents added.
    """
    if not documents:
        return 0

    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Add in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        store.add_documents(batch)

    return len(documents)
