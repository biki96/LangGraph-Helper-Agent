"""RAG subgraph nodes."""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from lha.graphs.rag.state import RagState
from lha.graphs.shared import get_queries
from lha.vectorstore import query_with_scores


def create_retrieve_node(
    vectorstore: VectorStore,
    k: int = 3,
    max_docs: int = 10,
    score_threshold: float | None = None,
) -> Callable[[RagState], dict]:
    """Create retrieve node.

    Args:
        vectorstore: Vector store to query.
        k: Max documents to retrieve per query.
        max_docs: Max total documents after dedupe (top by score).
        score_threshold: Min similarity score.

    Returns:
        Retrieve node function.
    """

    def retrieve(state: RagState) -> dict:
        """Retrieve relevant documents from multiple queries, ranked by score."""
        queries = get_queries(state)

        if not queries:
            return {"documents": []}

        # Collect all results with scores, dedupe by content
        doc_scores: dict[int, tuple[Document, float]] = {}

        for q in queries:
            results = query_with_scores(vectorstore, q, k=k)
            for doc, score in results:
                # Apply threshold if set
                if score_threshold and score < score_threshold:
                    continue

                content_hash = hash(doc.page_content)
                # Keep highest score for duplicate content
                if content_hash not in doc_scores or score > doc_scores[content_hash][1]:
                    doc_scores[content_hash] = (doc, score)

        # Sort by score descending, take top max_docs
        ranked = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:max_docs]]

        return {"documents": top_docs}

    return retrieve
