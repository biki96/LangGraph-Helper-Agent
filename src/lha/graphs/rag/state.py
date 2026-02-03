"""RAG subgraph state."""

from typing import TypedDict

from langchain_core.documents import Document


class RagState(TypedDict, total=False):
    """State for RAG subgraph."""

    question: str
    queries: list[str]
    documents: list[Document]
