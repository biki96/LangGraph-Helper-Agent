"""Main graph state."""

from typing import Literal, TypedDict

from langchain_core.documents import Document

from lha.graphs.web.state import WebResult

Mode = Literal["offline", "online", "hybrid"]


class Source(TypedDict):
    """Source reference."""

    type: str  # "doc" or "web"
    title: str
    url: str  # file path for docs, URL for web


class MainState(TypedDict, total=False):
    """State for main graph."""

    # Input
    question: str

    # After rewrite
    queries: list[str]

    # From RAG subgraph
    documents: list[Document]

    # From Web subgraph
    web_results: list[WebResult]

    # Output
    answer: str
    docs_used: int
    web_results_used: int
    sources: list[Source]
