"""Web subgraph state."""

from typing import TypedDict

from pydantic import BaseModel


class WebResult(BaseModel):
    """Single web search result."""

    url: str = ""
    title: str = ""
    content: str = ""  # Snippet
    raw_content: str = ""  # Full page content
    score: float = 0.0


class WebState(TypedDict, total=False):
    """State for Web subgraph."""

    question: str
    queries: list[str]  # Multiple queries for parallel search
    results: list[WebResult]
