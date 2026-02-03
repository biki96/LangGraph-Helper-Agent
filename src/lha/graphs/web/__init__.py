"""Web subgraph - async web search with multi-query support."""

from lha.graphs.web.client import SearchDepth, TavilyClient, TavilySearchTool, expand_query
from lha.graphs.web.graph import build_web_graph
from lha.graphs.web.state import WebResult, WebState

__all__ = [
    "SearchDepth",
    "TavilyClient",
    "TavilySearchTool",
    "WebResult",
    "WebState",
    "build_web_graph",
    "expand_query",
]
