"""Web subgraph builder - async with multi-query support."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lha.graphs.web.client import SearchDepth
from lha.graphs.web.nodes import create_search_node
from lha.graphs.web.state import WebState


def build_web_graph(
    api_key: str,
    max_results: int = 3,
    max_total: int = 10,
    include_raw_content: bool = True,
    search_depth: SearchDepth = "advanced",
) -> CompiledStateGraph:
    """Build async Web subgraph.

    Flow: START → search → END

    Input: {"queries": ["q1", "q2", "q3"]} or {"question": "..."}

    Args:
        api_key: Tavily API key.
        max_results: Max results per query.
        max_total: Max total results (top by score).
        include_raw_content: Include full page content.
        search_depth: "basic" or "advanced".

    Returns:
        Compiled Web subgraph.
    """
    graph = StateGraph(WebState)

    graph.add_node(
        "search",
        create_search_node(api_key, max_results, max_total, include_raw_content, search_depth),
    )

    graph.add_edge(START, "search")
    graph.add_edge("search", END)

    return graph.compile()
