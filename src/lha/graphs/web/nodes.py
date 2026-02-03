"""Web subgraph nodes - async with multi-query support."""

from collections.abc import Awaitable, Callable

from lha.graphs.shared import get_queries
from lha.graphs.web.client import SearchDepth, TavilyClient
from lha.graphs.web.state import WebState


def create_search_node(
    api_key: str,
    max_results: int = 3,
    max_total: int = 10,
    include_raw_content: bool = True,
    search_depth: SearchDepth = "advanced",
) -> Callable[[WebState], Awaitable[dict]]:
    """Create async search node.

    Args:
        api_key: Tavily API key.
        max_results: Max results per query.
        max_total: Max total results (top by score).
        include_raw_content: Include full page content.
        search_depth: "basic" or "advanced".

    Returns:
        Async search node function.
    """
    client = TavilyClient(api_key, max_results, include_raw_content, search_depth)

    async def search(state: WebState) -> dict:
        """Search the web - supports single question or multiple queries."""
        queries = get_queries(state)

        if not queries:
            return {"results": []}

        # search_many returns deduped, sorted by score
        results = await client.search_many(queries)
        return {"results": results[:max_total]}

    return search
