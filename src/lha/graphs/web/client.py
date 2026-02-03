"""Async Tavily client wrapper and LangChain tool."""

import asyncio
import logging
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from lha.graphs.web.state import WebResult

logger = logging.getLogger(__name__)

SearchDepth = Literal["basic", "advanced"]


def expand_query(q: str) -> list[str]:
    """Expand single query into multiple search queries for comprehensive results.

    Args:
        q: Base query/topic to expand.

    Returns:
        List of expanded queries for parallel search.
    """
    from datetime import datetime

    year = datetime.now().year
    return [
        q,
        f"{q} documentation",
        f"{q} examples",
        f"{q} best practices {year}",
        f"{q} tutorial",
    ]


class TavilySearchInput(BaseModel):
    """Input schema for Tavily search tool."""

    query: str | list[str] = Field(
        description="Search query string, or list of queries for parallel search."
    )


class TavilyClient:
    """Async Tavily search client."""

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        include_raw_content: bool = True,
        search_depth: SearchDepth = "advanced",
    ):
        self.client = AsyncTavilyClient(api_key)
        self.max_results = max_results
        self.include_raw_content = include_raw_content
        self.search_depth: SearchDepth = search_depth

    async def search(self, query: str) -> list[WebResult]:
        """Search single query."""
        try:
            response = await self.client.search(
                query,
                max_results=self.max_results,
                include_raw_content=self.include_raw_content,
                search_depth=self.search_depth,
            )
            return [
                WebResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    content=r.get("content", ""),
                    raw_content=r.get("raw_content", ""),
                    score=r.get("score", 0.0),
                )
                for r in response.get("results", [])
            ]
        except (httpx.RequestError, httpx.HTTPStatusError, TimeoutError) as e:
            logger.warning(f"Tavily search failed for '{query[:50]}...': {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Tavily search: {e}")
            return []

    async def search_many(self, queries: list[str]) -> list[WebResult]:
        """Search multiple queries in parallel, return flattened results."""
        if not queries:
            return []

        responses = await asyncio.gather(
            *(self.search(q) for q in queries),
            return_exceptions=True,
        )

        # Flatten results, skip exceptions
        results: list[WebResult] = []
        seen_urls: set[str] = set()

        for response in responses:
            if isinstance(response, BaseException):
                continue
            for r in response:
                # Dedupe by URL
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    results.append(r)

        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results


class TavilySearchTool(BaseTool):
    """LangChain tool wrapper for Tavily search. Async-only."""

    name: str = "tavily_search"
    description: str = (
        "Search the web for information. Use this when you need current "
        "information or facts you don't know."
    )
    args_schema: type[BaseModel] = TavilySearchInput

    client: TavilyClient = Field(exclude=True)

    def _run(self, query: str | list[str]) -> dict[str, Any]:
        """Sync not supported."""
        raise RuntimeError("Async-only tool. Use `await tool.ainvoke(...)`.")

    async def _arun(self, query: str | list[str]) -> dict[str, Any]:
        """Async search, returns structured output."""
        if isinstance(query, str):
            results = await self.client.search(query)
        else:
            results = await self.client.search_many(query)
        return {"results": [r.model_dump() for r in results]}
