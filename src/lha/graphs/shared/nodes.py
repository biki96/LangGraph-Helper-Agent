"""Shared graph nodes."""

from datetime import datetime
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

REWRITE_PROMPT = """\
You are a search query optimizer. Given a user question, generate 3-5 focused search queries \
that will find the most relevant information.

Rules:
- Each query should target a different aspect of the question
- Use specific technical terms
- Include "{year}" for "best practices" or "latest" queries
- Keep queries concise (2-6 words each)

User question: {question}

Generate search queries:"""


class SearchQueries(BaseModel):
    """Optimized search queries."""

    queries: list[str] = Field(
        description="List of 3-5 optimized search queries",
        min_length=3,
        max_length=5,
    )


def create_rewrite_node(llm: BaseChatModel, state_key: str = "queries"):
    """Create query rewrite node using LLM.

    Args:
        llm: Chat model for query rewriting.
        state_key: Key in state to store generated queries.
    """
    structured_llm = llm.with_structured_output(SearchQueries)
    year = datetime.now().year

    async def rewrite(state: Any) -> dict[str, Any]:
        """Rewrite question into optimized search queries."""
        question = state.get("question", "")
        if not question:
            return {state_key: []}

        # If queries already provided, skip rewrite
        if state.get(state_key):
            return {}

        prompt = REWRITE_PROMPT.format(question=question, year=year)
        result = cast(SearchQueries, await structured_llm.ainvoke(prompt))
        return {state_key: result.queries}

    return rewrite
