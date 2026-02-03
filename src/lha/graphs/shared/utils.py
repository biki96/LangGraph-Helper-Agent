"""Shared utilities for graph nodes."""

from typing import Any


def get_queries(state: dict[str, Any]) -> list[str]:
    """Extract queries from state, with fallback to question.

    Args:
        state: Graph state with optional 'queries' and 'question' keys.

    Returns:
        List of non-empty query strings.
    """
    queries = state.get("queries") or [state.get("question", "")]
    return [q for q in queries if q]
