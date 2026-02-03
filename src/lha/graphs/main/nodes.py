"""Main graph nodes."""

from collections.abc import Awaitable, Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from lha.graphs.main.state import MainState, Mode, Source
from lha.graphs.shared import get_queries
from lha.graphs.web.state import WebResult

# =============================================================================
# Formatting
# =============================================================================


def format_docs(docs: list[Document]) -> str:
    """Format documents into context string."""
    if not docs:
        return ""

    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        parts.append(f"[{source}] {title}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def format_web_results(results: list[WebResult]) -> str:
    """Format web results into context string."""
    if not results:
        return ""

    parts = []
    for r in results:
        content = r.raw_content or r.content
        if content:
            parts.append(f"[{r.url}] {r.title}\n{content[:3000]}")

    return "\n\n---\n\n".join(parts)


# =============================================================================
# Subgraph Nodes
# =============================================================================


def create_rag_node(
    rag_graph: CompiledStateGraph,
) -> Callable[[MainState], Awaitable[dict]]:
    """Create async node that invokes RAG subgraph."""

    async def invoke_rag(state: MainState) -> dict:
        queries = get_queries(state)
        result = await rag_graph.ainvoke({"queries": queries})
        return {"documents": result.get("documents", [])}

    return invoke_rag


def create_web_node(
    web_graph: CompiledStateGraph,
) -> Callable[[MainState], Awaitable[dict]]:
    """Create async node that invokes Web subgraph."""

    async def invoke_web(state: MainState) -> dict:
        queries = get_queries(state)
        result = await web_graph.ainvoke({"queries": queries})
        return {"web_results": result.get("results", [])}

    return invoke_web


# =============================================================================
# Generator Node
# =============================================================================

SYSTEM_PROMPT = """You are an expert assistant for LangGraph and LangChain.

Answer based on the provided context. If insufficient, say so clearly.

Guidelines:
- Be concise and direct
- Include code examples when relevant
- Distinguish between LangGraph and LangChain

{context}"""


def create_generate_node(llm: BaseChatModel) -> Callable[[MainState], Awaitable[dict]]:
    """Create async generate node."""

    async def generate(state: MainState) -> dict:
        docs = state.get("documents", [])
        web_results = state.get("web_results", [])

        # Build context
        context_parts = []
        if docs:
            context_parts.append("Documentation:\n" + format_docs(docs))
        if web_results:
            context_parts.append("Web Search:\n" + format_web_results(web_results))

        context = "\n\n".join(context_parts) if context_parts else "No context available."

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=state.get("question")),
        ]

        response = await llm.ainvoke(messages)

        # Extract sources
        sources: list[Source] = []
        for doc in docs:
            sources.append({
                "type": "doc",
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("source", ""),
            })
        for r in web_results:
            sources.append({
                "type": "web",
                "title": r.title,
                "url": r.url,
            })

        return {
            "answer": response.content,
            "docs_used": len(docs),
            "web_results_used": len(web_results),
            "sources": sources,
        }

    return generate


# =============================================================================
# Router
# =============================================================================


def route_by_mode(state: MainState, config: RunnableConfig) -> str | list[str]:
    """Route based on mode. Returns list for parallel execution."""
    mode: Mode = config.get("configurable", {}).get("mode", "offline")

    if mode == "offline":
        return "rag"
    elif mode == "online":
        return "web"
    else:  # hybrid - parallel execution
        return ["rag", "web"]


def after_search(state: MainState, config: RunnableConfig) -> str:
    """After RAG/Web, go to generate."""
    return "generate"
