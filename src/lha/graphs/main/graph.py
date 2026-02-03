"""Main graph builder - composes RAG and Web subgraphs."""

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lha.graphs.main.nodes import (
    after_search,
    create_generate_node,
    create_rag_node,
    create_web_node,
    route_by_mode,
)
from lha.graphs.main.state import MainState
from lha.graphs.rag import build_rag_graph
from lha.graphs.shared import create_rewrite_node
from lha.graphs.web import SearchDepth, build_web_graph


def build_main_graph(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    rewrite_llm: BaseChatModel | None = None,
    tavily_api_key: str | None = None,
    k: int = 3,
    max_docs: int = 10,
    score_threshold: float | None = None,
    web_max_results: int = 3,
    web_max_total: int = 10,
    include_raw_content: bool = True,
    search_depth: SearchDepth = "advanced",
) -> CompiledStateGraph:
    """Build main graph with RAG and Web subgraphs.

    Flow based on mode:
        - offline: rewrite → rag → generate
        - online:  rewrite → web → generate
        - hybrid:  rewrite → rag → web → generate

    Args:
        vectorstore: Vector store for RAG.
        llm: Language model for generation.
        rewrite_llm: Optional faster model for query rewriting.
        tavily_api_key: Tavily API key for web search.
        k: Max documents to retrieve per query.
        max_docs: Max total RAG documents (top by score).
        score_threshold: Min similarity score.
        web_max_results: Max web results per query.
        web_max_total: Max total web results (top by score).
        include_raw_content: Include full page content.
        search_depth: "basic" or "advanced".

    Returns:
        Compiled main graph.
    """
    # Build subgraphs
    rag_graph = build_rag_graph(
        vectorstore, k=k, max_docs=max_docs, score_threshold=score_threshold
    )

    if tavily_api_key:
        web_graph = build_web_graph(
            tavily_api_key,
            max_results=web_max_results,
            max_total=web_max_total,
            include_raw_content=include_raw_content,
            search_depth=search_depth,
        )
        web_node = create_web_node(web_graph)
    else:

        async def web_node(state: MainState) -> dict:
            return {"web_results": []}

    # Build main graph
    graph = StateGraph(MainState)

    graph.add_node("rewrite", create_rewrite_node(rewrite_llm or llm))
    graph.add_node("rag", create_rag_node(rag_graph))
    graph.add_node("web", web_node)
    graph.add_node("generate", create_generate_node(llm))

    # Routing:
    #   offline: rewrite → rag → generate
    #   online:  rewrite → web → generate
    #   hybrid:  rewrite → [rag, web] (parallel) → generate
    graph.add_edge(START, "rewrite")
    graph.add_conditional_edges("rewrite", route_by_mode, ["rag", "web"])
    graph.add_conditional_edges("rag", after_search, ["generate"])
    graph.add_conditional_edges("web", after_search, ["generate"])
    graph.add_edge("generate", END)

    return graph.compile()
