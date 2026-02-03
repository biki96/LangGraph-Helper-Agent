"""RAG subgraph builder."""

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from lha.graphs.rag.nodes import create_retrieve_node
from lha.graphs.rag.state import RagState


def build_rag_graph(
    vectorstore: VectorStore,
    k: int = 3,
    max_docs: int = 10,
    score_threshold: float | None = None,
) -> CompiledStateGraph:
    """Build RAG subgraph.

    Flow: START → retrieve → END

    Args:
        vectorstore: Vector store for retrieval.
        k: Max documents to retrieve per query.
        max_docs: Max total documents after dedupe.
        score_threshold: Min similarity score.

    Returns:
        Compiled RAG subgraph.
    """
    graph = StateGraph(RagState)

    graph.add_node("retrieve", create_retrieve_node(vectorstore, k, max_docs, score_threshold))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)

    return graph.compile()
