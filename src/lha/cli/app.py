"""CLI application with Typer."""

import asyncio
import time
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown

from lha.cli.download import download_docs
from lha.config import get_settings

app = typer.Typer(
    name="lha",
    help="LangGraph Helper Agent - AI assistant for LangGraph and LangChain documentation.",
    add_completion=False,
)
console = Console()


@app.command()
def download(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save documentation files. Defaults to DATA_DIR/raw.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force redownload even if not modified.",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        "-s",
        help="Auto-reindex changed documents in vector store.",
    ),
    timeout: float = typer.Option(
        30.0,
        "--timeout",
        "-t",
        help="Request timeout in seconds.",
    ),
) -> None:
    """Download LangGraph and LangChain documentation for offline RAG."""
    from lha.documents import chunk_documents, get_source_for_file, load_single_file
    from lha.services import create_google_embeddings
    from lha.vectorstore import add_documents, delete_by_source

    try:
        settings = get_settings()
    except ValidationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1) from None

    output_dir = output_dir or settings.data_dir / "raw"

    console.print("[bold]LHA Documentation Downloader[/bold]")
    console.print(f"[dim]Output: {output_dir}[/dim]\n")

    results = download_docs(output_dir, force=force, timeout=timeout, console=console)

    # Summary
    downloaded = sum(1 for r in results if r.status == "downloaded")
    cached = sum(1 for r in results if r.status == "not_modified")
    failed = sum(1 for r in results if r.status == "failed")

    console.print(f"\n[dim]Done: {downloaded} downloaded, {cached} cached, {failed} failed[/dim]")

    # Auto-reindex changed files
    if sync and downloaded > 0:
        console.print("\n[bold]Syncing vector store...[/bold]")

        embeddings = create_google_embeddings(
            settings.google_api_key, model=settings.embedding_model
        )

        for r in results:
            if r.status != "downloaded":
                continue

            source = get_source_for_file(r.path)

            # Delete old documents for this source
            with console.status(f"[yellow]Removing old {source} docs...[/yellow]"):
                deleted = delete_by_source(settings.chroma_persist_dir, source)

            # Load and chunk new documents
            with console.status(f"[blue]Loading {source}...[/blue]"):
                _, docs = load_single_file(r.path)
                chunks = chunk_documents(docs)

            # Add to vector store
            with console.status(f"[green]Indexing {source}...[/green]"):
                added = add_documents(chunks, embeddings, settings.chroma_persist_dir)

            console.print(f"  [green]✓ {source}:[/green] {deleted} removed, {added} added")

        console.print("[green]Vector store synced![/green]")

    if failed > 0:
        raise typer.Exit(1)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about LangGraph/LangChain."),
    mode: str = typer.Option(
        None,
        "--mode",
        "-m",
        help="offline (RAG) | online (Web) | hybrid (both). Default: AGENT_MODE.",
    ),
    k: int | None = typer.Option(
        None,
        "--k",
        "-k",
        help="Max documents to retrieve. Defaults to RETRIEVAL_K.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Min similarity score (0-1). Defaults to RETRIEVAL_SCORE_THRESHOLD.",
    ),
) -> None:
    """Ask a question about LangGraph or LangChain documentation."""
    from lha.graphs import Mode, build_main_graph
    from lha.services import create_google_embeddings, create_google_llm
    from lha.vectorstore import create_chroma_store

    # Validate input
    MAX_QUESTION_LENGTH = 2000
    question = question.strip()
    if not question:
        console.print("[red]Error: Question cannot be empty.[/red]")
        raise typer.Exit(1)
    if len(question) > MAX_QUESTION_LENGTH:
        console.print(f"[red]Error: Question too long (max {MAX_QUESTION_LENGTH} chars).[/red]")
        raise typer.Exit(1)

    try:
        settings = get_settings()
    except ValidationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1) from None

    # Check vector store exists
    if not settings.chroma_persist_dir.exists():
        console.print(f"[red]Error: Vector store not found at {settings.chroma_persist_dir}[/red]")
        console.print("[dim]Run 'lha reindex' to build the index first.[/dim]")
        raise typer.Exit(1)

    # Use CLI args or fall back to settings
    effective_mode: Mode = mode or settings.agent_mode.value  # type: ignore
    k = k or settings.retrieval_k
    threshold = threshold if threshold is not None else settings.retrieval_score_threshold

    # Check tavily key
    tavily_key = settings.tavily_api_key
    if not tavily_key or tavily_key == "your_tavily_api_key_here":
        tavily_key = None
        if effective_mode in ("online", "hybrid"):
            console.print(
                f"[yellow]Warning: {effective_mode} mode requires TAVILY_API_KEY. "
                "Falling back to offline.[/yellow]"
            )
            effective_mode = "offline"

    console.print(f"[dim]Mode: {effective_mode}[/dim]")

    # Create components
    embeddings = create_google_embeddings(settings.google_api_key, model=settings.embedding_model)
    vectorstore = create_chroma_store(embeddings, settings.chroma_persist_dir)
    llm = create_google_llm(settings.google_api_key, model=settings.llm_model)

    # Use faster model for rewrite if different
    rewrite_llm = None
    if settings.rewrite_model != settings.llm_model:
        rewrite_llm = create_google_llm(settings.google_api_key, model=settings.rewrite_model)

    # Create and run graph
    graph = build_main_graph(
        vectorstore,
        llm,
        rewrite_llm=rewrite_llm,
        tavily_api_key=tavily_key,
        k=k,
        score_threshold=threshold,
    )

    # Node display names
    node_labels = {
        "rewrite": ("blue", "Rewriting queries"),
        "rag": ("cyan", "Retrieving docs"),
        "web": ("magenta", "Searching web"),
        "generate": ("green", "Generating answer"),
    }

    def format_status(active: set[str]) -> str:
        parts = []
        for name in ["rewrite", "rag", "web", "generate"]:
            if name in active and name in node_labels:
                color, label = node_labels[name]
                parts.append(f"[bold {color}]{label}[/bold {color}]")
        return " + ".join(parts) if parts else "[dim]Starting...[/dim]"

    async def run_with_status() -> tuple[dict, dict, dict]:
        result = {}
        active_nodes: set[str] = set()
        node_start_times: dict[str, float] = {}
        node_durations: dict[str, float] = {}
        token_usage = {"input": 0, "output": 0}
        total_start = time.time()

        with console.status("[dim]Starting...[/dim]") as status:
            async for event in graph.astream_events(
                {"question": question},
                config={"configurable": {"mode": effective_mode}},
                version="v2",
            ):
                event_type = event["event"]
                name = event.get("name", "")

                if event_type == "on_chain_start":
                    if name in node_labels:
                        active_nodes.add(name)
                        node_start_times[name] = time.time()
                        status.update(format_status(active_nodes))

                elif event_type == "on_chain_end":
                    if name in node_labels:
                        active_nodes.discard(name)
                        if name in node_start_times:
                            node_durations[name] = time.time() - node_start_times[name]
                        status.update(format_status(active_nodes))
                    if name == "LangGraph":
                        result = event["data"]["output"]

                elif event_type == "on_chat_model_end":
                    # Extract token usage from LLM response
                    output = event.get("data", {}).get("output", {})
                    usage = getattr(output, "usage_metadata", None)
                    if usage:
                        token_usage["input"] += usage.get("input_tokens", 0)
                        token_usage["output"] += usage.get("output_tokens", 0)

        total_time = time.time() - total_start
        node_durations["total"] = total_time
        return result, node_durations, token_usage

    result, timings, tokens = asyncio.run(run_with_status())

    # Display answer
    console.print()
    console.print(Markdown(result.get("answer", "No answer generated.")))

    # Display stats
    console.print()

    # Counts
    counts = [f"Docs: {result.get('docs_used', 0)}"]
    if result.get("web_results_used"):
        counts.append(f"Web: {result['web_results_used']}")

    # Tokens
    total_tokens = tokens["input"] + tokens["output"]
    token_stats = f"Tokens: {tokens['input']} in / {tokens['output']} out / {total_tokens} total"

    # Timings per node
    time_parts = []
    for node in ["rewrite", "rag", "web", "generate"]:
        if node in timings:
            time_parts.append(f"{node}: {timings[node]:.1f}s")
    time_parts.append(f"total: {timings.get('total', 0):.1f}s")

    console.print(f"[dim]{' | '.join(counts)}[/dim]")
    console.print(f"[dim]{token_stats}[/dim]")
    console.print(f"[dim]{' | '.join(time_parts)}[/dim]")

    # Sources
    sources = result.get("sources", [])
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        seen = set()
        for src in sources:
            key = (src["type"], src["url"])
            if key in seen:
                continue
            seen.add(key)
            icon = "📄" if src["type"] == "doc" else "🌐"
            title = src["title"] or "Untitled"
            url = src["url"]
            if src["type"] == "web":
                console.print(f"  {icon} [{title}]({url})")
            else:
                console.print(f"  {icon} {title} [dim]({url})[/dim]")


@app.command()
def reindex() -> None:
    """Rebuild the vector store index from documentation."""
    from lha.documents import chunk_documents, load_all_documents
    from lha.services import create_google_embeddings
    from lha.vectorstore import index_to_chroma

    try:
        settings = get_settings()
    except ValidationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1) from None

    console.print("[bold]LHA Reindex[/bold]")
    console.print(f"[dim]Data: {settings.data_dir}[/dim]")
    console.print(f"[dim]Index: {settings.chroma_persist_dir}[/dim]\n")

    # Load and chunk documents
    with console.status("[bold blue]Loading documents...[/bold blue]"):
        docs = load_all_documents(settings.data_dir)
    console.print(f"Loaded {len(docs)} documents")

    with console.status("[bold blue]Chunking...[/bold blue]"):
        chunks = chunk_documents(docs)
    console.print(f"Created {len(chunks)} chunks")

    # Index
    console.print(f"\nIndexing with {settings.embedding_model}...")
    embeddings = create_google_embeddings(settings.google_api_key, model=settings.embedding_model)
    index_to_chroma(chunks, embeddings, settings.chroma_persist_dir)

    console.print(f"\n[green]Done! Index saved to {settings.chroma_persist_dir}[/green]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version."),
) -> None:
    """LangGraph Helper Agent CLI."""
    if version:
        console.print("lha version 0.1.0")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
