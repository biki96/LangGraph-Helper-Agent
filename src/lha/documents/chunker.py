"""Document chunker using LangChain text splitters."""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Headers to split on
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Chunk documents by markdown headers, then by size.

    Uses MarkdownHeaderTextSplitter to split by headers first,
    then RecursiveCharacterTextSplitter for size limits.

    Args:
        documents: Documents to chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between chunks.

    Returns:
        List of chunked Documents with header metadata.
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: list[Document] = []

    for doc in documents:
        # First split by headers
        header_splits = md_splitter.split_text(doc.page_content)

        # Then split large sections by size
        sized_splits = text_splitter.split_documents(header_splits)

        # Merge original metadata with header metadata
        for chunk in sized_splits:
            chunk.metadata = {**doc.metadata, **chunk.metadata}

        all_chunks.extend(sized_splits)

    return all_chunks
