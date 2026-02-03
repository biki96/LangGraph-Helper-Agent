"""Document loading and chunking."""

from lha.documents.chunker import chunk_documents
from lha.documents.loader import (
    PARSERS,
    BaseParser,
    FrontmatterParser,
    MarkdownParser,
    SourceUrlParser,
    get_source_for_file,
    load_all_documents,
    load_directory,
    load_file,
    load_single_file,
)

__all__ = [
    # Chunker
    "chunk_documents",
    # Loader functions
    "load_all_documents",
    "load_directory",
    "load_file",
    "load_single_file",
    "get_source_for_file",
    # Parsers (for extension)
    "BaseParser",
    "FrontmatterParser",
    "MarkdownParser",
    "SourceUrlParser",
    "PARSERS",
]
