"""Document loader with pluggable format parsers."""

import re
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseParser(ABC):
    """Base class for document parsers."""

    @abstractmethod
    def can_parse(self, content: str) -> bool:
        """Check if this parser can handle the content."""
        ...

    @abstractmethod
    def parse(self, content: str, source: str) -> list[Document]:
        """Parse content into Documents."""
        ...


class FrontmatterParser(BaseParser):
    """Parser for LangGraph format: ---\\npath.md\\n---\\ncontent"""

    def can_parse(self, content: str) -> bool:
        return content.strip().startswith("---")

    def parse(self, content: str, source: str) -> list[Document]:
        pattern = r"^---\n(.+?\.md)\n---\n"
        parts = re.split(pattern, content, flags=re.MULTILINE)

        docs = []
        for i in range(1, len(parts) - 1, 2):
            path, text = parts[i].strip(), parts[i + 1].strip()
            if text:
                title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "path": path,
                            "title": title.group(1) if title else path,
                        },
                    )
                )
        return docs


class SourceUrlParser(BaseParser):
    """Parser for LangChain format: # Title\\nSource: URL\\ncontent"""

    def can_parse(self, content: str) -> bool:
        return bool(re.search(r"^#\s+.+\nSource:", content, re.MULTILINE))

    def parse(self, content: str, source: str) -> list[Document]:
        sections = re.split(r"(?=^#\s+[^#])", content, flags=re.MULTILINE)

        docs = []
        for section in sections:
            section = section.strip()
            title = re.match(r"^#\s+(.+?)(?:\n|$)", section)
            url = re.search(r"^Source:\s*(.+?)(?:\n|$)", section, re.MULTILINE)
            if title and section:
                docs.append(
                    Document(
                        page_content=section,
                        metadata={
                            "source": source,
                            "path": url.group(1) if url else "",
                            "title": title.group(1).strip(),
                        },
                    )
                )
        return docs


class MarkdownParser(BaseParser):
    """Parser for plain markdown files - splits by H1 headers."""

    def can_parse(self, content: str) -> bool:
        return True  # Fallback parser

    def parse(self, content: str, source: str) -> list[Document]:
        sections = re.split(r"(?=^#\s+[^#])", content, flags=re.MULTILINE)

        docs = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            title = re.match(r"^#\s+(.+?)(?:\n|$)", section)
            docs.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": source,
                        "title": title.group(1).strip() if title else "Untitled",
                    },
                )
            )
        return docs


# Parser registry - order matters (first match wins)
PARSERS: list[BaseParser] = [
    FrontmatterParser(),
    SourceUrlParser(),
    MarkdownParser(),  # Fallback
]


def load_file(file_path: Path, source: str | None = None) -> list[Document]:
    """Load a file using the appropriate parser.

    Args:
        file_path: Path to the file.
        source: Source identifier. Defaults to filename stem.

    Returns:
        List of Documents.
    """
    content = file_path.read_text(encoding="utf-8")
    source = source or file_path.stem

    for parser in PARSERS:
        if parser.can_parse(content):
            return parser.parse(content, source)

    return []


def load_directory(
    directory: Path,
    pattern: str = "*.txt",
    source_map: dict[str, str] | None = None,
) -> list[Document]:
    """Load all matching files from a directory.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern for files.
        source_map: Optional mapping of filename substrings to source names.

    Returns:
        List of all Documents.
    """
    source_map = source_map or {}
    docs = []

    for file_path in directory.glob(pattern):
        # Determine source from map or filename
        source = next(
            (v for k, v in source_map.items() if k in file_path.name.lower()),
            file_path.stem,
        )
        docs.extend(load_file(file_path, source))

    return docs


def load_all_documents(data_dir: Path) -> list[Document]:
    """Load all llms.txt files from data/raw/ directory."""
    return load_directory(
        data_dir / "raw",
        pattern="*_llms_full.txt",
        source_map={"langgraph": "langgraph", "langchain": "langchain"},
    )


# Source mapping for incremental indexing
SOURCE_MAP = {"langgraph": "langgraph", "langchain": "langchain"}


def get_source_for_file(file_path: Path) -> str:
    """Determine source name for a file based on filename."""
    name = file_path.name.lower()
    for key, source in SOURCE_MAP.items():
        if key in name:
            return source
    return file_path.stem


def load_single_file(file_path: Path) -> tuple[str, list[Document]]:
    """Load a single file and return source name and documents.

    Args:
        file_path: Path to the file.

    Returns:
        Tuple of (source_name, documents).
    """
    source = get_source_for_file(file_path)
    docs = load_file(file_path, source)
    return source, docs
