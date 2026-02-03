"""Documentation download functionality."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

MANIFEST_NAME = "manifest.json"

DOCS = {
    "langgraph_llms_full": {
        "url": "https://langchain-ai.github.io/langgraph/llms-full.txt",
        "filename": "langgraph_llms_full.txt",
    },
    "langchain_llms_full": {
        "url": "https://docs.langchain.com/llms-full.txt",
        "filename": "langchain_llms_full.txt",
    },
}


@dataclass(frozen=True)
class DownloadResult:
    name: str
    url: str
    path: Path
    status: str  # "downloaded" | "not_modified" | "failed"
    bytes: int = 0
    sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None


def sha256_file(path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    """Load manifest from disk."""
    if not path.exists():
        return {"version": 1, "generated_at": None, "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Save manifest to disk."""
    manifest["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def download_one(
    client: httpx.Client,
    name: str,
    url: str,
    dest: Path,
    etag: str | None,
    last_modified: str | None,
    force: bool,
    console: Console,
) -> DownloadResult:
    """Download a single file with conditional request support."""
    headers: dict[str, str] = {"User-Agent": "LangGraph-Helper-Agent/0.1"}
    if not force:
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

    try:
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code == 304:
                return DownloadResult(
                    name=name,
                    url=url,
                    path=dest,
                    status="not_modified",
                    etag=etag,
                    last_modified=last_modified,
                )

            r.raise_for_status()

            total = int(r.headers.get("Content-Length", "0")) or None
            new_etag = r.headers.get("ETag")
            new_last_modified = r.headers.get("Last-Modified")

            dest.parent.mkdir(parents=True, exist_ok=True)

            bytes_written = 0
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {dest.name}", total=total)
                with dest.open("wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
                        bytes_written += len(chunk)
                        progress.update(task, advance=len(chunk))

            digest = sha256_file(dest)
            return DownloadResult(
                name=name,
                url=url,
                path=dest,
                status="downloaded",
                bytes=bytes_written,
                sha256=digest,
                etag=new_etag,
                last_modified=new_last_modified,
            )

    except httpx.HTTPStatusError as e:
        return DownloadResult(
            name=name,
            url=url,
            path=dest,
            status="failed",
            error=f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
        )
    except httpx.RequestError as e:
        return DownloadResult(
            name=name,
            url=url,
            path=dest,
            status="failed",
            error=f"Network error: {e!s}",
        )


def download_docs(
    output_dir: Path,
    force: bool = False,
    timeout: float = 30.0,
    console: Console | None = None,
) -> list[DownloadResult]:
    """Download all documentation files.

    Args:
        output_dir: Directory to save files.
        force: Force redownload even if not modified.
        timeout: Request timeout in seconds.
        console: Rich console for output.

    Returns:
        List of download results.
    """
    console = console or Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    files: dict[str, Any] = manifest.setdefault("files", {})

    results: list[DownloadResult] = []

    limits = httpx.Limits(max_connections=5, max_keepalive_connections=5)
    with httpx.Client(timeout=timeout, limits=limits, follow_redirects=True) as client:
        for name, cfg in DOCS.items():
            url = cfg["url"]
            dest = output_dir / cfg["filename"]

            prev = files.get(name, {})
            res = download_one(
                client=client,
                name=name,
                url=url,
                dest=dest,
                etag=prev.get("etag"),
                last_modified=prev.get("last_modified"),
                force=force,
                console=console,
            )

            results.append(res)

            if res.status == "downloaded":
                console.print(f"[green]✓ {name}: downloaded ({res.bytes / 1024:.1f} KB)[/green]")
                files[name] = {
                    "url": url,
                    "filename": dest.name,
                    "etag": res.etag,
                    "last_modified": res.last_modified,
                    "sha256": res.sha256,
                    "bytes": res.bytes,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            elif res.status == "not_modified":
                console.print(f"[cyan]• {name}: not modified (cached)[/cyan]")
            else:
                console.print(f"[red]✗ {name}: {res.error}[/red]")

    save_manifest(manifest_path, manifest)
    return results
