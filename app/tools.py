"""LangChain tools: web search, fetch page, store chunks (grounding)."""

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Callable

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.vector_store import VectorStoreClient


def _truncate(s: str, max_len: int = 12000) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def search_web(query: str, max_results: int = 2) -> str:
    """DuckDuckGo search; returns titles + snippets + URLs (no API key)."""
    lines: list[str] = []
    try:
        with DDGS() as ddgs:
            raw = ddgs.text(query.strip(), max_results=max_results) or []
    except Exception as e:  # noqa: BLE001
        return f"Search failed ({type(e).__name__}): {e}"
    for i, r in enumerate(raw):
        title = r.get("title") or ""
        href = r.get("href") or r.get("url") or ""
        body = r.get("body") or ""
        lines.append(f"{i + 1}. {title}\n   URL: {href}\n   {body}")
    return "\n\n".join(lines) if lines else "No results found. Try a shorter or different query."


def read_web_page(url: str) -> str:
    """Fetch a page and return main text (best-effort)."""
    try:
        r = httpx.get(url, follow_redirects=True, timeout=25.0, headers={"User-Agent": "ResearchBot/1.0"})
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"Failed to fetch URL: {e}"
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return _truncate(text, 10000)


class StoreInput(BaseModel):
    excerpt: str = Field(..., description="Factual excerpt to store; keep close to source wording.")
    source_url: str = Field(..., description="Canonical URL this excerpt came from.")
    title: str = Field("", description="Page or article title if known.")


def make_tools(
    session_id: str,
    vs_client: VectorStoreClient,
    loop: asyncio.AbstractEventLoop,
    on_store: Callable[[str, str, str], None] | None = None,
):
    """Tools bound to a session via the MCP vector-store client.

    store_research_chunk runs from a sync LangChain thread; it schedules the
    async MCP call onto the main event loop with run_coroutine_threadsafe.
    """

    def store_research_chunk(excerpt: str, source_url: str, title: str = "") -> str:
        eid = str(uuid.uuid4())
        meta = {"source_url": source_url, "title": title or source_url}
        future = asyncio.run_coroutine_threadsafe(
            vs_client.add_documents(
                session_id=session_id,
                texts=[excerpt],
                metadatas=[meta],
                ids=[eid],
            ),
            loop,
        )
        try:
            future.result(timeout=30)
        except Exception as e:  # noqa: BLE001
            return f"Failed to store chunk: {e}"
        if on_store:
            on_store(excerpt, source_url, title)
        return f"Stored chunk {eid} ({len(excerpt)} chars) from {source_url}"

    store_tool = StructuredTool.from_function(
        name="store_research_chunk",
        description=(
            "Save a grounded excerpt and its source URL into the research memory. "
            "Call this after read_web_page with factual text you may cite later."
        ),
        func=lambda excerpt, source_url, title="": store_research_chunk(excerpt, source_url, title),
        args_schema=StoreInput,
    )

    search_tool = StructuredTool.from_function(
        name="search_web",
        description="Search the web for sources. Input: concise search query.",
        func=search_web,
    )

    read_tool = StructuredTool.from_function(
        name="read_web_page",
        description="Read and extract text from a specific HTTP(S) URL.",
        func=read_web_page,
    )

    return [search_tool, read_tool, store_tool]
