"""MCP server that owns the ChromaDB instance.

Exposes four tools consumed by the research pipeline MCP client:
  - create_collection
  - add_documents
  - query_context
  - collection_count

Run via stdio transport (spawned as a subprocess by VectorStoreClient).

Usage:
    python chroma_mcp_server.py
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

CHROMA_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent / "chroma_data"),
)

_client = chromadb.PersistentClient(path=CHROMA_DIR)
_emb = embedding_functions.DefaultEmbeddingFunction()

mcp = FastMCP("chroma-vector-store")


def _collection_name(session_id: str) -> str:
    return f"research_{session_id.replace('-', '_')}"[:512]


def _get(session_id: str):
    return _client.get_or_create_collection(
        _collection_name(session_id), embedding_function=_emb
    )


@mcp.tool()
def create_collection(session_id: str) -> dict:
    """Ensure a per-session collection exists. Safe to call multiple times.

    Args:
        session_id: Unique research session identifier.

    Returns:
        {"name": str, "session_id": str}
    """
    col = _get(session_id)
    return {"name": col.name, "session_id": session_id}


@mcp.tool()
def add_documents(
    session_id: str,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> dict:
    """Add documents to a session's collection.

    Args:
        session_id: Research session identifier.
        texts: Document texts to embed and store.
        metadatas: Parallel list of metadata dicts (same length as texts).
        ids: Parallel list of unique string IDs (same length as texts).

    Returns:
        {"added": int}
    """
    if not texts:
        return {"added": 0}
    col = _get(session_id)
    col.add(documents=texts, metadatas=metadatas, ids=ids)
    return {"added": len(texts)}


@mcp.tool()
def query_context(
    session_id: str,
    query: str,
    n_results: int = 8,
) -> list[dict]:
    """Semantic search over a session's stored documents.

    Args:
        session_id: Research session identifier.
        query: Query string to embed and search.
        n_results: Maximum number of results to return.

    Returns:
        List of {"text": str, "metadata": dict, "distance": float}
    """
    col = _get(session_id)
    count = col.count()
    if count == 0:
        return []
    k = min(n_results, max(1, count))
    res = col.query(query_texts=[query], n_results=k)
    out: list[dict] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    for doc, meta, dist in zip(docs[0], metas[0], dists[0], strict=False):
        out.append({"text": doc, "metadata": meta or {}, "distance": dist})
    return out


@mcp.tool()
def collection_count(session_id: str) -> dict:
    """Return the number of stored documents for a session.

    Args:
        session_id: Research session identifier.

    Returns:
        {"count": int}
    """
    col = _get(session_id)
    return {"count": col.count()}


if __name__ == "__main__":
    mcp.run(transport="stdio")
