"""MCP client wrapper for the ChromaDB vector store server.

Spawns chroma_mcp_server.py as a subprocess and communicates via the MCP
stdio transport. The rest of the app calls the async methods here instead of
touching ChromaDB directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_SERVER_SCRIPT = str(Path(__file__).resolve().parent.parent / "chroma_mcp_server.py")


class VectorStoreClient:
    """Async MCP client that owns the connection to chroma_mcp_server."""

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._exit_stack = None

    async def connect(self) -> None:
        """Spawn the MCP server process and perform the MCP handshake."""
        from contextlib import AsyncExitStack

        params = StdioServerParameters(
            command=sys.executable,
            args=[_SERVER_SCRIPT],
        )
        self._exit_stack = AsyncExitStack()
        transport = await self._exit_stack.enter_async_context(stdio_client(params))
        read_stream, write_stream = transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._session = session

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    async def _call(self, tool: str, **kwargs) -> object:
        assert self._session is not None, "VectorStoreClient not connected"
        result = await self._session.call_tool(tool, arguments=kwargs)
        if result.isError:
            raise RuntimeError(f"MCP tool '{tool}' returned an error: {result.content}")
        # FastMCP 1.27 wraps the return value as {"result": <value>} in
        # structuredContent (content may be empty for structured responses).
        if result.structuredContent is not None:
            sc = result.structuredContent
            return sc["result"] if isinstance(sc, dict) and list(sc) == ["result"] else sc
        if result.content:
            raw = result.content[0].text  # type: ignore[union-attr]
            return json.loads(raw)
        return None

    # ------------------------------------------------------------------
    # Public API (mirrors chroma_mcp_server tools)
    # ------------------------------------------------------------------

    async def create_collection(self, session_id: str) -> None:
        """Ensure the per-session collection exists on the server."""
        await self._call("create_collection", session_id=session_id)

    async def add_documents(
        self,
        session_id: str,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> int:
        """Add documents; returns the count of added items."""
        result = await self._call(
            "add_documents",
            session_id=session_id,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return (result or {}).get("added", 0)  # type: ignore[union-attr]

    async def query_context(
        self,
        session_id: str,
        query: str,
        n_results: int = 8,
    ) -> list[dict]:
        """Semantic search; returns list of {text, metadata, distance}."""
        result = await self._call(
            "query_context",
            session_id=session_id,
            query=query,
            n_results=n_results,
        )
        return result or []  # type: ignore[return-value]

    async def collection_count(self, session_id: str) -> int:
        result = await self._call("collection_count", session_id=session_id)
        return (result or {}).get("count", 0)  # type: ignore[union-attr]
