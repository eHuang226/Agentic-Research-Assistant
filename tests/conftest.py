"""Shared fixtures: isolated sessions, mocked vector store, async HTTP client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app import main as main_module
from app.main import app, lifespan


@pytest.fixture(autouse=True)
def _clear_sessions():
    main_module.sessions.clear()
    yield
    main_module.sessions.clear()


@pytest.fixture
def mock_vs_client():
    client = MagicMock()

    async def connect() -> None:
        pass

    async def close() -> None:
        pass

    async def create_collection(session_id: str) -> None:
        pass

    async def add_documents(session_id, texts, metadatas, ids) -> int:
        return 1

    async def query_context(session_id, query, n_results=8) -> list:
        return []

    async def collection_count(session_id: str) -> int:
        return 0

    client.connect = connect
    client.close = close
    client.create_collection = create_collection
    client.add_documents = add_documents
    client.query_context = query_context
    client.collection_count = collection_count
    return client


@pytest.fixture
async def client(mock_vs_client, monkeypatch):
    monkeypatch.setattr(main_module, "vs_client", mock_vs_client)
    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
