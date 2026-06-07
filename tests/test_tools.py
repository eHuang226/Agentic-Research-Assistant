"""Unit tests for tools.py (network calls mocked)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.tools import _truncate, make_tools, read_web_page, search_web


class TestTruncate:
    def test_collapses_whitespace(self):
        assert _truncate("hello   world") == "hello world"

    def test_truncates_long_text(self):
        long = "a" * 50
        out = _truncate(long, max_len=20)
        assert len(out) == 20
        assert out.endswith("...")


def test_search_web_formats_results():
    fake_results = [
        {"title": "T1", "href": "https://a.com", "body": "snippet"},
        {"title": "T2", "url": "https://b.com", "body": "b"},
    ]
    mock_ddgs = MagicMock()
    mock_ddgs.__enter__.return_value.text.return_value = fake_results
    mock_ddgs.__exit__.return_value = None

    with patch("app.tools.DDGS", return_value=mock_ddgs):
        out = search_web("query", max_results=2)

    assert "T1" in out
    assert "https://a.com" in out
    assert "https://b.com" in out


def test_search_web_handles_failure():
    with patch("app.tools.DDGS", side_effect=RuntimeError("network")):
        out = search_web("q")
    assert out.startswith("Search failed")


def test_read_web_page_success():
    html = "<html><body><p>Hello</p><script>x</script></body></html>"
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    with patch("app.tools.httpx.get", return_value=mock_resp):
        out = read_web_page("https://example.com")

    assert "Hello" in out
    assert "x" not in out


def test_read_web_page_fetch_error():
    with patch("app.tools.httpx.get", side_effect=Exception("timeout")):
        out = read_web_page("https://bad.example")
    assert out.startswith("Failed to fetch")


@pytest.mark.asyncio
async def test_store_tool_schedules_add_documents(mock_vs_client):
    loop = asyncio.get_running_loop()
    stored: list[tuple] = []

    def on_store(excerpt, url, title):
        stored.append((excerpt, url, title))

    tools = make_tools("sess-1", mock_vs_client, loop, on_store=on_store)
    store = next(t for t in tools if t.name == "store_research_chunk")
    # invoke is sync and blocks the loop via future.result(); run in a worker thread.
    result = await asyncio.to_thread(
        store.invoke,
        {"excerpt": "fact", "source_url": "https://x.com", "title": "X"},
    )

    assert "Stored chunk" in result
    assert stored == [("fact", "https://x.com", "X")]
