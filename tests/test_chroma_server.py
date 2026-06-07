"""Tests for chroma_mcp_server helpers (no MCP subprocess)."""

from chroma_mcp_server import _collection_name


def test_collection_name_replaces_hyphens():
    name = _collection_name("abc-def-123")
    assert name == "research_abc_def_123"


def test_collection_name_truncated_at_512():
    long_id = "x" * 600
    assert len(_collection_name(long_id)) <= 512
