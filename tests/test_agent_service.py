"""Unit tests for agent_service helpers and pipeline pieces (no live LLM)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_service import (
    _drain_feedback,
    _parse_verdict_json,
    new_session_id,
    plan_subtopics,
    synthesize_report,
    verify_subtopic,
)


class TestParseVerdictJson:
    def test_valid_pass(self):
        raw = json.dumps({"status": "pass", "issues": [], "summary": "Looks good."})
        v = _parse_verdict_json(raw)
        assert v["status"] == "pass"
        assert v["issues"] == []
        assert v["summary"] == "Looks good."

    def test_markdown_wrapped(self):
        raw = '```json\n{"status": "fail", "issues": ["gap"], "summary": "Weak."}\n```'
        v = _parse_verdict_json(raw)
        assert v["status"] == "fail"
        assert v["issues"] == ["gap"]

    def test_invalid_status_defaults_warn(self):
        v = _parse_verdict_json('{"status": "maybe", "issues": [], "summary": "x"}')
        assert v["status"] == "warn"

    def test_unparsable_returns_warn(self):
        v = _parse_verdict_json("not json at all")
        assert v["status"] == "warn"
        assert "Could not parse" in v["issues"][0]


class TestDrainFeedback:
    @pytest.mark.asyncio
    async def test_drains_until_empty(self):
        q: asyncio.Queue = asyncio.Queue()
        await q.put("a")
        await q.put("b")
        msgs = await _drain_feedback(q, max_items=10)
        assert msgs == ["a", "b"]
        assert q.empty()


class TestNewSessionId:
    def test_unique_uuids(self):
        a, b = new_session_id(), new_session_id()
        assert a != b
        assert len(a) == 36


@pytest.mark.asyncio
async def test_plan_subtopics_parses_json_array():
    msg = MagicMock()
    msg.content = '["Q1?", "Q2?", "Q3?"]'
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=msg)
    emit = AsyncMock()

    with patch("app.agent_service._llm", return_value=mock_llm):
        out = await plan_subtopics("AI safety", emit)

    assert out == ["Q1?", "Q2?", "Q3?"]


@pytest.mark.asyncio
async def test_plan_subtopics_fallback_lines():
    msg = MagicMock()
    msg.content = "- First question\n- Second\n- Third"
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=msg)
    emit = AsyncMock()

    with patch("app.agent_service._llm", return_value=mock_llm):
        out = await plan_subtopics("topic", emit)

    assert len(out) == 3
    assert out[0] == "First question"


@pytest.mark.asyncio
async def test_synthesize_report_no_chunks():
    vs = AsyncMock()
    vs.query_context = AsyncMock(return_value=[])
    emit = AsyncMock()

    report = await synthesize_report("topic", "sid", vs, emit, [])
    assert "No stored research chunks" in report
    emit.assert_awaited()


@pytest.mark.asyncio
async def test_verify_subtopic_no_chunks_fails():
    vs = AsyncMock()
    vs.query_context = AsyncMock(return_value=[])
    emit = AsyncMock()

    verdict = await verify_subtopic("topic", "sub?", 0, "sid", vs, "", emit)
    assert verdict["status"] == "fail"
    emit.assert_awaited()
