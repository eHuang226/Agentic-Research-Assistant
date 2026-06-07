"""FastAPI endpoint tests (pipeline and vector store mocked)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_start_research_returns_session(client):
    with patch("app.main.run_research_pipeline", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "Report body"
        r = await client.post("/api/research", json={"topic": "quantum dots"})
        for _ in range(50):
            if mock_run.await_count:
                break
            await asyncio.sleep(0.02)

    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "started"
    assert "session_id" in data
    mock_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_research_rejects_short_topic(client):
    r = await client.post("/api/research", json={"topic": "ab"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_start_research_conflict_when_running(client):
    from app import main as main_module

    sid = "fixed-session-id"
    with patch("app.main.run_research_pipeline", new_callable=AsyncMock):
        r1 = await client.post(
            "/api/research", json={"topic": "first topic", "session_id": sid}
        )
    assert r1.status_code == 200
    main_module.sessions[sid]["status"] = "running"

    r2 = await client.post(
        "/api/research", json={"topic": "second topic", "session_id": sid}
    )
    assert r2.status_code == 409


@pytest.mark.asyncio
async def test_feedback_unknown_session(client):
    r = await client.post(
        "/api/sessions/nope/feedback", json={"message": "focus EU sources"}
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_feedback_when_not_running(client):
    from app import main as main_module

    sid = "sess-fb"
    main_module.sessions[sid] = {
        "events": asyncio.Queue(),
        "feedback": asyncio.Queue(),
        "status": "done",
    }
    r = await client.post(
        f"/api/sessions/{sid}/feedback", json={"message": "too late"}
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_feedback_ok_while_running(client):
    from app import main as main_module

    sid = "sess-run"
    events: asyncio.Queue = asyncio.Queue()
    feedback: asyncio.Queue = asyncio.Queue()
    main_module.sessions[sid] = {
        "events": events,
        "feedback": feedback,
        "status": "running",
    }

    r = await client.post(
        f"/api/sessions/{sid}/feedback", json={"message": "cite primary sources"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert await feedback.get() == "cite primary sources"


@pytest.mark.asyncio
async def test_report_not_ready(client):
    from app import main as main_module

    sid = "sess-pending"
    main_module.sessions[sid] = {"status": "running"}
    r = await client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 202


@pytest.mark.asyncio
async def test_report_ready(client):
    from app import main as main_module

    sid = "sess-done"
    main_module.sessions[sid] = {"status": "done", "report": "# Done"}
    r = await client.get(f"/api/sessions/{sid}/report")
    assert r.status_code == 200
    assert r.json()["report"] == "# Done"


@pytest.mark.asyncio
async def test_events_stream_emits_and_closes(client):
    from app import main as main_module

    sid = "sess-sse"
    events: asyncio.Queue = asyncio.Queue()
    main_module.sessions[sid] = {"events": events, "status": "running"}

    async def pump():
        await events.put({"kind": "plan", "subtopics": ["a"]})
        await events.put({"kind": "_done"})

    asyncio.create_task(pump())

    async with client.stream("GET", f"/api/sessions/{sid}/events") as resp:
        assert resp.status_code == 200
        chunks = []
        async for line in resp.aiter_lines():
            chunks.append(line)
            if line.startswith("event: done"):
                break

    body = "\n".join(chunks)
    assert "plan" in body
    assert "event: done" in body
