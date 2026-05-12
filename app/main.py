"""FastAPI: research sessions, SSE log stream, mid-run feedback queue."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent_service import new_session_id, run_research_pipeline
from app.schemas import FeedbackRequest, ResearchRequest, ResearchResponse
from app.vector_store import VectorStoreClient

load_dotenv()

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"

vs_client = VectorStoreClient()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await vs_client.connect()
    yield
    await vs_client.close()


app = FastAPI(title="Agentic Research Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> { events: Queue, feedback: Queue, task: Task | None, status: str }
sessions: dict[str, dict] = {}


@app.post("/api/research", response_model=ResearchResponse)
async def start_research(body: ResearchRequest):
    sid = body.session_id or new_session_id()
    if sid in sessions and sessions[sid].get("status") == "running":
        raise HTTPException(status_code=409, detail="Session already running")

    events: asyncio.Queue = asyncio.Queue()
    feedback: asyncio.Queue = asyncio.Queue()

    sessions[sid] = {
        "events": events,
        "feedback": feedback,
        "status": "running",
        "topic": body.topic,
    }

    async def job():
        try:
            report = await run_research_pipeline(body.topic, sid, vs_client, events, feedback)
            sessions[sid]["report"] = report
        except Exception as e:  # noqa: BLE001
            logger.exception("Research pipeline failed for session %s", sid)
            await events.put({"kind": "error", "message": str(e)})
        finally:
            sessions[sid]["status"] = "done"
            await events.put({"kind": "_done"})

    asyncio.create_task(job())
    return ResearchResponse(session_id=sid, status="started")


@app.post("/api/sessions/{session_id}/feedback")
async def post_feedback(session_id: str, body: FeedbackRequest):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if s.get("status") != "running":
        raise HTTPException(status_code=400, detail="Session not accepting feedback (not running)")
    await s["feedback"].put(body.message)
    await s["events"].put({"kind": "feedback_received", "message": body.message})
    return {"ok": True}


@app.get("/api/sessions/{session_id}/report")
async def get_report(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if "report" not in s:
        raise HTTPException(status_code=202, detail="Report not ready yet")
    return {"session_id": session_id, "report": s["report"]}


@app.get("/api/sessions/{session_id}/events")
async def session_events(session_id: str):
    s = sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")

    events: asyncio.Queue = s["events"]

    async def gen():
        while True:
            item = await events.get()
            if item.get("kind") == "_done":
                yield "event: done\ndata: {}\n\n"
                break
            line = json.dumps(item, ensure_ascii=False)
            yield f"data: {line}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


_assets_dir = FRONTEND_DIST / "assets"
if _assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="vite_assets")


@app.get("/")
async def root_index():
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "message": "Frontend build not found. From frontend/: npm install && npm run build — or npm run dev with Vite proxy.",
        "docs": "/docs",
    }
