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
from app.vector_store import get_client, get_collection

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).resolve().parent.parent / "chroma_data"))
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

chroma_client = get_client(CHROMA_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    # Persistent Chroma client needs no explicit close in most versions


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
    collection = get_collection(chroma_client, sid)

    sessions[sid] = {
        "events": events,
        "feedback": feedback,
        "status": "running",
        "topic": body.topic,
    }

    async def job():
        try:
            await run_research_pipeline(body.topic, sid, collection, events, feedback)
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


if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root_index():
    index = FRONTEND_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Frontend not found; open /docs for API."}
