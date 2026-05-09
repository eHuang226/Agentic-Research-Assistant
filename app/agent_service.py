"""Plan → ReAct sub-research per subtopic (with feedback between steps) → RAG synthesis."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from app.tools import make_tools
from app.vector_store import query_context

logger = logging.getLogger(__name__)


def _model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _llm() -> ChatOpenAI:
    return ChatOpenAI(model=_model_name(), temperature=0.2)


class SSEBridgeHandler(BaseCallbackHandler):
    """Forward LangChain events to the SSE queue (sync LangChain callbacks → asyncio loop)."""

    def __init__(self, schedule_event):
        """schedule_event: sync (kind, payload) -> None; must schedule work on the main event loop."""
        self._schedule = schedule_event

    def _safe_emit(self, kind: str, payload: dict):
        try:
            self._schedule(kind, payload)
        except Exception:
            logger.debug("SSE bridge schedule failed", exc_info=True)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._safe_emit("tool_start", {"tool": name, "input_preview": (input_str or "")[:500]})

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        text = str(output) if output is not None else ""
        self._safe_emit("tool_end", {"output_preview": text[:800]})


async def _emit(events: asyncio.Queue, kind: str, payload: dict) -> None:
    await events.put({"kind": kind, **payload})


async def _drain_feedback(q: asyncio.Queue, max_items: int = 20) -> list[str]:
    msgs: list[str] = []
    for _ in range(max_items):
        try:
            msgs.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return msgs


async def plan_subtopics(topic: str, emit) -> list[str]:
    llm = _llm()
    prompt = (
        "You break a research topic into exactly 3 concrete sub-questions for web research. "
        "Return ONLY a JSON array of exactly 3 strings, no markdown.\nTopic: {topic}"
    )
    msg = await llm.ainvoke([HumanMessage(content=prompt.format(topic=topic))])
    text = (msg.content or "").strip()
    if "```" in text:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return [x.strip() for x in data if x.strip()][:3]
    except json.JSONDecodeError:
        pass
    # Fallback: split lines
    lines = [ln.strip("- •\t ") for ln in text.splitlines() if ln.strip()]
    return lines[:3] if lines else [topic]


def _build_subtopic_agent(tools, extra_context: str):
    llm = _llm()
    ctx = f"\n\nUser corrections / constraints:\n{extra_context}\n" if extra_context.strip() else ""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful research agent. Use tools to find and store grounded excerpts. "
                "Rules: (1) Prefer primary sources and reputable pages. (2) After reading a page, "
                "call store_research_chunk with a short faithful excerpt and the same URL. "
                "Do not invent URLs or quotes. If uncertain, search again or say so in Final Answer."
                + ctx,
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=3,
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )


async def synthesize_report(topic: str, collection, emit, user_notes: list[str]) -> str:
    await emit("synthesis", {"message": "Retrieving grounded context from vector store…"})
    chunks = query_context(collection, topic, n_results=12)
    if not chunks:
        return (
            "No stored research chunks were found. Run sub-research with successful store_research_chunk calls, "
            "or check network/tool errors in the logs."
        )
    context_blocks = []
    for i, c in enumerate(chunks):
        meta = c.get("metadata") or {}
        url = meta.get("source_url", "")
        title = meta.get("title", "")
        context_blocks.append(f"[{i + 1}] ({title}) {url}\n{c.get('text', '')}")
    context = "\n\n".join(context_blocks)
    notes = "\n".join(user_notes) if user_notes else "(none)"
    llm = _llm()
    prompt = f"""Write a structured research summary for the topic below.
Use ONLY the CONTEXT excerpts for factual claims. Every paragraph should cite sources like [1], [2] matching the bracket numbers in CONTEXT.
If CONTEXT is insufficient for a claim, omit it or say it is not supported by retrieved sources.
End with a "Sources" list of URLs from CONTEXT.

TOPIC: {topic}

USER FEEDBACK / CONSTRAINTS (honor when consistent with CONTEXT):
{notes}

CONTEXT:
{context}
"""
    msg = await llm.ainvoke([HumanMessage(content=prompt)])
    return (msg.content or "").strip()


async def run_research_pipeline(
    topic: str,
    session_id: str,
    collection,
    events: asyncio.Queue,
    feedback_queue: asyncio.Queue,
) -> str:
    loop = asyncio.get_running_loop()

    async def emit(kind: str, payload: dict | None = None):
        await _emit(events, kind, payload or {})

    await emit("session", {"session_id": session_id, "topic": topic})
    await emit("reasoning", {"message": "Planning sub-topics…"})

    subtopics = await plan_subtopics(topic, emit)
    await emit("plan", {"subtopics": subtopics, "message": f"Planned {len(subtopics)} research tracks."})

    user_feedback_accum: list[str] = []

    for idx, sub_q in enumerate(subtopics):
        corrections = await _drain_feedback(feedback_queue)
        if corrections:
            user_feedback_accum.extend(corrections)
            await emit("feedback_applied", {"messages": corrections, "at_subtopic_index": idx})

        extra = "\n".join(user_feedback_accum)

        def schedule_event(kind: str, payload: dict) -> None:
            coro = emit(kind, payload)
            try:
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                coro.close()

        def on_store(excerpt: str, source_url: str, title: str):
            coro = emit("stored", {"url": source_url, "title": title, "chars": len(excerpt)})
            try:
                asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                coro.close()

        tools = make_tools(collection, on_store=on_store)
        handler = SSEBridgeHandler(schedule_event)
        executor = _build_subtopic_agent(tools, extra_context=extra)

        await emit("subtopic_start", {"index": idx, "question": sub_q})

        def run_agent():
            return executor.invoke(
                {"input": f"Research and store grounded notes for: {sub_q}"},
                config={"callbacks": [handler]},
            )

        try:
            out = await asyncio.to_thread(run_agent)
            await emit(
                "subtopic_end",
                {"index": idx, "preview": str(out.get("output", ""))[:1200]},
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Subtopic agent failed")
            await emit("error", {"index": idx, "message": str(e)})

    final_notes = user_feedback_accum + await _drain_feedback(feedback_queue)
    report = await synthesize_report(topic, collection, emit, final_notes)
    await emit("final", {"report": report})
    return report


def new_session_id() -> str:
    return str(uuid.uuid4())
