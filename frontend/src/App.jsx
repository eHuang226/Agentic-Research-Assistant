import { useCallback, useEffect, useRef, useState } from "react";
import { marked } from "marked";
import DOMPurify from "dompurify";

function reportFromMarkdown(markdown) {
  try {
    return DOMPurify.sanitize(marked.parse(markdown));
  } catch {
    return "";
  }
}

export default function App() {
  const [topic, setTopic] = useState("");
  const [feedback, setFeedback] = useState("");
  const [logLines, setLogLines] = useState([]);
  const [reportHtml, setReportHtml] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [running, setRunning] = useState(false);
  const [streamOpen, setStreamOpen] = useState(false);

  const esRef = useRef(null);
  const logEndRef = useRef(null);

  const pushLog = useCallback((kind, obj) => {
    const ts = new Date().toISOString().slice(11, 23);
    const text = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    setLogLines((prev) => [...prev, { id: `${ts}-${prev.length}`, ts, kind, text }]);
  }, []);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines]);

  const disconnectStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
    setStreamOpen(false);
    setRunning(false);
  }, []);

  useEffect(() => () => disconnectStream(), [disconnectStream]);

  const resetUi = () => {
    setLogLines([]);
    setReportHtml("");
  };

  const startResearch = async () => {
    const t = topic.trim();
    if (!t) return;

    disconnectStream();
    resetUi();
    setRunning(true);
    setStreamOpen(true);

    const res = await fetch("/api/research", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic: t }),
    });

    if (!res.ok) {
      pushLog("http_error", { status: res.status, text: await res.text() });
      setRunning(false);
      setStreamOpen(false);
      return;
    }

    const data = await res.json();
    setSessionId(data.session_id);
    pushLog("session_started", data);

    const es = new EventSource(`/api/sessions/${data.session_id}/events`);
    esRef.current = es;

    es.onmessage = (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        const k = payload.kind || "event";
        pushLog(k, payload);
        if (k === "final" && payload.report) {
          const html = reportFromMarkdown(payload.report);
          if (html) setReportHtml(html);
          else setReportHtml(`<p>${DOMPurify.sanitize(payload.report)}</p>`);
        }
      } catch {
        pushLog("parse_error", { raw: ev.data });
      }
    };

    es.addEventListener("done", () => {
      pushLog("stream", { done: true });
      es.close();
      esRef.current = null;
      setStreamOpen(false);
      setRunning(false);
    });

    es.onerror = () => {
      pushLog("sse_error", {});
      if (esRef.current) esRef.current.close();
      esRef.current = null;
      setStreamOpen(false);
      setRunning(false);
    };
  };

  const sendFeedback = async () => {
    const message = feedback.trim();
    if (!sessionId || !message) return;
    const res = await fetch(`/api/sessions/${sessionId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok) {
      pushLog("feedback_error", { status: res.status, text: await res.text() });
      return;
    }
    setFeedback("");
    pushLog("feedback_sent", { message });
  };

  return (
    <>
      <header className="header">
        <h1>Agentic Research Assistant</h1>
        <p className="sub">Live agent logs · vector-grounded synthesis · mid-run feedback</p>
      </header>

      <main className="layout">
        <section className="panel controls">
          <label htmlFor="topic">Research topic</label>
          <textarea
            id="topic"
            rows={3}
            placeholder="e.g. The future of silicon photonics"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />
          <div className="row">
            <button type="button" onClick={startResearch} disabled={running}>
              Start research
            </button>
            <button type="button" className="secondary" onClick={disconnectStream} disabled={!streamOpen}>
              Disconnect stream
            </button>
          </div>
          <label htmlFor="feedback">Correct the agent mid-run</label>
          <textarea
            id="feedback"
            rows={2}
            placeholder="e.g. Focus on datacenter interconnects, ignore consumer optics"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
          />
          <button
            type="button"
            className="secondary"
            onClick={sendFeedback}
            disabled={!sessionId || !running}
          >
            Send feedback
          </button>
          <p className="hint">
            Feedback is applied before the next sub-topic. Requires <code>OPENAI_API_KEY</code> on the server.
            <br />
            <span className="hint-dev">API from repo root: </span>
            <code>python run_server.py</code>
            <span className="hint-dev"> or </span>
            <code>python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000</code>
            <br />
            <span className="hint-dev">From </span>
            <code>frontend/</code>
            <span className="hint-dev">: </span>
            <code>npm run backend</code>
            <span className="hint-dev"> · Dev UI: </span>
            <code>npm run dev</code>
            <span className="hint-dev"> · Prod: </span>
            <code>npm run build</code>
          </p>
        </section>

        <section className="panel log-panel">
          <div className="log-header">Agent log</div>
          <pre className="log">
            {logLines.map((line) => (
              <div key={line.id}>
                <span className="ts">{line.ts}</span> <span className="kind">{line.kind}</span>
                {"\n"}
                {line.text}
                {"\n"}
              </div>
            ))}
            <span ref={logEndRef} />
          </pre>
        </section>

        <section className="panel report-panel">
          <div className="log-header">Final report</div>
          <article className="report" dangerouslySetInnerHTML={{ __html: reportHtml }} />
        </section>
      </main>
    </>
  );
}
