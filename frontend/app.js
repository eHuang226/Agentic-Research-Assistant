const topicEl = document.getElementById("topic");
const logEl = document.getElementById("log");
const reportEl = document.getElementById("report");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const feedbackEl = document.getElementById("feedback");
const feedbackBtn = document.getElementById("feedbackBtn");

let es = null;
let sessionId = null;

function appendLog(kind, obj) {
  const ts = new Date().toISOString().slice(11, 23);
  const line = document.createElement("div");
  line.innerHTML = `<span class="ts">${ts}</span> <span class="kind">${kind}</span>\n${JSON.stringify(obj, null, 2)}\n`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function resetUi() {
  logEl.textContent = "";
  reportEl.textContent = "";
}

function setReportMarkdown(markdown) {
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    try {
      const html = marked.parse(markdown);
      reportEl.innerHTML = DOMPurify.sanitize(html);
      return;
    } catch (_) {
      /* fall through */
    }
  }
  reportEl.textContent = markdown;
}

startBtn.addEventListener("click", async () => {
  const topic = topicEl.value.trim();
  if (!topic) return;

  if (es) {
    es.close();
    es = null;
  }
  resetUi();
  startBtn.disabled = true;
  stopBtn.disabled = false;
  feedbackBtn.disabled = false;

  const res = await fetch("/api/research", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic }),
  });
  if (!res.ok) {
    appendLog("http_error", { status: res.status, text: await res.text() });
    startBtn.disabled = false;
    stopBtn.disabled = true;
    feedbackBtn.disabled = true;
    return;
  }
  const data = await res.json();
  sessionId = data.session_id;
  appendLog("session_started", data);

  es = new EventSource(`/api/sessions/${sessionId}/events`);
  es.onmessage = (ev) => {
    try {
      const payload = JSON.parse(ev.data);
      const k = payload.kind || "event";
      appendLog(k, payload);
      if (k === "final" && payload.report) {
        setReportMarkdown(payload.report);
      }
    } catch {
      appendLog("parse_error", { raw: ev.data });
    }
  };
  es.addEventListener("done", () => {
    appendLog("stream", { done: true });
    es.close();
    es = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    feedbackBtn.disabled = true;
  });
  es.onerror = () => {
    appendLog("sse_error", {});
    if (es) es.close();
    es = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };
});

stopBtn.addEventListener("click", () => {
  if (es) {
    es.close();
    es = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

feedbackBtn.addEventListener("click", async () => {
  const message = feedbackEl.value.trim();
  if (!sessionId || !message) return;
  const res = await fetch(`/api/sessions/${sessionId}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    appendLog("feedback_error", { status: res.status, text: await res.text() });
    return;
  }
  feedbackEl.value = "";
  appendLog("feedback_sent", { message });
});
