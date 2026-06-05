import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed: ${response.status}`);
  }
  return response.json();
}

function App() {
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [retrievalLogs, setRetrievalLogs] = useState([]);
  const [progress, setProgress] = useState({ value: 0, stage: "Waiting for upload" });
  const [messages, setMessages] = useState([
    makeMessage("assistant", "Upload a document, then ask a question about it."),
  ]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [copiedKey, setCopiedKey] = useState("");
  const fileInputRef = useRef(null);
  const messagesRef = useRef(null);

  const modelLabel = useMemo(() => {
    if (!health) return "Checking backend...";
    return `LLM mode: ${health.llm} | index ${health.index_session || "unknown"}`;
  }, [health]);

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  async function refreshAll(documentId = null) {
    const [nextHealth, nextDocuments, nextChunks, nextLogs] = await Promise.all([
      api("/api/health"),
      api("/api/documents"),
      api(documentId ? `/api/chunks?document_id=${documentId}` : "/api/chunks"),
      api("/api/retrieval-logs?limit=12"),
    ]);
    setHealth(nextHealth);
    setDocuments(nextDocuments);
    setChunks(nextChunks);
    setRetrievalLogs(nextLogs);
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("new_session", "true");
    setProgress({ value: 2, stage: "uploading" });
    const result = await api("/api/upload", { method: "POST", body: form });
    pollProgress(result.job_id);
  }

  function pollProgress(jobId) {
    const timer = window.setInterval(async () => {
      try {
        const job = await api(`/api/progress/${jobId}`);
        setProgress({ value: job.progress || 0, stage: job.stage || "working" });
        if (job.status === "complete") {
          window.clearInterval(timer);
          setMessages((items) => [
            ...items,
            makeMessage("assistant", `Processed ${job.filename}: ${job.chunk_count} chunks stored.`),
          ]);
          await refreshAll(job.document_id);
        }
        if (job.status === "error") {
          window.clearInterval(timer);
          setMessages((items) => [...items, makeMessage("assistant", `Upload failed: ${job.error}`)]);
        }
      } catch (error) {
        window.clearInterval(timer);
        setMessages((items) => [...items, makeMessage("assistant", error.message)]);
      }
    }, 450);
  }

  async function copyMessage(message, key) {
    await navigator.clipboard.writeText(message.text);
    setCopiedKey(key);
    window.setTimeout(() => setCopiedKey(""), 1200);
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    setQuestion("");
    setBusy(true);
    setMessages((items) => [...items, makeMessage("user", trimmed)]);
    try {
      const result = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      setMessages((items) => [
        ...items,
        makeMessage("assistant", result.answer, {
          sources: result.sources || [],
          quality: result.quality,
          confidence: result.confidence,
          retrieverRoute: result.retriever_route,
        }),
      ]);
      await refreshAll();
    } catch (error) {
      setMessages((items) => [...items, makeMessage("assistant", error.message)]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <section className="panel upload-panel">
          <div>
            <h1>Local RAG</h1>
            <p>{modelLabel}</p>
          </div>
          <button className="drop-zone" type="button" onClick={() => fileInputRef.current?.click()}>
            <span>Upload document</span>
            <small>PDF, DOCX, TXT, CSV, JSON, code, and more</small>
          </button>
          <input
            ref={fileInputRef}
            className="hidden-input"
            type="file"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) uploadFile(file).catch((error) => alert(error.message));
              event.target.value = "";
            }}
          />
          <div className="progress-wrap">
            <div className="progress-label">
              <span>{progress.stage}</span>
              <strong>{Math.round(progress.value)}%</strong>
            </div>
            <div className="progress-track">
              <div className="progress-bar" style={{ width: `${Math.round(progress.value)}%` }} />
            </div>
          </div>
        </section>

        <section className="panel docs-panel">
          <div className="panel-heading">
            <h2>Documents</h2>
            <button type="button" onClick={() => refreshAll()}>Refresh</button>
          </div>
          <div className="documents">
            {documents.length === 0 && <div className="empty">No documents yet.</div>}
            {documents.map((doc) => (
              <article className="doc-item" key={doc.id}>
                <strong>{doc.filename}</strong>
                <div className="meta">{doc.chunk_count} chunks</div>
                <button type="button" onClick={() => refreshAll(doc.id)}>View chunks</button>
              </article>
            ))}
          </div>
        </section>

        <section className="panel logs-panel">
          <div className="panel-heading">
            <h2>Retrieval Log</h2>
            <button type="button" onClick={() => refreshAll()}>Refresh</button>
          </div>
          <div className="retrieval-logs">
            {retrievalLogs.length === 0 && <div className="empty">No questions logged yet.</div>}
            {retrievalLogs.map((log) => (
              <details className="log-item" key={log.id}>
                <summary>
                  <span>{log.question}</span>
                  <strong>{Math.round((log.overall_score || 0) * 100)}%</strong>
                </summary>
                <div className="meta">
                  {log.grade || "ungraded"} | {log.source_count} sources | {formatTime(log.created_at)}
                </div>
                <div className="log-answer">{log.answer}</div>
                {!!log.payload?.sources?.length && (
                  <div className="log-sources">
                    {log.payload.sources.slice(0, 5).map((source) => (
                      <div className="proof-row" key={`${log.id}-${source.filename}-${source.chunk_index}`}>
                        #{source.chunk_index} | {source.table_title || source.section || "No section"} | score {source.score}
                        {source.contains_table ? " | table" : ""}
                      </div>
                    ))}
                  </div>
                )}
              </details>
            ))}
          </div>
        </section>
      </aside>

      <section className="chat-panel">
        <div className="messages" ref={messagesRef}>
          {messages.map((message, index) => (
            <article className={`message ${message.role}`} key={message.id || `${message.role}-${index}`}>
              <p>{message.text}</p>
              {message.confidence && <ConfidenceBadge confidence={message.confidence} />}
              {message.retrieverRoute && <RetrieverRoute route={message.retrieverRoute} />}
              {message.quality && <QualityReport quality={message.quality} />}
              {!!message.sources?.length && (
                <details className="sources-dropdown">
                  <summary>Retrieved sources ({message.sources.length})</summary>
                  <div className="sources">
                    {message.sources.map((source, sourceIndex) => (
                      <div className="source" key={`${source.filename}-${source.chunk_index}-${sourceIndex}`}>
                        <div className="source-title">
                          [{source.citation_id || `S${sourceIndex + 1}`}] {source.metadata?.section_title || "No section"} - score {source.score}
                        </div>
                        <div className="source-meta">
                          {source.filename}
                          {source.metadata?.page_start ? ` | p.${source.metadata.page_start}` : ""}
                          {source.metadata?.contains_table ? " | table" : ""}
                          {source.metadata?.table_title ? ` | ${source.metadata.table_title}` : ""}
                        </div>
                        {source.metadata?.contains_table && (
                          <div className="source-meta">
                            {source.metadata?.table_columns?.length
                              ? `Columns: ${source.metadata.table_columns.join(", ")}`
                              : "Table columns unavailable"}
                            {source.metadata?.table_row_count ? ` | rows: ${source.metadata.table_row_count}` : ""}
                          </div>
                        )}
                        {source.metadata?.contains_table && source.metadata?.table_rows?.length ? (
                          <div className="source-meta table-rows">
                            Rows: {source.metadata.table_rows.slice(0, 6).join(" ; ")}
                          </div>
                        ) : null}
                        <div className="source-text">{source.text}</div>
                      </div>
                    ))}
                  </div>
                </details>
              )}
              <div className="message-footer">
                <span>{formatTime(message.createdAt)}</span>
                <button
                  className="copy-button"
                  type="button"
                  title="Copy message"
                  aria-label="Copy message"
                  onClick={() => copyMessage(message, message.id || `${message.role}-${index}`)}
                >
                  <CopyIcon />
                </button>
                {copiedKey === (message.id || `${message.role}-${index}`) && <span>Copied</span>}
              </div>
            </article>
          ))}
          {busy && (
            <article className="message assistant">
              <p>Thinking...</p>
              <div className="message-footer">
                <span>{formatTime(new Date().toISOString())}</span>
              </div>
            </article>
          )}
        </div>
        <form className="chat-form" onSubmit={submitQuestion}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask about your uploaded documents..."
            autoComplete="off"
          />
          <button type="submit" disabled={busy}>Send</button>
        </form>
      </section>

      <aside className="chunks-panel">
        <div className="panel-heading">
          <h2>Chunks</h2>
          <a href="/api/chunks-file" download>Download</a>
        </div>
        <div className="chunks">
          {chunks.length === 0 && <div className="empty">Chunks will appear here after upload.</div>}
          {chunks.map((chunk) => (
            <article className="chunk-item" key={chunk.id}>
              <strong>{chunk.filename} #{chunk.chunk_index}</strong>
              <div className="meta">
                {chunk.metadata?.section_title || "No section"} | p.{chunk.metadata?.page_start || "?"}
                {chunk.metadata?.contains_table ? " | table" : ""}
                {chunk.metadata?.table_title ? ` | ${chunk.metadata.table_title}` : ""}
              </div>
              {chunk.metadata?.contains_table && chunk.metadata?.table_columns?.length ? (
                <div className="meta">Columns: {chunk.metadata.table_columns.join(", ")}</div>
              ) : null}
              {chunk.metadata?.contains_table && chunk.metadata?.table_rows?.length ? (
                <div className="meta table-rows">Rows: {chunk.metadata.table_rows.slice(0, 6).join(" ; ")}</div>
              ) : null}
              <div className="chunk-text">{chunk.text.slice(0, 900)}</div>
            </article>
          ))}
        </div>
      </aside>
    </main>
  );
}

function makeMessage(role, text, extra = {}) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    role,
    text,
    createdAt: new Date().toISOString(),
    ...extra,
  };
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function CopyIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none">
      <rect x="9" y="9" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="2" />
      <path d="M5 15V7a2 2 0 0 1 2-2h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function QualityReport({ quality }) {
  const checks = Object.entries(quality.checks || {});
  const profile = quality.question_profile;
  return (
    <details className="quality-dropdown">
      <summary>
        Answer quality: {quality.grade} ({Math.round((quality.overall_score || 0) * 100)}%)
      </summary>
      {profile && (
        <div className="quality-profile">
          <strong>{profile.label}</strong>
          <span>{profile.answer_template}</span>
        </div>
      )}
      <div className="quality-grid">
        {checks.map(([name, check]) => (
          <div className={`quality-check ${check.label}`} key={name}>
            <div className="quality-title">
              <span>{formatCheckName(name)}</span>
              <strong>{Math.round((check.score || 0) * 100)}%</strong>
            </div>
            <div className="quality-detail">{check.detail}</div>
          </div>
        ))}
      </div>
      {!!quality.retrieval_evidence?.length && (
        <div className="retrieval-proof">
          <strong>Retrieval proof</strong>
          {quality.retrieval_evidence.slice(0, 3).map((item) => (
            <div className="proof-row" key={`${item.filename}-${item.chunk_index}`}>
              {item.section} | #{item.chunk_index} | score {item.score}
              {item.contains_table ? " | table" : ""}
            </div>
          ))}
        </div>
      )}
    </details>
  );
}

function ConfidenceBadge({ confidence }) {
  return (
    <div className="confidence-badge">
      Confidence: {confidence.confidence || "unknown"}
      {confidence.top_score !== undefined ? ` | top score ${confidence.top_score}` : ""}
      {confidence.dominant_section ? ` | ${confidence.dominant_section}` : ""}
    </div>
  );
}

function RetrieverRoute({ route }) {
  const capabilities = route.active_capabilities || [];
  return (
    <details className="route-dropdown">
      <summary>Retriever route: {route.primary}</summary>
      <div className="route-body">
        <div>{route.route_notes}</div>
        <div>{(route.retrievers || []).join(" + ")}</div>
        {!!capabilities.length && (
          <div className="route-capabilities">
            {capabilities.map((capability) => (
              <div className="route-capability" key={capability.key}>
                <strong>{capability.label}</strong>
                <span>{capability.role || capability.signal}</span>
                <small>{capability.implemented_as}</small>
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}

function formatCheckName(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

createRoot(document.getElementById("root")).render(<App />);
