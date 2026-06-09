import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bug,
  Database,
  Download,
  FileText,
  FolderOpen,
  History,
  LoaderCircle,
  MessageSquare,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRight,
  Plus,
  RefreshCw,
  Sun,
  ArrowUp,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
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
  const [chatSessions, setChatSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [progress, setProgress] = useState({ value: 0, stage: "Waiting for upload" });
  const [showUploadProgress, setShowUploadProgress] = useState(false);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [draftQuestion, setDraftQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [copiedKey, setCopiedKey] = useState("");
  const [processingStartedAt, setProcessingStartedAt] = useState(null);
  const [processingElapsedMs, setProcessingElapsedMs] = useState(0);
  const [debugMode, setDebugMode] = useState(false);
  const [theme, setTheme] = useState("light");
  const [activeView, setActiveView] = useState("chat");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const fileInputRef = useRef(null);
  const messagesRef = useRef(null);

  const documentCount = documents.length;
  const chunkCount = chunks.length;

  const modelLabel = useMemo(() => {
    if (!health) return "Checking backend";
    const model = compactModelStatus(health.llm);
    const index = health.documents !== undefined ? `${health.documents} docs` : "index unknown";
    return `${health.ok ? "Backend online" : "Backend offline"} · ${model} · ${index}`;
  }, [health]);

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    if (messagesRef.current) {
      messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (!busy || !processingStartedAt) return undefined;
    const timer = window.setInterval(() => {
      setProcessingElapsedMs(Date.now() - processingStartedAt);
    }, 160);
    return () => window.clearInterval(timer);
  }, [busy, processingStartedAt]);

  async function refreshAll(documentId = null) {
    const [nextHealth, nextDocuments, nextChunks, nextLogs, nextChatSessions] = await Promise.all([
      api("/api/health"),
      api("/api/documents"),
      api(documentId ? `/api/chunks?document_id=${documentId}` : "/api/chunks"),
      api("/api/retrieval-logs?limit=12"),
      api("/api/chat-sessions?limit=40"),
    ]);
    setHealth(nextHealth);
    setDocuments(nextDocuments);
    setChunks(nextChunks);
    setRetrievalLogs(nextLogs);
    setChatSessions(nextChatSessions);
  }

  async function uploadFile(file) {
    const form = new FormData();
    form.append("file", file);
    form.append("new_session", "true");
    setShowUploadProgress(true);
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
          window.setTimeout(() => setShowUploadProgress(false), 2500);
        }
        if (job.status === "error") {
          window.clearInterval(timer);
          setMessages((items) => [...items, makeMessage("assistant", `Upload failed: ${job.error}`)]);
          window.setTimeout(() => setShowUploadProgress(false), 3000);
        }
      } catch (error) {
        window.clearInterval(timer);
        setMessages((items) => [...items, makeMessage("assistant", error.message)]);
        window.setTimeout(() => setShowUploadProgress(false), 3000);
      }
    }, 450);
  }

  async function copyMessage(message, key) {
    await navigator.clipboard.writeText(message.text);
    setCopiedKey(key);
    window.setTimeout(() => setCopiedKey(""), 1200);
  }

  async function copyCitation(source, key) {
    await navigator.clipboard.writeText(citationReference(source));
    setCopiedKey(key);
    window.setTimeout(() => setCopiedKey(""), 1200);
  }

  async function copySourceEvidence(source, key) {
    await navigator.clipboard.writeText(sourceEvidenceText(source));
    setCopiedKey(key);
    window.setTimeout(() => setCopiedKey(""), 1200);
  }

  async function copyFullResponse(message, key) {
    await navigator.clipboard.writeText(fullResponseText(message));
    setCopiedKey(key);
    window.setTimeout(() => setCopiedKey(""), 1200);
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || busy) return;
    setHistoryIndex(-1);
    setDraftQuestion("");
    if (documentCount === 0) {
      setMessages((items) => [
        ...items,
        makeMessage("user", trimmed),
        makeMessage("assistant", "No indexed documents are available. Upload and index a document before asking document-grounded questions.", {
          tone: "empty",
        }),
      ]);
      setQuestion("");
      return;
    }
    const requestedAt = performance.now();
    setQuestion("");
    setBusy(true);
    setProcessingStartedAt(Date.now());
    setProcessingElapsedMs(0);
    setMessages((items) => [...items, makeMessage("user", trimmed)]);
    try {
      const result = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, session_id: activeSessionId }),
      });
      if (result.session_id) setActiveSessionId(result.session_id);
      const clientElapsedMs = Math.round(performance.now() - requestedAt);
      setMessages((items) => [
        ...items,
        makeMessage("assistant", result.answer, {
          sources: result.sources || [],
          quality: result.quality,
          confidence: result.confidence,
          retrieverRoute: result.retriever_route,
          retrieval: result.retrieval
            ? {
                ...result.retrieval,
                timing: {
                  ...(result.retrieval.timing || {}),
                  client_total_ms: clientElapsedMs,
                },
              }
            : null,
        }),
      ]);
      await refreshAll();
    } catch (error) {
      setMessages((items) => [...items, makeMessage("assistant", `Retrieval failed: ${error.message}`, { tone: "error" })]);
    } finally {
      setBusy(false);
      setProcessingStartedAt(null);
    }
  }

  function startNewChat() {
    setMessages([]);
    setActiveSessionId(null);
    setQuestion("");
    setHistoryIndex(-1);
    setDraftQuestion("");
    setCopiedKey("");
    setProcessingElapsedMs(0);
    setActiveView("chat");
  }

  async function openChatSession(sessionId) {
    if (!sessionId || busy) return;
    const session = await api(`/api/chat-sessions/${encodeURIComponent(sessionId)}`);
    setActiveSessionId(session.id);
    setMessages(messagesFromSession(session));
    setQuestion("");
    setHistoryIndex(-1);
    setDraftQuestion("");
    setCopiedKey("");
    setProcessingElapsedMs(0);
    setActiveView("chat");
  }

  return (
    <main className={`app-shell ${activeView !== "chat" ? "page-mode" : ""}${sidebarCollapsed ? " sidebar-collapsed" : ""}`}>
      <header className="app-header">
        <div className="brand-block">
          <div className="brand-mark">R</div>
          <div>
            <h1>RAG Chat</h1>
          </div>
        </div>
        <WorkspaceNav activeView={activeView} setActiveView={setActiveView} />
        <div className="header-actions topbar-actions">
          <span
            className={`status-dot ${!health ? "checking" : health.ok ? "online" : "offline"}`}
            title={health?.index_session ? `Index session ${health.index_session}` : modelLabel}
            aria-label={!health ? "Backend checking" : health.ok ? "Backend online" : "Backend offline"}
          />
          <button
            className="theme-toggle"
            type="button"
            title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
            onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
          </button>
        </div>
      </header>

      {activeView === "chat" ? (
      <>
      <aside className="sidebar workspace-sidebar">
        <div className="sidebar-top-actions">
          <button className="new-chat-button" type="button" onClick={startNewChat} title="New chat" aria-label="New chat">
            <Plus size={16} />
            <span>New chat</span>
          </button>
          <button
            className="collapse-sidebar-button"
            type="button"
            onClick={() => setSidebarCollapsed((current) => !current)}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
        </div>
        <section className="panel upload-panel compact-upload-panel">
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
        </section>
        <section className="panel chat-history-panel">
          <div className="panel-heading">
            <h2>Chat history</h2>
            <button type="button" onClick={() => refreshAll()} title="Refresh chat history" aria-label="Refresh chat history">
              <RefreshCw size={14} />
            </button>
          </div>
          <div className="chat-history-list">
            {chatSessions.length === 0 && (
              <div className="chat-history-empty">No saved chats yet.</div>
            )}
            {chatSessions.map((session) => (
              <button
                className={`chat-history-item${session.id === activeSessionId ? " active" : ""}`}
                type="button"
                key={session.id}
                onClick={() => openChatSession(session.id).catch((error) => alert(error.message))}
              >
                <MessageSquare size={14} />
                <span>
                  <strong>{session.title || "New chat"}</strong>
                  <small>
                    {session.message_count || 0} messages | {formatTime(session.updated_at)}
                    {session.active_index === false ? " | old index" : ""}
                  </small>
                </span>
              </button>
            ))}
          </div>
        </section>
      </aside>

      <section className="chat-panel">
        <div className="chat-header">
          <div>
            <h2>Engineering assistant</h2>
            <p>Concise, cited answers from your indexed documents.</p>
          </div>
          <div className="chat-header-status">
            <span>{health?.ok ? "Backend online" : "Checking backend"}</span>
          </div>
        </div>
        <div className="messages-viewport" ref={messagesRef}>
          <div className="messages">
          {messages.map((message, index) => {
            const insufficient = message.role === "assistant" && isInsufficientEvidence(message);
            return (
              <article
                className={`message ${message.role}${insufficient ? " insufficient-evidence-message" : ""}${message.tone ? ` ${message.tone}-message` : ""}`}
                key={message.id || `${message.role}-${index}`}
              >
                {insufficient && <InsufficientEvidenceNotice message={message} />}
                <AnswerText text={message.text} sources={message.sources || []} />
                
                {debugMode && message.retrieval && <RetrievalTransparency retrieval={message.retrieval} />}
                {debugMode && <DebugDetails message={message} />}
                {debugMode && message.retrieverRoute && <RetrieverRoute route={message.retrieverRoute} />}
                {debugMode && message.quality && <QualityReport quality={message.quality} />}
                
                {message.role === "assistant" && (
                  <div className="message-footer assistant-actions">
                    <span>{formatTime(message.createdAt)}</span>
                    <button
                      className="copy-button"
                      type="button"
                      title="Copy answer"
                      aria-label="Copy answer"
                      onClick={() => copyMessage(message, message.id || `${message.role}-${index}`)}
                    >
                      <CopyIcon />
                    </button>
                    <button
                      className="copy-button"
                      type="button"
                      title="Copy full response with sources"
                      aria-label="Copy full response with sources"
                      onClick={() => copyFullResponse(message, fullResponseCopyKey(message, index))}
                    >
                      <CopyAllIcon />
                    </button>
                    {copiedKey === (message.id || `${message.role}-${index}`) && <span>Copied</span>}
                    {copiedKey === fullResponseCopyKey(message, index) && <span>Full response copied</span>}

                    {!insufficient && (message.confidence || message.retrieval) && (
                      <ConfidenceBadge confidence={message.confidence || {}} retrieval={message.retrieval || {}} />
                    )}
                    
                    {!!message.sources?.length && (
                      <details className="sources-dropdown">
                        <summary>Sources ({message.sources.length})</summary>
                        <div className="sources">
                          {message.sources.map((source, sourceIndex) => (
                            <SourceCard
                              key={`${source.filename}-${source.chunk_index}-${sourceIndex}`}
                              source={source}
                              fallbackId={`S${sourceIndex + 1}`}
                              citationCopied={copiedKey === citationCopyKey(source, sourceIndex)}
                              evidenceCopied={copiedKey === evidenceCopyKey(source, sourceIndex)}
                              onCopyCitation={() => copyCitation(source, citationCopyKey(source, sourceIndex))}
                              onCopyEvidence={() => copySourceEvidence(source, evidenceCopyKey(source, sourceIndex))}
                              debugMode={debugMode}
                            />
                          ))}
                        </div>
                      </details>
                    )}

                    <div style={{ flexGrow: 1 }} />
                  </div>
                )}
              </article>
            );
          })}
          {busy && (
            <article className="message assistant">
              <ProcessingStatus elapsedMs={processingElapsedMs} />
              <div className="message-footer">
                <span>{formatTime(new Date().toISOString())}</span>
              </div>
            </article>
          )}
          </div>
        </div>
        <div className={`input-area${messages.length === 0 && !busy ? " centered-composer" : ""}`}>
          {messages.length === 0 && !busy && (
            <div className="empty-chat-greeting" aria-label="Greeting">
              <span>{timeGreeting()}</span>
              <strong>Hariesh Kai</strong>
            </div>
          )}
          <form className={`chat-form${busy ? " busy" : ""}`} onSubmit={submitQuestion}>
            <button
              className="composer-upload-button"
              type="button"
              title="Upload document"
              aria-label="Upload document"
              onClick={() => fileInputRef.current?.click()}
              disabled={busy}
            >
              <Plus size={20} />
            </button>
            <input
              value={question}
              onChange={(event) => {
                setQuestion(event.target.value);
                setHistoryIndex(-1);
                setDraftQuestion(event.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === "ArrowUp" || e.key === "ArrowDown") {
                  if (e.target.selectionStart !== e.target.selectionEnd) return;
                  if (e.target.selectionStart > 0 && e.target.selectionStart < e.target.value.length) return;
                  
                  const userQuestions = messages.filter(m => m.role === "user").map(m => m.text);
                  if (userQuestions.length === 0) return;

                  e.preventDefault();

                  if (e.key === "ArrowUp") {
                    if (historyIndex === -1) {
                      const nextIndex = userQuestions.length - 1;
                      setHistoryIndex(nextIndex);
                      setQuestion(userQuestions[nextIndex]);
                    } else if (historyIndex > 0) {
                      const nextIndex = historyIndex - 1;
                      setHistoryIndex(nextIndex);
                      setQuestion(userQuestions[nextIndex]);
                    }
                  } else if (e.key === "ArrowDown") {
                    if (historyIndex !== -1) {
                      if (historyIndex < userQuestions.length - 1) {
                        const nextIndex = historyIndex + 1;
                        setHistoryIndex(nextIndex);
                        setQuestion(userQuestions[nextIndex]);
                      } else {
                        setHistoryIndex(-1);
                        setQuestion(draftQuestion);
                      }
                    }
                  }
                }
              }}
              placeholder={documentCount ? "Message RAG Chat..." : "Upload a document before asking..."}
              autoComplete="off"
              disabled={busy}
            />
            <button
              type="submit"
              disabled={busy || !question.trim()}
              title={busy ? "Processing" : "Send message"}
              aria-label={busy ? "Processing question" : "Send message"}
            >
              {busy ? <LoaderCircle className="send-spinner" size={17} /> : <ArrowUp size={19} />}
            </button>
          </form>
          <div className="input-scope">
            <span>{documentCount} documents</span>
            <span>{chunkCount} chunks</span>
            {showUploadProgress && (
            <div className="composer-progress">
              <div className="progress-label">
                <span>{progress.stage}</span>
                <strong>{Math.round(progress.value)}%</strong>
              </div>
              <div className="progress-track">
                <div className="progress-bar" style={{ width: `${Math.round(progress.value)}%` }} />
              </div>
            </div>
            )}
          </div>
        </div>
      </section>

      </>
      ) : (
        <WorkspacePage
          activeView={activeView}
          documents={documents}
          chunks={chunks}
          retrievalLogs={retrievalLogs}
          refreshAll={refreshAll}
          debugMode={debugMode}
          setActiveView={setActiveView}
        />
      )}
    </main>
  );
}

function WorkspaceNav({ activeView, setActiveView }) {
  return (
    <nav className="app-nav topbar-nav" aria-label="Workspace sections">
      <button className={activeView === "chat" ? "active" : ""} type="button" onClick={() => setActiveView("chat")}>
        <MessageSquare size={15} />
        <span>Chat</span>
      </button>
      <button className={activeView === "documents" ? "active" : ""} type="button" onClick={() => setActiveView("documents")}>
        <FolderOpen size={15} />
        <span>Documents</span>
      </button>
      <button className={activeView === "evidence" ? "active" : ""} type="button" onClick={() => setActiveView("evidence")}>
        <PanelRight size={15} />
        <span>Evidence</span>
      </button>
      <button className={activeView === "logs" ? "active" : ""} type="button" onClick={() => setActiveView("logs")}>
        <History size={15} />
        <span>Logs</span>
      </button>
    </nav>
  );
}

function WorkspacePage({ activeView, documents, chunks, retrievalLogs, refreshAll, setActiveView }) {
  if (activeView === "documents") {
    return (
      <section className="workspace-page">
        <PageHeader
          icon={<FolderOpen size={18} />}
          title="Documents"
          description="Indexed source documents available to the chatbot."
          action={<button type="button" onClick={() => refreshAll()}><RefreshCw size={15} /> Refresh</button>}
        />
        <div className="page-grid documents-page-grid">
          {documents.length === 0 && <div className="page-empty">No indexed documents.</div>}
          {documents.map((doc) => (
            <article className="document-card" key={doc.id}>
              <div>
                <strong>{doc.filename}</strong>
                <span>{doc.chunk_count} chunks</span>
              </div>
              <button type="button" onClick={() => refreshAll(doc.id)}>View evidence</button>
            </article>
          ))}
        </div>
      </section>
    );
  }

  if (activeView === "evidence") {
    return (
      <section className="workspace-page">
        <PageHeader
          icon={<PanelRight size={18} />}
          title="Evidence"
          description="Document chunks, table rows, sections, and pages used by retrieval."
          action={<a href="/api/chunks-file" download><Download size={15} /> Download</a>}
        />
        <div className="evidence-page-list">
          {chunks.length === 0 && <div className="page-empty">No evidence chunks loaded.</div>}
          {chunks.map((chunk) => (
            <article className="evidence-row" key={chunk.id}>
              <div className="evidence-row-header">
                <strong>{chunk.filename} #{chunk.chunk_index}</strong>
                <span>{chunk.metadata?.section_title || "No section"} | p.{chunk.metadata?.page_start || "?"}</span>
              </div>
              <div className="source-tags">
                {chunk.metadata?.contains_table ? <span>table</span> : <span>text</span>}
                {chunk.metadata?.table_title ? <span>{chunk.metadata.table_title}</span> : null}
                {chunk.metadata?.table_row_count ? <span>{chunk.metadata.table_row_count} rows</span> : null}
              </div>
              {chunk.metadata?.contains_table && chunk.metadata?.table_columns?.length ? (
                <div className="source-meta">Columns: {chunk.metadata.table_columns.join(", ")}</div>
              ) : null}
              {chunk.metadata?.contains_table && chunk.metadata?.table_rows?.length ? (
                <div className="evidence-table-rows">
                  {chunk.metadata.table_rows.slice(0, 10).map((row, index) => (
                    <div key={`${chunk.id}-row-${index}`}>{row}</div>
                  ))}
                </div>
              ) : null}
              <div className="chunk-text">{chunk.text.slice(0, 1400)}</div>
            </article>
          ))}
        </div>
      </section>
    );
  }

  return (
    <section className="workspace-page">
      <PageHeader
        icon={<History size={18} />}
        title="Retrieval Log"
        description="Recent questions, answer quality, and retrieved source traces."
        action={<button type="button" onClick={() => refreshAll()}><RefreshCw size={15} /> Refresh</button>}
      />
      <div className="logs-page-list">
        {retrievalLogs.length === 0 && <div className="page-empty">No retrieval logs yet.</div>}
        {retrievalLogs.map((log) => (
          <article className="log-card" key={log.id}>
            <div className="log-card-header">
              <strong>{log.question}</strong>
              <span>{Math.round((log.overall_score || 0) * 100)}%</span>
            </div>
            <div className="meta">{log.grade || "ungraded"} | {log.source_count} sources | {formatTime(log.created_at)}</div>
            <div className="log-answer">{log.answer}</div>
            {!!log.payload?.sources?.length && (
              <div className="log-source-list">
                {log.payload.sources.slice(0, 8).map((source) => (
                  <button type="button" key={`${log.id}-${source.filename}-${source.chunk_index}`} onClick={() => setActiveView("evidence")}>
                    #{source.chunk_index} | {source.table_title || source.section || "No section"} | score {source.score}
                  </button>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function PageHeader({ icon, title, description, action }) {
  return (
    <div className="page-header">
      <div>
        <span className="page-icon">{icon}</span>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      <div className="page-actions">{action}</div>
    </div>
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

function messagesFromSession(session) {
  const rows = session.messages || [];
  if (!rows.length) {
    return [makeMessage("assistant", "This chat has no messages yet.")];
  }
  return rows.map((row) => {
    const payload = row.payload || {};
    return makeMessage(row.role, row.text, {
      id: `saved-${row.id}`,
      createdAt: row.created_at,
      sources: payload.sources || [],
      quality: payload.quality,
      confidence: payload.confidence,
      retrieverRoute: payload.retriever_route,
      retrieval: payload.retrieval,
      tone: payload.tone,
    });
  });
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function timeGreeting(date = new Date()) {
  const hour = date.getHours();
  if (hour >= 5 && hour < 12) return "Good Morning,";
  if (hour >= 12 && hour < 17) return "Good Afternoon,";
  if (hour >= 17 && hour < 21) return "Good Evening,";
  return "Good Night,";
}

function compactModelStatus(value) {
  const text = String(value || "").trim();
  if (!text) return "model unknown";
  if (text.toLowerCase().includes("llama-cpp")) return "local model";
  if (text.toLowerCase().includes("extractive")) return "extractive mode";
  if (text.toLowerCase().includes("loads on first chat")) return "model ready";
  return text.length > 34 ? `${text.slice(0, 34).trim()}...` : text;
}

function CopyIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none">
      <rect x="9" y="9" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="2" />
      <path d="M5 15V7a2 2 0 0 1 2-2h8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function CopyAllIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="15" height="15" fill="none">
      <rect x="7" y="7" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="2" />
      <path d="M4 13V6a2 2 0 0 1 2-2h7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M11 20h7a2 2 0 0 0 2-2v-7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function ChatEmptyState() {
  return (
    <section className="chat-empty-state" aria-label="No indexed documents">
      <strong>No document context indexed</strong>
      <span>Upload a document to enable grounded answers, citations, and retrieval evidence.</span>
    </section>
  );
}

function InsufficientEvidenceNotice({ message }) {
  const retrieval = message.retrieval || {};
  const quality = message.quality || {};
  const route = retrieval.primary_route || retrieval.route?.primary || message.retrieverRoute?.primary || "unknown";
  const notFound = isNotFoundAnswer(message.text);
  const sourceCount = Number(retrieval.source_count ?? message.sources?.length ?? 0);
  const reason = notFound
    ? "The retrieved document context did not support an answer."
    : sourceCount === 0
      ? "No usable source chunks were retrieved."
      : "The retrieved evidence was below the answer threshold.";

  return (
    <section className="not-found-panel" aria-label="Insufficient evidence">
      <div className="not-found-heading">
        <strong>{notFound ? "Not found in retrieved context" : "Insufficient evidence"}</strong>
        <span>{reason}</span>
      </div>
      <div className="not-found-metrics">
        <span>{sourceCount} sources</span>
        <span>Top {formatScore(retrieval.top_score)}</span>
        <span>{formatRouteName(route)}</span>
        {quality.grade ? <span>{quality.grade} {formatScore(quality.overall_score)}</span> : null}
      </div>
      <details className="not-found-details">
        <summary>Trace</summary>
        <div>
          {retrieval.retrieved_query ? <span>Query: {retrieval.retrieved_query}</span> : null}
          {retrieval.dominant_section ? <span>Section: {retrieval.dominant_section}</span> : null}
          <span>Decision: answer withheld unless retrieved evidence directly supports it.</span>
        </div>
      </details>
    </section>
  );
}

function ProcessingStatus({ elapsedMs }) {
  const activeIndex = processingStageIndex(elapsedMs);
  return (
    <section className="processing-status" aria-label="Processing status">
      <div className="processing-heading">
        <strong>{PROCESSING_STAGES[Math.min(activeIndex, PROCESSING_STAGES.length - 1)].label}</strong>
        <span>{formatMs(elapsedMs)}</span>
      </div>
      <div className="processing-steps">
        {PROCESSING_STAGES.map((stage, index) => (
          <div
            className={`processing-step ${index < activeIndex ? "done" : ""}${index === activeIndex ? " active" : ""}`}
            key={stage.label}
          >
            <span />
            <div>
              <strong>{stage.label}</strong>
              <small>{stage.detail}</small>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function DebugDetails({ message }) {
  const retrieval = message.retrieval || {};
  const route = message.retrieverRoute || retrieval.route || {};
  const quality = message.quality || {};
  const profile = message.quality?.question_profile || retrieval.question_type;
  const timing = retrieval.timing || {};
  const sourceDebugCount = (message.sources || []).filter((source) => source.debug).length;

  return (
    <details className="debug-panel">
      <summary>Debug details</summary>
      <div className="debug-grid">
        <DebugField label="Question profile" value={profile?.label || profile?.type_id || "unknown"} />
        <DebugField label="Answer template" value={profile?.answer_template || "none"} />
        <DebugField label="Primary route" value={route.primary || retrieval.primary_route || "unknown"} />
        <DebugField label="Active retrievers" value={(route.retrievers || retrieval.active_retrievers || []).join(" + ") || "none"} />
        <DebugField label="Quality" value={`${quality.grade || "ungraded"} ${formatScore(quality.overall_score)}`} />
        <DebugField label="Evidence" value={retrieval.evidence_label || retrieval.evidence_status || "unknown"} />
        <DebugField label="Retrieval time" value={formatMs(timing.retrieval_ms)} />
        <DebugField label="Total time" value={formatMs(timing.client_total_ms || timing.total_ms)} />
        <DebugField label="Debug sources" value={`${sourceDebugCount}/${message.sources?.length || 0}`} />
      </div>
      {!!quality.checks && (
        <div className="debug-section">
          <strong>Quality checks</strong>
          <DebugKeyValues value={Object.fromEntries(Object.entries(quality.checks).map(([key, check]) => [key, `${formatScore(check.score)} ${check.label || ""}`]))} />
        </div>
      )}
      {!!retrieval.source_signals && (
        <div className="debug-section">
          <strong>Source signals</strong>
          <DebugKeyValues value={retrieval.source_signals} />
        </div>
      )}
      {!!route.active_capabilities?.length && (
        <div className="debug-section">
          <strong>Capabilities</strong>
          <div className="debug-lines">
            {route.active_capabilities.map((capability) => (
              <span key={capability.key}>{capability.label}: {capability.implemented_as}</span>
            ))}
          </div>
        </div>
      )}
    </details>
  );
}

function DebugField({ label, value }) {
  return (
    <div className="debug-field">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AnswerText({ text }) {
  // Completely strip out [S1], [S1, S2], or [S1 | verbose text]
  const cleanedText = String(text || "").replace(/\[S\d+[^\]]*\]/g, "").trim();

  return (
    <div className="answer-text markdown-body">
      <ReactMarkdown>
        {cleanedText}
      </ReactMarkdown>
    </div>
  );
}

function isInsufficientEvidence(message) {
  return (
    isNotFoundAnswer(message.text)
    || message.retrieval?.evidence_status === "not_enough"
    || Boolean(message.retrieval) && Number(message.retrieval?.source_count || 0) === 0 && message.role === "assistant"
  );
}

function isNotFoundAnswer(text) {
  const normalized = String(text || "").trim().toLowerCase();
  return (
    normalized === "not found in the retrieved document context."
    || /^not found(?:\s*\[s\d+\])?\.?$/.test(normalized)
  );
}

function SourceCard({ source, fallbackId, citationCopied, evidenceCopied, onCopyCitation, onCopyEvidence, debugMode }) {
  const citation = source.citation_id || fallbackId;
  const metadata = source.metadata || {};
  const title = source.display_title || metadata.table_title || metadata.section_title || "No section";
  const section = source.section_title || metadata.section_title || "No section";
  const page = source.page_label || (metadata.page_start ? `p.${metadata.page_start}` : "page unknown");
  const filename = source.filename || "unknown file";
  const chunkNumber = source.chunk_index ?? "unknown";
  const kind = source.source_kind || (metadata.contains_table ? "table" : "text");
  const tableRows = metadata.table_rows || [];

  return (
    <div className="source">
      <div className="source-heading">
        <span className={`citation-badge ${kind}`}>[{citation}]</span>
        <div className="source-heading-text">
          <strong>{title}</strong>
          <span>{section}</span>
        </div>
        <div className="source-copy-actions">
          <button className="source-action-button" type="button" title="Copy citation" aria-label="Copy citation" onClick={onCopyCitation}>
            <CopyIcon />
            <span>Citation</span>
          </button>
          <button className="source-action-button" type="button" title="Copy source evidence" aria-label="Copy source evidence" onClick={onCopyEvidence}>
            <CopyAllIcon />
            <span>Evidence</span>
          </button>
        </div>
      </div>
      <div className="source-detail-grid">
        <div>
          <span>File</span>
          <strong>{filename}</strong>
        </div>
        <div>
          <span>Section</span>
          <strong>{section}</strong>
        </div>
        <div>
          <span>Page</span>
          <strong>{page}</strong>
        </div>
        <div>
          <span>Chunk</span>
          <strong>{chunkNumber}</strong>
        </div>
      </div>
      <div className="source-tags">
        <span>{kind}</span>
        <span>score {source.score}</span>
        {metadata.table_row_count ? <span>{metadata.table_row_count} rows</span> : null}
      </div>
      <div className="source-preview">
        <span>Evidence preview</span>
        <p>{compactEvidence(source.text)}</p>
      </div>
      <details className="source-evidence">
        <summary>Evidence</summary>
        {metadata.contains_table && (
          <div className="source-meta">
            {metadata.table_columns?.length ? `Columns: ${metadata.table_columns.join(", ")}` : "Table columns unavailable"}
          </div>
        )}
        {metadata.contains_table && tableRows.length ? (
          <div className="evidence-table-rows">
            {tableRows.slice(0, 10).map((row, rowIndex) => (
              <div key={`${citation}-row-${rowIndex}`}>{row}</div>
            ))}
          </div>
        ) : null}
        <div className="source-text">{source.text}</div>
        <div className="source-reference">{citationReference(source)}</div>
        {debugMode && <SourceDebug source={source} />}
      </details>
      {citationCopied && <div className="source-copied">Citation copied</div>}
      {evidenceCopied && <div className="source-copied">Evidence copied</div>}
    </div>
  );
}

function SourceDebug({ source }) {
  const metadata = source.metadata || {};
  const debugMetadata = pickMetadata(metadata, [
    "content_types",
    "section_path",
    "parent_section",
    "page_start",
    "page_end",
    "page_label_start",
    "page_label_end",
    "contains_table",
    "table_title",
    "table_row_count",
    "table_quality_score",
    "has_numeric_constraints",
    "safety_critical",
    "safety_flags",
    "semantic_labels",
    "technical_identifiers",
    "engineering_entities",
    "ontology_concepts",
    "language_code",
    "translation_confidence",
    "index_backend",
    "embedding_backend",
    "embedding_dimensions",
  ]);
  return (
    <div className="source-debug">
      <strong>Source debug</strong>
      {!!source.debug?.scores && <DebugKeyValues value={source.debug.scores} />}
      {!!source.debug?.details && <DebugObject title="Retriever details" value={source.debug.details} />}
      <DebugObject title="Metadata" value={debugMetadata} />
    </div>
  );
}

function DebugKeyValues({ value }) {
  const entries = Object.entries(value || {}).filter(([, item]) => item !== undefined && item !== null && item !== "" && item !== 0);
  if (!entries.length) return <div className="debug-empty">No values</div>;
  return (
    <div className="debug-key-values">
      {entries.map(([key, item]) => (
        <div key={key}>
          <span>{formatRouteName(key)}</span>
          <strong>{formatDebugValue(item)}</strong>
        </div>
      ))}
    </div>
  );
}

function DebugObject({ title, value }) {
  const compact = compactDebugObject(value);
  return (
    <details className="debug-object">
      <summary>{title}</summary>
      <pre>{JSON.stringify(compact, null, 2)}</pre>
    </details>
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

function citationCopyKey(source, index) {
  return `citation-${source.citation_id || index}-${source.chunk_index}`;
}

function evidenceCopyKey(source, index) {
  return `evidence-${source.citation_id || index}-${source.chunk_index}`;
}

function fullResponseCopyKey(message, index) {
  return `full-${message.id || index}`;
}

function citationReference(source) {
  if (source.reference) return source.reference;
  const metadata = source.metadata || {};
  const citation = source.citation_id || "S?";
  const section = source.section_title || metadata.section_title || "No section";
  const page = source.page_label || (metadata.page_start ? `p.${metadata.page_start}` : "page unknown");
  return `[${citation}] ${source.filename || "unknown file"} | ${section} | ${page} | chunk ${source.chunk_index ?? "unknown"}`;
}

function sourceEvidenceText(source) {
  const metadata = source.metadata || {};
  const rows = metadata.table_rows?.length
    ? `\nTable rows:\n${metadata.table_rows.slice(0, 20).map((row) => `- ${row}`).join("\n")}`
    : "";
  const columns = metadata.table_columns?.length ? `\nColumns: ${metadata.table_columns.join(", ")}` : "";
  return `${citationReference(source)}${columns}${rows}\n\nEvidence:\n${source.text || ""}`.trim();
}

function fullResponseText(message) {
  const lines = [`Answer:\n${message.text || ""}`];
  if (message.retrieval) {
    const timing = message.retrieval.timing || {};
    const retrievalMs = Number(timing.retrieval_ms || 0);
    const backendTotalMs = Number(timing.total_ms || 0);
    lines.push(
      [
        "Retrieval:",
        `- Evidence: ${message.retrieval.evidence_label || message.retrieval.evidence_status || "unknown"}`,
        `- Sources: ${message.retrieval.source_count ?? message.sources?.length ?? 0}`,
        `- Top score: ${formatScore(message.retrieval.top_score)}`,
        `- Route: ${formatRouteName(message.retrieval.primary_route || message.retrieval.route?.primary || "unknown")}`,
        `- Retrieval time: ${formatMs(retrievalMs)}`,
        `- Answer time: ${formatMs(Math.max(0, backendTotalMs - retrievalMs))}`,
        `- Total time: ${formatMs(timing.client_total_ms || backendTotalMs)}`,
        `- Scope: indexed documents only`,
      ].join("\n")
    );
  }
  if (message.sources?.length) {
    lines.push(`Sources:\n${message.sources.map((source) => `- ${citationReference(source)}`).join("\n")}`);
  }
  return lines.join("\n\n").trim();
}

function pickMetadata(metadata, keys) {
  return Object.fromEntries(
    keys
      .filter((key) => metadata[key] !== undefined && metadata[key] !== null && metadata[key] !== "" && metadata[key] !== false)
      .map((key) => [key, metadata[key]])
  );
}

function compactDebugObject(value, depth = 0) {
  if (depth > 2) return "[nested]";
  if (Array.isArray(value)) return value.slice(0, 12).map((item) => compactDebugObject(item, depth + 1));
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined && item !== null && item !== "" && item !== false)
        .slice(0, 40)
        .map(([key, item]) => [key, compactDebugObject(item, depth + 1)])
    );
  }
  return value;
}

function formatDebugValue(value) {
  if (typeof value === "number") return value > 1 ? value.toFixed(3) : formatScore(value);
  if (Array.isArray(value)) return value.slice(0, 5).join(", ");
  if (value && typeof value === "object") return JSON.stringify(compactDebugObject(value));
  return String(value);
}

function compactEvidence(text) {
  const compact = String(text || "").replace(/\s+/g, " ").trim();
  if (compact.length <= 220) return compact;
  return `${compact.slice(0, 220).trim()}...`;
}

function RetrievalTransparency({ retrieval }) {
  const route = retrieval.route || {};
  const signals = Object.entries(retrieval.source_signals || {}).filter(([, count]) => Number(count) > 0);
  const activeRetrievers = retrieval.active_retrievers || route.retrievers || [];
  const timing = retrieval.timing || {};
  const retrievalMs = Number(timing.retrieval_ms || 0);
  const backendTotalMs = Number(timing.total_ms || 0);
  const generationMs = Math.max(0, backendTotalMs - retrievalMs);
  const roundTripMs = Number(timing.client_total_ms || backendTotalMs || 0);
  const questionType = retrieval.question_type?.label || retrieval.question_type?.type_id || "Question";

  return (
    <section className="retrieval-transparency" aria-label="Retrieval transparency">
      <div className="retrieval-strip">
        <span className={`evidence-pill ${retrieval.evidence_status || "unknown"}`}>
          {retrieval.evidence_label || "Evidence"}
        </span>
        {retrieval.resolved_follow_up && <span className="follow-up-pill">Follow-up linked</span>}
        <span>{questionType}</span>
        <span>{retrieval.source_count || 0} sources</span>
        <span>Top {formatScore(retrieval.top_score)}</span>
        <span>{formatMs(timing.total_ms || timing.client_total_ms)}</span>
      </div>
      <div className="retrieval-meta-grid">
        <div>
          <span>Route</span>
          <strong>{formatRouteName(retrieval.primary_route || route.primary || "hybrid")}</strong>
        </div>
        <div>
          <span>Section</span>
          <strong>{retrieval.dominant_section || "No section"}</strong>
        </div>
        <div>
          <span>Avg score</span>
          <strong>{formatScore(retrieval.average_score)}</strong>
        </div>
        <div>
          <span>Quality</span>
          <strong>
            {retrieval.answer_quality?.grade || "ungraded"} {formatScore(retrieval.answer_quality?.overall_score)}
          </strong>
        </div>
      </div>
      {!!activeRetrievers.length && (
        <div className="retriever-chips">
          {activeRetrievers.slice(0, 8).map((name) => (
            <span className="retriever-chip" key={name}>{formatRouteName(name)}</span>
          ))}
        </div>
      )}
      {!!signals.length && (
        <div className="signal-row">
          {signals.map(([name, count]) => (
            <span className="signal-chip" key={name}>
              {formatRouteName(name)} {count}
            </span>
          ))}
        </div>
      )}
      <details className="retrieval-details">
        <summary>Retrieval details</summary>
        <div className="retrieval-detail-list">
          {retrieval.resolved_follow_up && <span>Follow-up resolved against chat history</span>}
          <span>Retrieval {formatMs(retrievalMs)} | answer {formatMs(generationMs)}</span>
          <span>Backend total {formatMs(backendTotalMs)} | browser round trip {formatMs(roundTripMs)}</span>
          {!!retrieval.source_sections?.length && (
            <span>
              Sections: {retrieval.source_sections.map((item) => `${item.section} (${item.count})`).join("; ")}
            </span>
          )}
        </div>
      </details>
    </section>
  );
}

function ConfidenceBadge({ confidence = {}, retrieval = {} }) {
  const level = retrieval.evidence_status || confidence.confidence || "unknown";
  const evidenceLabel = retrieval.evidence_label || evidenceStatusLabel(level);
  const score = confidence.top_score ?? retrieval.top_score;
  return (
    <div className={`confidence-badge ${level}`}>
      <span>Evidence: {evidenceLabel}</span>
      {confidence.confidence ? <span>Confidence: {confidence.confidence}</span> : null}
      {score !== undefined ? <span>Top: {formatScore(score)}</span> : null}
      {confidence.dominant_section || retrieval.dominant_section ? (
        <span>{confidence.dominant_section || retrieval.dominant_section}</span>
      ) : null}
    </div>
  );
}

function evidenceStatusLabel(value) {
  const status = String(value || "unknown").replace(/_/g, " ");
  return status.charAt(0).toUpperCase() + status.slice(1);
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

const PROCESSING_STAGES = [
  { label: "Classifying question", detail: "Detecting question type and answer rules", afterMs: 0 },
  { label: "Selecting retrievers", detail: "Choosing route, filters, and source strategy", afterMs: 450 },
  { label: "Retrieving evidence", detail: "Searching indexed document chunks", afterMs: 950 },
  { label: "Checking grounding", detail: "Scoring source support and evidence quality", afterMs: 1600 },
  { label: "Writing grounded answer", detail: "Preparing answer, citations, and trace data", afterMs: 2400 },
];

function processingStageIndex(elapsedMs) {
  let index = 0;
  for (let stageIndex = 0; stageIndex < PROCESSING_STAGES.length; stageIndex += 1) {
    if (elapsedMs >= PROCESSING_STAGES[stageIndex].afterMs) index = stageIndex;
  }
  return index;
}

function formatScore(value) {
  const score = Number(value);
  if (!Number.isFinite(score)) return "0%";
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`;
}

function formatMs(value) {
  const ms = Number(value);
  if (!Number.isFinite(ms) || ms <= 0) return "0 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function formatRouteName(value) {
  return String(value || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatCheckName(value) {
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

createRoot(document.getElementById("root")).render(<App />);
