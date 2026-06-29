import {
  ArrowDownToLine,
  Bot,
  CalendarDays,
  FileText,
  Library,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  Send,
  Trash2,
  User,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  addChatSource,
  chatWithPaper,
  clearChatHistory,
  createChatSession,
  deleteChatSession,
  downloadPapers,
  getChatSession,
  getPdfFileUrl,
  indexDownloadedPdf,
  listChatThreads,
  listDownloadedPdfs,
  removeChatSource,
  searchPapers,
} from "./api";

const DEFAULT_QUERY = "Agentic RAG";

function normalizePaper(paper) {
  return {
    ...paper,
    arxivUrl: paper.arxiv_url ?? paper.url,
    pdfUrl: paper.pdf_url,
    published: paper.published ?? "Unknown date",
  };
}

function paperIdFromFilename(filename) {
  return filename.replace(/\.pdf$/i, "");
}

function sourceFromPdf(pdf, paperId = paperIdFromFilename(pdf.filename)) {
  return {
    paper_id: paperId,
    title: pdf.filename,
    filename: pdf.filename,
    path: pdf.path,
  };
}

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [maxResults, setMaxResults] = useState(5);
  const [sortBy, setSortBy] = useState("submittedDate");
  const [onlinePapers, setOnlinePapers] = useState([]);
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [chatThreads, setChatThreads] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [question, setQuestion] = useState("");
  const [isSourceModalOpen, setIsSourceModalOpen] = useState(false);
  const [sourceTab, setSourceTab] = useState("local");
  const [paperOverlay, setPaperOverlay] = useState(null);
  const [deleteCandidate, setDeleteCandidate] = useState(null);

  const [chatListState, setChatListState] = useState({ loading: false, error: "" });
  const [pdfListState, setPdfListState] = useState({ loading: false, error: "" });
  const [searchState, setSearchState] = useState({ loading: false, error: "" });
  const [sourceState, setSourceState] = useState({ loading: false, error: "", message: "" });
  const [chatState, setChatState] = useState({ loading: false, error: "" });

  const sourceIds = useMemo(() => new Set((activeChat?.sources ?? []).map((source) => source.paper_id)), [activeChat]);

  useEffect(() => {
    void refreshChatThreads();
    void refreshDownloadedPdfs();
  }, []);

  async function refreshChatThreads() {
    setChatListState({ loading: true, error: "" });
    try {
      const response = await listChatThreads();
      setChatThreads(response.chats ?? []);
      setChatListState({ loading: false, error: "" });
    } catch (error) {
      setChatListState({ loading: false, error: error.message });
    }
  }

  async function refreshDownloadedPdfs() {
    setPdfListState({ loading: true, error: "" });
    try {
      const response = await listDownloadedPdfs();
      setDownloadedPdfs(response ?? []);
      setPdfListState({ loading: false, error: "" });
    } catch (error) {
      setPdfListState({ loading: false, error: error.message });
    }
  }

  async function createNewChat() {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    try {
      const session = await createChatSession("New chat");
      setActiveChat(session);
      setQuestion("");
      setIsSourceModalOpen(true);
      await refreshChatThreads();
    } catch (error) {
      setChatListState({ loading: false, error: error.message });
    }
  }

  async function openChat(chatId) {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    try {
      const session = await getChatSession(chatId);
      setActiveChat(session);
      setQuestion("");
      setIsSourceModalOpen(false);
    } catch (error) {
      setChatListState({ loading: false, error: error.message });
    }
  }

  async function deleteChat(chatId) {
    setChatListState({ loading: true, error: "" });
    try {
      await deleteChatSession(chatId);
      setDeleteCandidate(null);
      if (activeChat?.chat_id === chatId) {
        setActiveChat(null);
        setQuestion("");
        setIsSourceModalOpen(false);
        setPaperOverlay(null);
      }
      await refreshChatThreads();
    } catch (error) {
      setChatListState({ loading: false, error: error.message });
    }
  }

  async function ensureActiveChat() {
    if (activeChat) return activeChat;
    const session = await createChatSession("New chat");
    setActiveChat(session);
    await refreshChatThreads();
    return session;
  }

  async function addLocalPdfToChat(pdf) {
    setSourceState({ loading: true, error: "", message: `Indexing ${pdf.filename}...` });
    try {
      const chat = await ensureActiveChat();
      const indexResponse = await indexDownloadedPdf(pdf.filename);
      const session = await addChatSource(chat.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      setActiveChat(session);
      setSourceState({
        loading: false,
        error: "",
        message: `${pdf.filename} added to this chat.`,
      });
      await refreshChatThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function ingestOnlinePaper(paper) {
    if (!paper.pdfUrl) {
      setSourceState({ loading: false, error: "This paper does not have a PDF URL.", message: "" });
      return;
    }

    setSourceState({ loading: true, error: "", message: "Downloading and indexing PDF..." });
    try {
      const chat = await ensureActiveChat();
      const downloadResponse = await downloadPapers([paper.pdfUrl]);
      const downloadedPath = downloadResponse.files?.[0];
      if (!downloadedPath) {
        throw new Error(downloadResponse.errors?.[0]?.error ?? "PDF download failed.");
      }

      await refreshDownloadedPdfs();
      const filename = downloadedPath.split("/").pop();
      const indexResponse = await indexDownloadedPdf(filename);
      const session = await addChatSource(chat.chat_id, {
        paper_id: indexResponse.paper_id,
        title: paper.title,
        filename,
        path: downloadedPath,
      });
      setActiveChat(session);
      setSourceState({
        loading: false,
        error: "",
        message: `${paper.title} added to this chat.`,
      });
      await refreshChatThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function removeSource(source) {
    if (!activeChat) return;
    setSourceState({ loading: true, error: "", message: "" });
    try {
      const session = await removeChatSource(activeChat.chat_id, source.paper_id);
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: "Source removed from this chat." });
      await refreshChatThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function handleSearch(event) {
    event.preventDefault();
    setSearchState({ loading: true, error: "" });
    try {
      const response = await searchPapers({ query, maxResults, sortBy });
      setOnlinePapers((response.papers ?? []).map(normalizePaper));
      setSearchState({ loading: false, error: "" });
    } catch (error) {
      setSearchState({ loading: false, error: error.message });
    }
  }

  async function handleClearHistory() {
    if (!activeChat) return;
    setChatState({ loading: true, error: "" });
    try {
      await clearChatHistory(activeChat.chat_id);
      const session = await getChatSession(activeChat.chat_id);
      setActiveChat(session);
      setChatState({ loading: false, error: "" });
      await refreshChatThreads();
    } catch (error) {
      setChatState({ loading: false, error: error.message });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !activeChat || sourceState.loading || activeChat.sources.length === 0) return;

    const optimisticMessage = {
      role: "user",
      content: trimmedQuestion,
      citations: [],
      created_at: new Date().toISOString(),
    };
    setActiveChat((chat) => ({ ...chat, messages: [...chat.messages, optimisticMessage] }));
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      await chatWithPaper({
        question: trimmedQuestion,
        chatId: activeChat.chat_id,
        topK: 5,
        scoreThreshold: 0.25,
      });
      const session = await getChatSession(activeChat.chat_id);
      setActiveChat(session);
      setChatState({ loading: false, error: "" });
      await refreshChatThreads();
    } catch (error) {
      setChatState({ loading: false, error: error.message });
    }
  }

  return (
    <main className="research-shell">
      <aside className="chat-rail" aria-label="Chats">
        <div className="rail-header">
          <div>
            <h1>AI Research Assistant</h1>
            <p>{chatThreads.length ? `${chatThreads.length} chats` : "No chats yet"}</p>
          </div>
          <button aria-label="New chat" onClick={createNewChat} type="button">
            <Plus size={18} aria-hidden="true" />
          </button>
        </div>

        {chatListState.error ? <div className="error-box compact-box">{chatListState.error}</div> : null}

        <div className="rail-list">
          {chatThreads.length === 0 && !chatListState.loading ? (
            <div className="empty-state compact">
              <MessageSquare size={22} aria-hidden="true" />
              <span>Create a chat to start a research session.</span>
            </div>
          ) : null}
          {chatThreads.map((thread) => (
            <ChatThreadCard
              active={activeChat?.chat_id === thread.chat_id}
              key={thread.chat_id}
              onDelete={() => setDeleteCandidate(thread)}
              onClick={() => openChat(thread.chat_id)}
              thread={thread}
            />
          ))}
          {chatListState.loading ? <div className="rail-loading">Loading chats...</div> : null}
        </div>
      </aside>

      <section className="workspace-panel" aria-label="Current chat">
        {activeChat ? (
          <ChatWorkspace
            activeChat={activeChat}
            chatState={chatState}
            onAddSources={() => setIsSourceModalOpen(true)}
            onAsk={handleAsk}
            onClearHistory={handleClearHistory}
            onOpenSource={setPaperOverlay}
            onRemoveSource={removeSource}
            question={question}
            setQuestion={setQuestion}
            sourceState={sourceState}
          />
        ) : (
          <EmptyWorkspace onCreateChat={createNewChat} />
        )}
      </section>

      {isSourceModalOpen ? (
        <AddSourcesModal
          downloadedPdfs={downloadedPdfs}
          maxResults={maxResults}
          onAddLocalPdf={addLocalPdfToChat}
          onClose={() => setIsSourceModalOpen(false)}
          onIngestOnlinePaper={ingestOnlinePaper}
          onRefreshPdfs={refreshDownloadedPdfs}
          onSearch={handleSearch}
          onlinePapers={onlinePapers}
          pdfListState={pdfListState}
          query={query}
          searchState={searchState}
          setMaxResults={setMaxResults}
          setQuery={setQuery}
          setSortBy={setSortBy}
          sortBy={sortBy}
          sourceIds={sourceIds}
          sourceState={sourceState}
          sourceTab={sourceTab}
          setSourceTab={setSourceTab}
        />
      ) : null}

      {paperOverlay ? <PaperPreviewOverlay onClose={() => setPaperOverlay(null)} source={paperOverlay} /> : null}

      {deleteCandidate ? (
        <ConfirmDeleteChatDialog
          chat={deleteCandidate}
          deleting={chatListState.loading}
          onCancel={() => setDeleteCandidate(null)}
          onConfirm={() => deleteChat(deleteCandidate.chat_id)}
        />
      ) : null}
    </main>
  );
}

function ChatWorkspace({
  activeChat,
  chatState,
  onAddSources,
  onAsk,
  onClearHistory,
  onOpenSource,
  onRemoveSource,
  question,
  setQuestion,
  sourceState,
}) {
  const canChat = activeChat.sources.length > 0 && !sourceState.loading;

  return (
    <>
      <header className="workspace-header">
        <div>
          <h2>{activeChat.title}</h2>
          <p>
            {activeChat.sources.length
              ? `${activeChat.sources.length} sources attached`
              : "Add sources before asking questions."}
          </p>
        </div>
        <div className="workspace-actions">
          <button className="secondary-action" disabled={activeChat.messages.length === 0 || chatState.loading} onClick={onClearHistory} type="button">
            Clear
          </button>
          <button className="primary-action" onClick={onAddSources} type="button">
            <Plus size={17} aria-hidden="true" />
            Add sources
          </button>
        </div>
      </header>

      <section className="sources-strip" aria-label="Sources">
        {activeChat.sources.length ? (
          activeChat.sources.map((source) => (
            <div className="source-chip" key={source.paper_id}>
              <button onClick={() => onOpenSource(source)} type="button">
                <FileText size={16} aria-hidden="true" />
                <span>{source.title}</span>
              </button>
              <button aria-label={`Remove ${source.title}`} onClick={() => onRemoveSource(source)} type="button">
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
          ))
        ) : (
          <button className="source-empty-button" onClick={onAddSources} type="button">
            <Library size={18} aria-hidden="true" />
            Add local PDFs or web papers
          </button>
        )}
      </section>

      {sourceState.message ? <div className="success-box compact-box">{sourceState.message}</div> : null}
      {sourceState.error ? <div className="error-box compact-box">{sourceState.error}</div> : null}

      <div className="chat-log">
        {activeChat.messages.length === 0 ? (
          <div className="empty-state compact">
            <Bot size={22} aria-hidden="true" />
            <span>{canChat ? "Ask a grounded question across this chat's sources." : "Attach at least one source to begin."}</span>
          </div>
        ) : null}
        {activeChat.messages.map((message, index) => (
          <ChatMessage key={`${message.role}-${message.created_at}-${index}`} message={message} />
        ))}
        {chatState.loading ? (
          <div className="message assistant">
            <Bot size={18} aria-hidden="true" />
            <div className="message-bubble">Thinking...</div>
          </div>
        ) : null}
      </div>

      {chatState.error ? <div className="error-box compact-error">{chatState.error}</div> : null}

      <form className="chat-form" onSubmit={onAsk}>
        <input
          disabled={!canChat || chatState.loading}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={canChat ? "Ask across the selected sources..." : "Add sources before chatting"}
          value={question}
        />
        <button disabled={!canChat || !question.trim() || chatState.loading} type="submit">
          <Send size={17} aria-hidden="true" />
          <span>Ask</span>
        </button>
      </form>
    </>
  );
}

function AddSourcesModal({
  downloadedPdfs,
  maxResults,
  onAddLocalPdf,
  onClose,
  onIngestOnlinePaper,
  onRefreshPdfs,
  onSearch,
  onlinePapers,
  pdfListState,
  query,
  searchState,
  setMaxResults,
  setQuery,
  setSortBy,
  sortBy,
  sourceIds,
  sourceState,
  sourceTab,
  setSourceTab,
}) {
  return (
    <div className="overlay-backdrop" role="dialog" aria-modal="true" aria-label="Add sources">
      <section className="sources-modal">
        <header className="modal-header">
          <div>
            <h2>Add sources</h2>
            <p>Attach local PDFs or ingest web papers into the current chat.</p>
          </div>
          <button aria-label="Close" onClick={onClose} type="button">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="tabs" role="tablist">
          <button className={sourceTab === "local" ? "active" : ""} onClick={() => setSourceTab("local")} type="button">
            Local PDFs
          </button>
          <button className={sourceTab === "web" ? "active" : ""} onClick={() => setSourceTab("web")} type="button">
            Search web
          </button>
        </div>

        {sourceState.message ? <div className="success-box compact-box">{sourceState.message}</div> : null}
        {sourceState.error ? <div className="error-box compact-box">{sourceState.error}</div> : null}

        {sourceTab === "local" ? (
          <section className="modal-body">
            <div className="section-title-row prominent">
              <div>
                <h2>Your library</h2>
                <p>{downloadedPdfs.length ? `${downloadedPdfs.length} local PDFs available` : "No local PDFs found"}</p>
              </div>
              <button aria-label="Refresh PDFs" disabled={pdfListState.loading} onClick={onRefreshPdfs} type="button">
                <RefreshCw size={16} aria-hidden="true" />
              </button>
            </div>
            <div className="source-picker-list">
              {downloadedPdfs.map((pdf) => {
                const paperId = paperIdFromFilename(pdf.filename);
                const isAdded = sourceIds.has(paperId);
                return (
                  <button className="source-picker-item" disabled={isAdded || sourceState.loading} key={pdf.path} onClick={() => onAddLocalPdf(pdf)} type="button">
                    <FileText size={20} aria-hidden="true" />
                    <span className="source-picker-title">{pdf.filename}</span>
                    <span className="source-picker-meta">
                      {formatBytes(pdf.size_bytes)} · {formatDateTime(pdf.modified_at)}
                    </span>
                    <span className="source-picker-action">{isAdded ? "Added" : "Add"}</span>
                  </button>
                );
              })}
              {!downloadedPdfs.length && !pdfListState.loading ? <div className="empty-state">No downloaded PDFs found.</div> : null}
            </div>
          </section>
        ) : (
          <section className="modal-body">
            <form className="search-form compact-search" onSubmit={onSearch}>
              <div className="search-input-row">
                <Search size={18} aria-hidden="true" />
                <input onChange={(event) => setQuery(event.target.value)} placeholder="Agentic RAG" value={query} />
                <button disabled={searchState.loading || !query.trim()} type="submit">
                  Search
                </button>
              </div>
              <div className="search-controls">
                <label>
                  Results
                  <input max="20" min="1" onChange={(event) => setMaxResults(Number(event.target.value))} type="number" value={maxResults} />
                </label>
                <label>
                  Sort
                  <select onChange={(event) => setSortBy(event.target.value)} value={sortBy}>
                    <option value="submittedDate">Submitted date</option>
                    <option value="lastUpdatedDate">Last updated</option>
                    <option value="relevance">Relevance</option>
                  </select>
                </label>
              </div>
            </form>
            {searchState.error ? <div className="error-box">{searchState.error}</div> : null}
            <div className="online-result-list modal-results">
              {onlinePapers.length === 0 && !searchState.loading ? (
                <div className="empty-state compact">
                  <Search size={22} aria-hidden="true" />
                  <span>Search results will appear here.</span>
                </div>
              ) : null}
              {onlinePapers.map((paper) => (
                <OnlinePaperCard disabled={sourceState.loading} key={paper.paper_id} onClick={() => onIngestOnlinePaper(paper)} paper={paper} />
              ))}
            </div>
          </section>
        )}
      </section>
    </div>
  );
}

function EmptyWorkspace({ onCreateChat }) {
  return (
    <div className="workspace-empty">
      <MessageSquare size={34} aria-hidden="true" />
      <h2>Start a research chat</h2>
      <p>Create a chat, attach sources, then ask questions grounded in those papers.</p>
      <button className="primary-action" onClick={onCreateChat} type="button">
        <Plus size={17} aria-hidden="true" />
        New chat
      </button>
    </div>
  );
}

function PaperPreviewOverlay({ onClose, source }) {
  const pdfUrl = source.filename ? getPdfFileUrl(source.filename) : null;

  return (
    <div className="overlay-backdrop" role="dialog" aria-modal="true" aria-label="Paper preview">
      <section className="paper-overlay">
        <div className="paper-detail-header">
          <button className="secondary-action" onClick={onClose} type="button">
            <X size={17} aria-hidden="true" />
            Close
          </button>
          <div className="paper-detail-title">
            <div className="paper-kind">Source</div>
            <h1>{source.title}</h1>
            <p>{source.path ?? source.paper_id}</p>
          </div>
          {source.path ? (
            <span className="index-badge neutral">
              <ArrowDownToLine size={13} aria-hidden="true" />
              Local file
            </span>
          ) : null}
        </div>

        {pdfUrl ? (
          <iframe className="pdf-frame" src={`${pdfUrl}#view=FitH`} title="Paper PDF" />
        ) : (
          <div className="pdf-placeholder">
            <FileText size={28} aria-hidden="true" />
            <span>PDF preview unavailable.</span>
          </div>
        )}
      </section>
    </div>
  );
}

function ConfirmDeleteChatDialog({ chat, deleting, onCancel, onConfirm }) {
  return (
    <div className="overlay-backdrop" role="dialog" aria-modal="true" aria-label="Delete chat confirmation">
      <section className="confirm-dialog">
        <div className="confirm-icon">
          <Trash2 size={22} aria-hidden="true" />
        </div>
        <div>
          <h2>Delete this chat?</h2>
          <p>
            This will remove the conversation and its attached source list. Local PDF files and indexed vectors will stay
            in the system.
          </p>
          <div className="confirm-chat-title">{chat.title}</div>
        </div>
        <div className="confirm-actions">
          <button className="secondary-action" disabled={deleting} onClick={onCancel} type="button">
            Cancel
          </button>
          <button className="danger-action" disabled={deleting} onClick={onConfirm} type="button">
            <Trash2 size={16} aria-hidden="true" />
            {deleting ? "Deleting" : "Delete"}
          </button>
        </div>
      </section>
    </div>
  );
}

function ChatThreadCard({ active, onClick, onDelete, thread }) {
  return (
    <div className={`chat-thread-card ${active ? "active" : ""}`}>
      <button className="chat-thread-open" onClick={onClick} type="button">
        <MessageSquare size={20} aria-hidden="true" />
        <span className="chat-thread-title">{thread.title}</span>
        <span className="chat-thread-last">{thread.last_message}</span>
        <span className="chat-thread-meta">
          {thread.source_count} sources · {thread.message_count} messages · {formatDateTime(thread.updated_at)}
        </span>
      </button>
      <button aria-label={`Delete ${thread.title}`} className="chat-thread-delete" onClick={onDelete} type="button">
        <Trash2 size={15} aria-hidden="true" />
      </button>
    </div>
  );
}

function OnlinePaperCard({ disabled, onClick, paper }) {
  return (
    <button className="online-paper-card" disabled={disabled} onClick={onClick} type="button">
      <span className="paper-title">{paper.title}</span>
      <span className="paper-meta">
        <CalendarDays size={14} aria-hidden="true" />
        {paper.published}
      </span>
      <span className="author-line">{paper.authors?.join(", ") || "Unknown authors"}</span>
      <span className="online-paper-action">Ingest & add</span>
    </button>
  );
}

function ChatMessage({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      {isUser ? <User size={18} aria-hidden="true" /> : <Bot size={18} aria-hidden="true" />}
      <div className="message-content">
        <div className="message-bubble">{message.content}</div>
        {message.citations?.length ? (
          <div className="citation-list">
            {message.citations.map((citation) => (
              <span className="citation-pill" key={citation.chunk_id ?? `${citation.paper_id}-${citation.page_number}`}>
                {citation.title || citation.paper_id}
                {citation.page_number ? `, p. ${citation.page_number}` : ""}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "Unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDateTime(value) {
  if (!value) return "Unknown time";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export default App;
