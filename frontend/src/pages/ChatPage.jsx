import {
  ArrowLeft,
  Bot,
  Menu,
  MessageSquare,
  PanelLeftClose,
  Plus,
  Send,
  Sparkles,
  Trash2,
  User,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  addChatSource,
  clearChatHistory,
  createChatSession,
  deleteChatSession,
  getChatSession,
  getPdfFileUrl,
  indexDownloadedPdf,
  listChatThreads,
  listDownloadedPdfs,
  removeChatSource,
  streamChatWithPaper,
} from "../api.js";
import { evidenceQualityClass, formatDateTime, formatEvidenceQuality } from "../utils/format.js";
import { displayTitleFromFilename, paperIdFromFilename, sourceFromPdf } from "../utils/paper.js";

export default function ChatPage({ onBackHome, initialPaper }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [threads, setThreads] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [question, setQuestion] = useState("");
  const [threadsState, setThreadsState] = useState({ loading: false, error: "" });
  const [chatState, setChatState] = useState({ loading: false, error: "" });
  const [sourceState, setSourceState] = useState({ loading: false, error: "", message: "" });
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false);
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const [paperPreview, setPaperPreview] = useState(null);
  const initialPaperHandled = useRef(false);

  const sourceIds = useMemo(
    () => new Set((activeChat?.sources ?? []).map((source) => source.paper_id)),
    [activeChat],
  );

  useEffect(() => {
    void refreshThreads();
    void refreshPdfs();
  }, []);

  useEffect(() => {
    if (!initialPaper || initialPaperHandled.current) return;
    initialPaperHandled.current = true;
    void startChatWithPaper(initialPaper);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPaper]);

  async function refreshThreads() {
    setThreadsState({ loading: true, error: "" });
    try {
      const response = await listChatThreads();
      setThreads(response.chats ?? []);
      setThreadsState({ loading: false, error: "" });
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function refreshPdfs() {
    try {
      const pdfs = await listDownloadedPdfs();
      setDownloadedPdfs(pdfs ?? []);
    } catch {
      setDownloadedPdfs([]);
    }
  }

  async function createNewChat() {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    try {
      const session = await createChatSession("Cuộc trò chuyện mới");
      setActiveChat(session);
      setQuestion("");
      setSourcePanelOpen(false);
      await refreshThreads();
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function openChat(chatId) {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    try {
      const session = await getChatSession(chatId);
      setActiveChat(session);
      setQuestion("");
      if (window.innerWidth < 900) setSidebarOpen(false);
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function deleteChat(chatId) {
    setThreadsState({ loading: true, error: "" });
    try {
      await deleteChatSession(chatId);
      setDeleteCandidate(null);
      if (activeChat?.chat_id === chatId) {
        setActiveChat(null);
        setQuestion("");
      }
      await refreshThreads();
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function ensureActiveChat() {
    if (activeChat) return activeChat;
    const session = await createChatSession("Cuộc trò chuyện mới");
    setActiveChat(session);
    await refreshThreads();
    return session;
  }

  async function startChatWithPaper(pdf) {
    setSourceState({ loading: true, error: "", message: `Đang index ${pdf.filename}...` });
    setChatState({ loading: false, error: "" });
    try {
      const chat = await ensureActiveChat();
      const indexResponse = await indexDownloadedPdf(pdf.filename);
      const session = await addChatSource(chat.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: "" });
      setSourcePanelOpen(false);
      if (window.innerWidth < 900) setSidebarOpen(false);
      await refreshThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function addPdfToChat(pdf) {
    setSourceState({ loading: true, error: "", message: `Đang index ${pdf.filename}...` });
    try {
      const chat = await ensureActiveChat();
      const indexResponse = await indexDownloadedPdf(pdf.filename);
      const session = await addChatSource(chat.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: `${pdf.filename} đã được thêm.` });
      await refreshThreads();
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
      setSourceState({ loading: false, error: "", message: "Đã gỡ nguồn khỏi cuộc trò chuyện." });
      await refreshThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
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
      await refreshThreads();
    } catch (error) {
      setChatState({ loading: false, error: error.message });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !activeChat || sourceState.loading) return;

    const optimisticUser = {
      role: "user",
      content: trimmedQuestion,
      citations: [],
      created_at: new Date().toISOString(),
    };
    const optimisticAssistant = {
      role: "assistant",
      content: "",
      citations: [],
      created_at: new Date(Date.now() + 1).toISOString(),
      streaming: true,
    };
    const chatId = activeChat.chat_id;

    setActiveChat((chat) =>
      chat?.chat_id === chatId
        ? { ...chat, messages: [...chat.messages, optimisticUser, optimisticAssistant] }
        : chat,
    );
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      await streamChatWithPaper({
        question: trimmedQuestion,
        chatId,
        paperIds: activeChat.sources.length ? activeChat.sources.map((source) => source.paper_id) : undefined,
        topK: 5,
        scoreThreshold: 0.25,
        onToken: (token) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === optimisticAssistant.created_at
                  ? { ...message, content: `${message.content}${token}` }
                  : message,
              ),
            };
          });
        },
        onCitations: (citations) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === optimisticAssistant.created_at
                  ? { ...message, citations, streaming: false }
                  : message,
              ),
            };
          });
        },
      });
      const session = await getChatSession(chatId);
      setActiveChat((chat) => (chat?.chat_id === chatId ? session : chat));
      setChatState({ loading: false, error: "" });
      await refreshThreads();
    } catch (error) {
      setActiveChat((chat) => {
        if (!chat || chat.chat_id !== chatId) return chat;
        return {
          ...chat,
          messages: chat.messages.map((message) =>
            message.created_at === optimisticAssistant.created_at ? { ...message, streaming: false } : message,
          ),
        };
      });
      setChatState({ loading: false, error: error.message });
    }
  }

  function openCitation(citation) {
    const source = activeChat?.sources.find(
      (candidate) =>
        candidate.paper_id === citation.paper_id ||
        candidate.title === citation.title ||
        candidate.filename === citation.title,
    );
    const filename = source?.filename ?? (citation.title?.toLowerCase().endsWith(".pdf") ? citation.title : null);
    if (!filename) {
      setChatState({ loading: false, error: "Không mở được trích dẫn vì thiếu file PDF." });
      return;
    }
    setPaperPreview({
      ...(source ?? { paper_id: citation.paper_id, title: citation.title || citation.paper_id }),
      filename,
      pageNumber: citation.page_number ?? citation.page,
    });
  }

  const canChat = Boolean(activeChat) && !sourceState.loading;

  return (
    <div className={`chat-shell ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <aside className="chat-sidebar" aria-label="Danh sách cuộc trò chuyện">
        <div className="sidebar-top">
          <button className="btn-icon" onClick={onBackHome} type="button" aria-label="Về trang chủ">
            <ArrowLeft size={18} aria-hidden="true" />
          </button>
          <button className="sidebar-new-chat" onClick={createNewChat} type="button">
            <Plus size={16} aria-hidden="true" />
            Cuộc trò chuyện mới
          </button>
          <button
            className="btn-icon sidebar-collapse"
            onClick={() => setSidebarOpen(false)}
            type="button"
            aria-label="Thu gọn sidebar"
          >
            <PanelLeftClose size={18} aria-hidden="true" />
          </button>
        </div>

        {threadsState.error ? <div className="sidebar-error">{threadsState.error}</div> : null}

        <nav className="thread-list">
          {threads.length === 0 && !threadsState.loading ? (
            <p className="thread-empty">Chưa có cuộc trò chuyện nào.</p>
          ) : null}
          {threads.map((thread) => (
            <div className={`thread-item ${activeChat?.chat_id === thread.chat_id ? "active" : ""}`} key={thread.chat_id}>
              <button className="thread-button" onClick={() => openChat(thread.chat_id)} type="button">
                <MessageSquare size={15} aria-hidden="true" />
                <span className="thread-title">{thread.title}</span>
                <span className="thread-meta">{formatDateTime(thread.updated_at)}</span>
              </button>
              <button
                aria-label={`Xóa ${thread.title}`}
                className="thread-delete"
                onClick={() => setDeleteCandidate(thread)}
                type="button"
              >
                <Trash2 size={14} aria-hidden="true" />
              </button>
            </div>
          ))}
        </nav>
      </aside>

      <main className="chat-main">
        <header className="chat-topbar">
          <button className="btn-icon" onClick={() => setSidebarOpen(true)} type="button" aria-label="Mở sidebar">
            <Menu size={18} aria-hidden="true" />
          </button>
          <div className="chat-topbar-title">
            <Sparkles size={16} aria-hidden="true" />
            <span>{activeChat?.title ?? "Research Assistant"}</span>
          </div>
          <div className="chat-topbar-actions">
            {activeChat ? (
              <>
                <button className="btn-ghost btn-sm" disabled={!activeChat.messages.length || chatState.loading} onClick={handleClearHistory} type="button">
                  Xóa tin nhắn
                </button>
                <button className="btn-ghost btn-sm" onClick={() => setSourcePanelOpen(true)} type="button">
                  Nguồn tùy chọn ({activeChat.sources.length})
                </button>
              </>
            ) : null}
          </div>
        </header>

        <div className="chat-workspace">
          <div className="chat-body">
            {!activeChat ? (
              <div className="chat-welcome">
                <div className="welcome-icon">
                  <Bot size={28} aria-hidden="true" />
                </div>
                <h2>Bạn cần hỗ trợ gì hôm nay?</h2>
                <p>Tạo cuộc trò chuyện mới rồi hỏi AI. Agent sẽ tìm trong toàn bộ tài liệu local và dùng web khi thiếu context.</p>
                <button className="btn-primary" onClick={createNewChat} type="button">
                  <Plus size={16} aria-hidden="true" />
                  Bắt đầu cuộc trò chuyện
                </button>
              </div>
            ) : (
              <ChatMessages
                activeChat={activeChat}
                canChat={canChat}
                chatState={chatState}
                onOpenCitation={openCitation}
                sourceState={sourceState}
              />
            )}
          </div>

          {activeChat ? (
            <form className="chat-composer" onSubmit={handleAsk}>
              <div className="composer-box">
                <textarea
                  disabled={!canChat || chatState.loading}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      event.currentTarget.form?.requestSubmit();
                    }
                  }}
                  placeholder={canChat ? "Nhắn tin cho AI..." : "Tạo cuộc trò chuyện để bắt đầu"}
                  rows={1}
                  value={question}
                />
                <button
                  aria-label="Gửi"
                  className="composer-send"
                  disabled={!canChat || !question.trim() || chatState.loading}
                  type="submit"
                >
                  <Send size={18} aria-hidden="true" />
                </button>
              </div>
              <p className="composer-hint">Enter để gửi · Shift+Enter để xuống dòng</p>
            </form>
          ) : null}
        </div>
      </main>

      {sourcePanelOpen ? (
        <SourcePanel
          activeChat={activeChat}
          downloadedPdfs={downloadedPdfs}
          onAddPdf={addPdfToChat}
          onClose={() => setSourcePanelOpen(false)}
          onRemoveSource={removeSource}
          sourceIds={sourceIds}
          sourceState={sourceState}
        />
      ) : null}

      {deleteCandidate ? (
        <ConfirmDialog
          confirmLabel="Xóa"
          loading={threadsState.loading}
          message="Cuộc trò chuyện và danh sách nguồn sẽ bị xóa. File PDF local vẫn được giữ."
          onCancel={() => setDeleteCandidate(null)}
          onConfirm={() => deleteChat(deleteCandidate.chat_id)}
          title="Xóa cuộc trò chuyện?"
        />
      ) : null}

      {paperPreview ? (
        <PdfPreviewModal onClose={() => setPaperPreview(null)} source={paperPreview} />
      ) : null}
    </div>
  );
}

function ChatMessages({ activeChat, canChat, chatState, onOpenCitation, sourceState }) {
  const logRef = useRef(null);

  useEffect(() => {
    const log = logRef.current;
    if (!log) return;
    log.scrollTop = log.scrollHeight;
  }, [activeChat.messages, chatState.loading]);

  if (activeChat.messages.length === 0) {
    return (
      <div className="chat-welcome chat-welcome-inline">
        <div className="welcome-icon">
          <Bot size={24} aria-hidden="true" />
        </div>
        <h2>Hỏi về paper của bạn</h2>
        <p>
          {canChat
            ? "AI sẽ truy xuất toàn bộ tài liệu local, rồi dùng web nếu local context chưa đủ."
            : "Tạo cuộc trò chuyện để bắt đầu hỏi AI."}
        </p>
        {sourceState.error ? <div className="banner banner-error">{sourceState.error}</div> : null}
        {sourceState.message ? <div className="banner banner-success">{sourceState.message}</div> : null}
      </div>
    );
  }

  return (
    <div className="chat-log" ref={logRef}>
      {activeChat.messages.map((message, index) => (
        <ChatMessage key={`${message.role}-${message.created_at}-${index}`} message={message} onOpenCitation={onOpenCitation} />
      ))}
      {chatState.error ? <div className="banner banner-error">{chatState.error}</div> : null}
    </div>
  );
}

function ChatMessage({ message, onOpenCitation }) {
  const isUser = message.role === "user";
  const content = message.content || (message.streaming ? "" : "");

  return (
    <article className={`chat-message ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-body">
        <div className="message-role">{isUser ? "Bạn" : "AI"}</div>
        <div className="message-text">
          {content || (message.streaming ? <span className="typing-dots">Đang suy nghĩ</span> : null)}
          {message.streaming && content ? <span className="typing-cursor" aria-hidden="true" /> : null}
        </div>
        {message.citations?.length ? (
          <div className="citation-list">
            {message.citations.map((citation) => (
              <button
                className={`citation-pill ${evidenceQualityClass(citation.evidence_quality)}`}
                key={citation.chunk_id ?? `${citation.paper_id}-${citation.page_number}`}
                onClick={() => onOpenCitation(citation)}
                type="button"
              >
                {citation.title || citation.paper_id}
                {citation.page_number ? ` · tr.${citation.page_number}` : ""}
                <span className="citation-quality">{formatEvidenceQuality(citation.evidence_quality)}</span>
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function SourcePanel({ activeChat, downloadedPdfs, onAddPdf, onClose, onRemoveSource, sourceIds, sourceState }) {
  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label="Quản lý nguồn">
      <section className="source-panel">
        <header className="source-panel-header">
          <div>
            <h2>Nguồn paper</h2>
            <p>Nguồn là tùy chọn. Không chọn nguồn thì AI sẽ tìm trong toàn bộ tài liệu local.</p>
          </div>
          <button className="btn-icon" onClick={onClose} type="button" aria-label="Đóng">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        {sourceState.message ? <div className="banner banner-success">{sourceState.message}</div> : null}
        {sourceState.error ? <div className="banner banner-error">{sourceState.error}</div> : null}

        {activeChat?.sources.length ? (
          <section className="source-attached">
            <h3>Đã gắn ({activeChat.sources.length})</h3>
            <ul>
              {activeChat.sources.map((source) => (
                <li key={source.paper_id}>
                  <span>{source.title}</span>
                  <button aria-label={`Gỡ ${source.title}`} onClick={() => onRemoveSource(source)} type="button">
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ) : null}

        <section className="source-picker">
          <h3>Thư viện local</h3>
          <ul>
            {downloadedPdfs.map((pdf) => {
              const paperId = paperIdFromFilename(pdf.filename);
              const added = sourceIds.has(paperId);
              return (
                <li key={pdf.path}>
                  <button disabled={added || sourceState.loading} onClick={() => onAddPdf(pdf)} type="button">
                    <span>{displayTitleFromFilename(pdf.filename)}</span>
                    <span className="source-picker-action">{added ? "Đã thêm" : "Thêm"}</span>
                  </button>
                </li>
              );
            })}
            {!downloadedPdfs.length ? <li className="source-empty">Không có paper local.</li> : null}
          </ul>
        </section>
      </section>
    </div>
  );
}

function ConfirmDialog({ title, message, confirmLabel, loading, onCancel, onConfirm }) {
  return (
    <div className="overlay" role="dialog" aria-modal="true">
      <section className="confirm-dialog">
        <h2>{title}</h2>
        <p>{message}</p>
        <div className="confirm-actions">
          <button className="btn-ghost" disabled={loading} onClick={onCancel} type="button">
            Hủy
          </button>
          <button className="btn-danger" disabled={loading} onClick={onConfirm} type="button">
            {loading ? "Đang xóa..." : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

function PdfPreviewModal({ onClose, source }) {
  const pdfUrl = getPdfFileUrl(source.filename);
  const pageNumber = source.pageNumber;
  const fragment = pageNumber ? `#page=${pageNumber}&view=FitH` : "#view=FitH";

  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label="Xem PDF">
      <section className="pdf-modal">
        <header>
          <h2>{source.title}</h2>
          <button className="btn-icon" onClick={onClose} type="button" aria-label="Đóng">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <iframe src={`${pdfUrl}${fragment}`} title={source.title} />
      </section>
    </div>
  );
}
