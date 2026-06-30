import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowLeft,
  ArrowUp,
  Bot,
  BrainCircuit,
  CalendarDays,
  ChevronDown,
  Clipboard,
  FileText,
  Globe2,
  Library,
  Link2,
  MessageSquare,
  MoreVertical,
  PanelLeft,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Send,
  Trash2,
  UploadCloud,
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
  runResearch,
  streamChatWithPaper,
  updateChatSessionTitle,
  uploadLocalPdfs,
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
  const [mode, setMode] = useState("home");
  const [researchQuery, setResearchQuery] = useState(DEFAULT_QUERY);
  const [researchMaxResults, setResearchMaxResults] = useState(5);
  const [researchResult, setResearchResult] = useState(null);
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [chatThreads, setChatThreads] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [question, setQuestion] = useState("");
  const [isSourceModalOpen, setIsSourceModalOpen] = useState(false);
  const [paperOverlay, setPaperOverlay] = useState(null);
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const [renameCandidate, setRenameCandidate] = useState(null);
  const [renameTitle, setRenameTitle] = useState("");

  const [chatListState, setChatListState] = useState({ loading: false, error: "" });
  const [pdfListState, setPdfListState] = useState({ loading: false, error: "" });
  const [sourceState, setSourceState] = useState({ loading: false, error: "", message: "" });
  const [chatState, setChatState] = useState({ loading: false, error: "" });
  const [uploadState, setUploadState] = useState({ loading: false, error: "", message: "" });
  const [researchState, setResearchState] = useState({ loading: false, error: "" });

  const sourceIds = useMemo(() => new Set((activeChat?.sources ?? []).map((source) => source.paper_id)), [activeChat]);

  useEffect(() => {
    void refreshChatThreads();
    void refreshDownloadedPdfs();
  }, []);

  async function startChatMode() {
    setMode("chat");
    if (!activeChat) {
      setSourceState({ loading: false, error: "", message: "" });
      setChatState({ loading: false, error: "" });
      try {
        const session = await createChatSession("New chat");
        setActiveChat(session);
        setQuestion("");
        setIsSourceModalOpen(false);
        await refreshChatThreads();
      } catch (error) {
        setChatListState({ loading: false, error: error.message });
      }
      return;
    }
    setIsSourceModalOpen(false);
  }

  function returnHome() {
    setMode("home");
    setIsSourceModalOpen(false);
    setPaperOverlay(null);
  }

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

  function openRenameDialog(chat) {
    setRenameCandidate(chat);
    setRenameTitle(chat.title ?? "");
  }

  async function renameChat(event) {
    event.preventDefault();
    if (!renameCandidate) return;

    const trimmedTitle = renameTitle.trim();
    if (!trimmedTitle) {
      setChatListState({ loading: false, error: "Chat title cannot be empty." });
      return;
    }

    setChatListState({ loading: true, error: "" });
    try {
      const session = await updateChatSessionTitle(renameCandidate.chat_id, trimmedTitle);
      if (activeChat?.chat_id === session.chat_id) {
        setActiveChat(session);
      }
      setRenameCandidate(null);
      setRenameTitle("");
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

  async function uploadPdfFiles(files) {
    const pdfFiles = Array.from(files ?? []).filter((file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"));
    if (!pdfFiles.length) {
      setUploadState({ loading: false, error: "Please choose at least one PDF file.", message: "" });
      return;
    }

    setUploadState({ loading: true, error: "", message: `Uploading ${pdfFiles.length} PDF${pdfFiles.length > 1 ? "s" : ""}...` });
    setSourceState({ loading: true, error: "", message: "" });
    try {
      const chat = await ensureActiveChat();
      const uploadedPdfs = await uploadLocalPdfs(pdfFiles);
      let session = chat;
      for (const pdf of uploadedPdfs) {
        const indexResponse = await indexDownloadedPdf(pdf.filename);
        session = await addChatSource(session.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      }

      setActiveChat(session);
      setUploadState({
        loading: false,
        error: "",
        message: `${uploadedPdfs.length} PDF${uploadedPdfs.length > 1 ? "s" : ""} uploaded and added to this chat.`,
      });
      setSourceState({ loading: false, error: "", message: "" });
      await refreshDownloadedPdfs();
      await refreshChatThreads();
    } catch (error) {
      setUploadState({ loading: false, error: error.message, message: "" });
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

  function openCitation(citation) {
    const source = activeChat?.sources.find(
      (candidate) =>
        candidate.paper_id === citation.paper_id ||
        candidate.title === citation.title ||
        candidate.filename === citation.title,
    );
    const filename = source?.filename ?? (citation.title?.toLowerCase().endsWith(".pdf") ? citation.title : null);

    if (!filename) {
      setChatState({ loading: false, error: "Cannot open this citation because the PDF source is not available." });
      return;
    }

    setChatState({ loading: false, error: "" });
    setPaperOverlay({
      ...(source ?? {
        paper_id: citation.paper_id,
        title: citation.title || citation.paper_id,
      }),
      filename,
      citation,
      pageNumber: citation.page_number ?? citation.page,
    });
  }

  async function handleResearch(event) {
    event.preventDefault();
    const trimmedQuery = researchQuery.trim();
    if (!trimmedQuery) return;

    setResearchState({ loading: true, error: "" });
    setResearchResult(null);
    try {
      const response = await runResearch({ query: trimmedQuery, maxResults: researchMaxResults });
      setResearchResult({
        ...response,
        papers: (response.papers ?? []).map(normalizePaper),
      });
      setResearchState({ loading: false, error: "" });
    } catch (error) {
      setResearchState({ loading: false, error: error.message });
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
    const assistantMessage = {
      role: "assistant",
      content: "",
      citations: [],
      created_at: new Date(Date.now() + 1).toISOString(),
      streaming: true,
    };
    const chatId = activeChat.chat_id;
    setActiveChat((chat) => {
      if (!chat || chat.chat_id !== chatId) return chat;
      return { ...chat, messages: [...chat.messages, optimisticMessage, assistantMessage] };
    });
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      await streamChatWithPaper({
        question: trimmedQuestion,
        chatId,
        topK: 5,
        scoreThreshold: 0.25,
        onToken: (token) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === assistantMessage.created_at
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
                message.created_at === assistantMessage.created_at
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
      await refreshChatThreads();
    } catch (error) {
      setActiveChat((chat) => {
        if (!chat || chat.chat_id !== chatId) return chat;
        return {
          ...chat,
          messages: chat.messages.map((message) =>
            message.created_at === assistantMessage.created_at ? { ...message, streaming: false } : message,
          ),
        };
      });
      setChatState({ loading: false, error: error.message });
    }
  }

  if (mode === "home") {
    return <HomeScreen onStartChat={startChatMode} onStartResearch={() => setMode("research")} />;
  }

  if (mode === "research") {
    return (
      <ResearchWorkspace
        maxResults={researchMaxResults}
        onBack={returnHome}
        onChangeMaxResults={setResearchMaxResults}
        onChangeQuery={setResearchQuery}
        onSubmit={handleResearch}
        query={researchQuery}
        result={researchResult}
        state={researchState}
      />
    );
  }

  return (
    <main className="paper-chat-shell">
      <header className="paper-chat-topbar">
        <button aria-label="Trang chủ" className="paper-chat-brand" onClick={returnHome} type="button">
          <ArrowLeft size={17} aria-hidden="true" />
        </button>
        <h1>Chat với paper</h1>
        <button aria-label="New chat" className="paper-chat-top-action" onClick={createNewChat} type="button">
          <Plus size={18} aria-hidden="true" />
        </button>
      </header>

      <section className="paper-chat-layout">
      <aside className="chat-rail paper-chat-rail" aria-label="Chats">
        <div className="rail-header">
          <div>
            <h1>Cuộc trò chuyện</h1>
            <p>{chatThreads.length ? `${chatThreads.length} chats` : "No chats yet"}</p>
          </div>
          <div className="rail-actions">
            <button aria-label="New chat" onClick={createNewChat} type="button">
              <Plus size={18} aria-hidden="true" />
            </button>
          </div>
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

      <section className="workspace-panel paper-chat-workspace" aria-label="Current chat">
        {activeChat ? (
          <ChatWorkspace
            activeChat={activeChat}
            chatState={chatState}
            onAddSources={() => setIsSourceModalOpen(true)}
            onAsk={handleAsk}
            onClearHistory={handleClearHistory}
            onOpenCitation={openCitation}
            onOpenSource={setPaperOverlay}
            onRename={() => openRenameDialog(activeChat)}
            onRemoveSource={removeSource}
            question={question}
            setQuestion={setQuestion}
            sourceState={sourceState}
          />
        ) : (
          <EmptyWorkspace onCreateChat={createNewChat} />
        )}
      </section>
      </section>

      {isSourceModalOpen ? (
        <AddSourcesModal
          downloadedPdfs={downloadedPdfs}
          onAddLocalPdf={addLocalPdfToChat}
          onClose={() => setIsSourceModalOpen(false)}
          onRefreshPdfs={refreshDownloadedPdfs}
          onUploadPdfs={uploadPdfFiles}
          pdfListState={pdfListState}
          sourceIds={sourceIds}
          sourceState={sourceState}
          uploadState={uploadState}
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

      {renameCandidate ? (
        <RenameChatDialog
          onCancel={() => {
            setRenameCandidate(null);
            setRenameTitle("");
          }}
          onChange={setRenameTitle}
          onSubmit={renameChat}
          renaming={chatListState.loading}
          title={renameTitle}
        />
      ) : null}
    </main>
  );
}

function HomeScreen({ onStartChat, onStartResearch }) {
  return (
    <main className="home-shell">
      <section className="home-panel" aria-labelledby="home-title">
        <div className="home-heading">
          <span className="product-mark">
            <BrainCircuit size={20} aria-hidden="true" />
            AI Research Assistant
          </span>
          <h1 id="home-title">Chọn luồng làm việc</h1>
          <p>Tải PDF để hỏi đáp theo tài liệu, hoặc nhập query để agent thực hiện luồng nghiên cứu.</p>
        </div>

        <div className="mode-grid">
          <button className="mode-card" onClick={onStartChat} type="button">
            <span className="mode-icon">
              <MessageSquare size={26} aria-hidden="true" />
            </span>
            <span className="mode-title">Chat với paper</span>
            <span className="mode-copy">Tải local PDF lên, index tài liệu, rồi hỏi đáp với agent dựa trên nội dung paper.</span>
            <span className="mode-action">
              <UploadCloud size={17} aria-hidden="true" />
              Tải PDF
            </span>
          </button>

          <button className="mode-card research" onClick={onStartResearch} type="button">
            <span className="mode-icon">
              <Search size={26} aria-hidden="true" />
            </span>
            <span className="mode-title">Nghiên cứu</span>
            <span className="mode-copy">Nhập query để agent tìm paper, tổng hợp, so sánh và trả về kết quả nghiên cứu.</span>
            <span className="mode-action">
              <BrainCircuit size={17} aria-hidden="true" />
              Bắt đầu nghiên cứu
            </span>
          </button>
        </div>
      </section>
    </main>
  );
}

function ResearchWorkspace({ maxResults, onBack, onChangeMaxResults, onChangeQuery, onSubmit, query, result, state }) {
  const paperCount = result?.papers?.length ?? 0;
  const resultDate = new Intl.DateTimeFormat("vi-VN", {
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(new Date());

  return (
    <main className="research-mode-shell">
      <header className="research-topbar">
        <button aria-label="Trang chủ" className="research-brand" onClick={onBack} type="button">
          <ArrowLeft size={17} aria-hidden="true" />
        </button>
        <h1>{result?.query ? result.query : "Sổ ghi chú không có tiêu đề"}</h1>
        <button aria-label="Tùy chọn" className="research-icon-button" type="button">
          <MoreVertical size={18} aria-hidden="true" />
        </button>
      </header>

      <section className="research-notebook-shell" aria-label="Research workflow">
        <aside className="research-source-panel" aria-label="Nguồn">
          <div className="research-panel-header">
            <h2>Nguồn</h2>
            <button aria-label="Thu gọn nguồn" type="button">
              <PanelLeft size={16} aria-hidden="true" />
            </button>
          </div>

          <form className="research-side-search" onSubmit={onSubmit}>
            <button disabled={state.loading || !query.trim()} type="submit">
              <Plus size={16} aria-hidden="true" />
              Thêm nguồn
            </button>
            <div className="research-source-searchbox">
              <input
                onChange={(event) => onChangeQuery(event.target.value)}
                placeholder="Tìm nguồn mới trên web"
                value={query}
              />
              <div className="research-filter-row">
                <span>
                  <Globe2 size={15} aria-hidden="true" />
                  Web
                  <ChevronDown size={14} aria-hidden="true" />
                </span>
                <label>
                  <BrainCircuit size={15} aria-hidden="true" />
                  Nghiên cứu nhanh
                  <input
                    aria-label="Số kết quả"
                    max="20"
                    min="1"
                    onChange={(event) => onChangeMaxResults(Number(event.target.value))}
                    type="number"
                    value={maxResults}
                  />
                </label>
              </div>
              <Search size={18} aria-hidden="true" />
            </div>
          </form>

          <div className={`research-source-list ${paperCount ? "" : "empty"}`}>
            {paperCount ? (
              result.papers.map((paper) => (
                <a className="research-source-item" href={paper.arxivUrl ?? paper.url ?? paper.pdfUrl} key={paper.paper_id} rel="noreferrer" target="_blank">
                  <FileText size={16} aria-hidden="true" />
                  <span>{paper.title}</span>
                </a>
              ))
            ) : (
              <>
                <FileText size={25} aria-hidden="true" />
                <strong>Các nguồn đã lưu sẽ xuất hiện ở đây</strong>
                <p>Nhập query và chạy nghiên cứu để thêm paper vào danh sách nguồn.</p>
              </>
            )}
          </div>
        </aside>

        <section className="research-conversation" aria-label="Cuộc trò chuyện">
          <header className="research-conversation-header">
            <h2>Cuộc trò chuyện</h2>
            <button className="research-chip-button" type="button">
              <BrainCircuit size={15} aria-hidden="true" />
              Tùy chỉnh
            </button>
          </header>

          <div className="research-conversation-body">
            <div className="research-note-title">
              <div className="research-note-icon">
                <BrainCircuit size={24} aria-hidden="true" />
              </div>
              <h2>{result?.query ? result.query : "Sổ ghi chú không có tiêu đề"}</h2>
              <p>{paperCount} nguồn · {resultDate}</p>
            </div>

            {!result ? (
              <form className="research-hero-card" onSubmit={onSubmit}>
                <button aria-label="Đóng gợi ý" className="research-card-close" type="button">
                  <X size={20} aria-hidden="true" />
                </button>
                <h3>
                  Tạo bản tổng quan nghiên cứu từ
                  <span> query của bạn</span>
                </h3>
                <div className="research-web-input">
                  <input
                    onChange={(event) => onChangeQuery(event.target.value)}
                    placeholder="Tìm nguồn mới trên web"
                    value={query}
                  />
                  <div className="research-filter-row">
                    <span>
                      <Globe2 size={15} aria-hidden="true" />
                      Web
                      <ChevronDown size={14} aria-hidden="true" />
                    </span>
                    <label>
                      <BrainCircuit size={15} aria-hidden="true" />
                      Nghiên cứu nhanh
                      <input
                        aria-label="Số kết quả"
                        max="20"
                        min="1"
                        onChange={(event) => onChangeMaxResults(Number(event.target.value))}
                        type="number"
                        value={maxResults}
                      />
                    </label>
                  </div>
                  <button aria-label="Chạy nghiên cứu" disabled={state.loading || !query.trim()} type="submit">
                    <Search size={18} aria-hidden="true" />
                  </button>
                </div>
                <div className="research-dropzone">
                  <strong>hoặc thêm tài liệu của bạn</strong>
                  <p>pdf, hình ảnh, tài liệu, âm thanh, và các định dạng khác</p>
                  <div className="research-source-actions">
                    <span>
                      <UploadCloud size={15} aria-hidden="true" />
                      Tải tệp lên
                    </span>
                    <span>
                      <Link2 size={15} aria-hidden="true" />
                      Trang web
                    </span>
                    <span>
                      <Library size={15} aria-hidden="true" />
                      Drive
                    </span>
                    <span>
                      <Clipboard size={15} aria-hidden="true" />
                      Văn bản đã sao chép
                    </span>
                  </div>
                </div>
              </form>
            ) : null}

            {state.error ? <div className="research-error">{state.error}</div> : null}

            {state.loading ? (
              <div className="research-loading">
                <BrainCircuit size={24} aria-hidden="true" />
                <span>Agent đang tìm paper, phân tích và tổng hợp kết quả...</span>
              </div>
            ) : null}

            {result ? (
              <div className="research-results">
                <div className="research-results-header">
                  <div>
                    <h2>Kết quả cho "{result.query}"</h2>
                    <p>{paperCount ? `${paperCount} papers found` : "No papers returned"}</p>
                  </div>
                </div>

                {result.summary ? <ResearchTextBlock title="Summary" content={result.summary} /> : null}
                {result.comparison ? <ResearchTextBlock title="Comparison" content={result.comparison} /> : null}
                {result.report ? <ResearchTextBlock title="Report" content={result.report} /> : null}

                <section className="paper-result-grid" aria-label="Research papers">
                  {result.papers.map((paper) => (
                    <ResearchPaperCard key={paper.paper_id} paper={paper} />
                  ))}
                </section>
              </div>
            ) : null}
          </div>

          <form className="research-bottom-composer" onSubmit={onSubmit}>
            <input onChange={(event) => onChangeQuery(event.target.value)} placeholder="Bắt đầu nhập..." value={query} />
            <span>{paperCount} nguồn</span>
            <button aria-label="Gửi" disabled={state.loading || !query.trim()} type="submit">
              <ArrowUp size={18} aria-hidden="true" />
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}

function ResearchTextBlock({ title, content }) {
  return (
    <section className="research-text-block">
      <h3>{title}</h3>
      <p>{content}</p>
    </section>
  );
}

function ResearchPaperCard({ paper }) {
  const paperUrl = paper.arxivUrl ?? paper.url;

  return (
    <article className="research-paper-card">
      <h3>{paper.title}</h3>
      <div className="paper-meta">
        <CalendarDays size={14} aria-hidden="true" />
        {paper.published}
      </div>
      <p className="author-line">{paper.authors?.join(", ") || "Unknown authors"}</p>
      {paper.abstract ? <p className="paper-abstract">{paper.abstract}</p> : null}
      <div className="paper-links">
        {paperUrl ? (
          <a href={paperUrl} rel="noreferrer" target="_blank">
            Paper
          </a>
        ) : null}
        {paper.pdfUrl ? (
          <a href={paper.pdfUrl} rel="noreferrer" target="_blank">
            PDF
          </a>
        ) : null}
      </div>
    </article>
  );
}

function ChatWorkspace({
  activeChat,
  chatState,
  onAddSources,
  onAsk,
  onClearHistory,
  onOpenCitation,
  onOpenSource,
  onRename,
  onRemoveSource,
  question,
  setQuestion,
  sourceState,
}) {
  const canChat = activeChat.sources.length > 0 && !sourceState.loading;
  const chatLogRef = useRef(null);
  const questionBoxRef = useRef(null);

  useEffect(() => {
    const chatLog = chatLogRef.current;
    if (!chatLog) return;
    chatLog.scrollTop = chatLog.scrollHeight;
  }, [activeChat.chat_id, activeChat.messages, chatState.loading]);

  useEffect(() => {
    const questionBox = questionBoxRef.current;
    if (!questionBox || question) return;
    questionBox.style.height = "44px";
  }, [question]);

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
          <button className="secondary-action" onClick={onRename} type="button">
            <Pencil size={16} aria-hidden="true" />
            Rename
          </button>
          <button className="secondary-action" disabled={activeChat.messages.length === 0 || chatState.loading} onClick={onClearHistory} type="button">
            Clear
          </button>
          <button className="primary-action" onClick={onAddSources} type="button">
            <Plus size={17} aria-hidden="true" />
            Add PDFs
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
            Upload or choose local PDFs
          </button>
        )}
      </section>

      {sourceState.message ? <div className="success-box compact-box">{sourceState.message}</div> : null}
      {sourceState.error ? <div className="error-box compact-box">{sourceState.error}</div> : null}

      <div className="chat-log" ref={chatLogRef}>
        {activeChat.messages.length === 0 ? (
          <div className="empty-state compact">
            <Bot size={22} aria-hidden="true" />
            <span>{canChat ? "Ask a grounded question across this chat's sources." : "Attach at least one source to begin."}</span>
          </div>
        ) : null}
        {activeChat.messages.map((message, index) => (
          <ChatMessage key={`${message.role}-${message.created_at}-${index}`} message={message} onOpenCitation={onOpenCitation} />
        ))}
      </div>

      {chatState.error ? <div className="error-box compact-error">{chatState.error}</div> : null}

      <form className="chat-form" onSubmit={onAsk}>
        <textarea
          disabled={!canChat || chatState.loading}
          ref={questionBoxRef}
          onInput={(event) => {
            event.currentTarget.style.height = "44px";
            event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 150)}px`;
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              event.currentTarget.form?.requestSubmit();
            }
          }}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder={canChat ? "Ask across the selected sources..." : "Add sources before chatting"}
          rows="1"
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
  onAddLocalPdf,
  onClose,
  onRefreshPdfs,
  onUploadPdfs,
  pdfListState,
  sourceIds,
  sourceState,
  uploadState,
}) {
  return (
    <div className="overlay-backdrop" role="dialog" aria-modal="true" aria-label="Add sources">
      <section className="sources-modal">
        <header className="modal-header">
          <div>
            <h2>Add local PDFs</h2>
            <p>Upload paper PDFs, attach them to this chat, then ask grounded questions.</p>
          </div>
          <button aria-label="Close" onClick={onClose} type="button">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        {sourceState.message ? <div className="success-box compact-box">{sourceState.message}</div> : null}
        {sourceState.error ? <div className="error-box compact-box">{sourceState.error}</div> : null}
        {uploadState.message ? <div className="success-box compact-box">{uploadState.message}</div> : null}
        {uploadState.error ? <div className="error-box compact-box">{uploadState.error}</div> : null}

        <section className="modal-body">
          <label className={`upload-zone ${uploadState.loading ? "loading" : ""}`}>
            <UploadCloud size={28} aria-hidden="true" />
            <span className="upload-zone-title">{uploadState.loading ? "Uploading and indexing..." : "Upload PDF files"}</span>
            <span className="upload-zone-meta">Choose one or more local paper PDFs.</span>
            <input
              accept="application/pdf,.pdf"
              disabled={uploadState.loading || sourceState.loading}
              multiple
              onChange={(event) => {
                void onUploadPdfs(event.target.files);
                event.target.value = "";
              }}
              type="file"
            />
          </label>

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
  const pageNumber = source.pageNumber ?? source.citation?.page_number ?? source.citation?.page;
  const pdfFragment = pageNumber ? `#page=${pageNumber}&view=FitH` : "#view=FitH";
  const citation = source.citation;

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
            <p>
              {source.path ?? source.paper_id}
              {pageNumber ? ` · page ${pageNumber}` : ""}
            </p>
          </div>
          {source.path ? (
            <span className="index-badge neutral">
              <ArrowDownToLine size={13} aria-hidden="true" />
              Local file
            </span>
          ) : null}
        </div>

        {citation?.text ? (
          <EvidencePanel citation={citation} pageNumber={pageNumber} />
        ) : null}

        {pdfUrl ? (
          <iframe className="pdf-frame" src={`${pdfUrl}${pdfFragment}`} title="Paper PDF" />
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

function EvidencePanel({ citation, pageNumber }) {
  const score = citation.rerank_score ?? citation.score;
  const hasWeakEvidence = isWeakEvidence(citation);

  return (
    <section className="evidence-panel" aria-label="Evidence details">
      <div className="evidence-panel-header">
        <div>
          <strong>Referenced passage{pageNumber ? ` on page ${pageNumber}` : ""}</strong>
          <div className="citation-preview-meta">
            {citation.chunk_id ? <span>{citation.chunk_id}</span> : null}
            <span className={`evidence-badge ${evidenceQualityClass(citation.evidence_quality)}`}>
              {formatEvidenceQuality(citation.evidence_quality)}
            </span>
          </div>
        </div>
        {hasWeakEvidence ? (
          <div className="evidence-warning" role="status">
            <AlertTriangle size={15} aria-hidden="true" />
            Weak context
          </div>
        ) : null}
      </div>

      <div className="evidence-metrics" aria-label="Evidence quality metrics">
        <Metric label="Score" value={formatScore(score)} />
        <Metric label="Rerank" value={formatScore(citation.rerank_score)} />
        <Metric label="Vector" value={formatScore(citation.vector_score)} />
        <Metric label="Keyword" value={formatScore(citation.keyword_score)} />
        <Metric label="Source" value={formatRetrievalSources(citation.retrieval_sources)} wide />
      </div>

      <p className="evidence-passage">
        <HighlightedText text={citation.text} terms={citation.matched_terms} />
      </p>
    </section>
  );
}

function Metric({ label, value, wide = false }) {
  return (
    <div className={`evidence-metric ${wide ? "wide" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
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

function RenameChatDialog({ onCancel, onChange, onSubmit, renaming, title }) {
  return (
    <div className="overlay-backdrop" role="dialog" aria-modal="true" aria-label="Rename chat">
      <form className="rename-dialog" onSubmit={onSubmit}>
        <header className="rename-header">
          <div>
            <h2>Rename chat</h2>
            <p>Give this conversation a name that is easier to find later.</p>
          </div>
          <button aria-label="Close" disabled={renaming} onClick={onCancel} type="button">
            <X size={18} aria-hidden="true" />
          </button>
        </header>
        <label className="rename-field">
          Chat name
          <input
            autoFocus
            maxLength={160}
            onChange={(event) => onChange(event.target.value)}
            placeholder="Literature review"
            value={title}
          />
        </label>
        <div className="confirm-actions">
          <button className="secondary-action" disabled={renaming} onClick={onCancel} type="button">
            Cancel
          </button>
          <button className="primary-action" disabled={renaming || !title.trim()} type="submit">
            <Pencil size={16} aria-hidden="true" />
            {renaming ? "Saving" : "Save"}
          </button>
        </div>
      </form>
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

function ChatMessage({ message, onOpenCitation }) {
  const isUser = message.role === "user";
  const bubbleContent = message.content || (message.streaming ? "Thinking..." : "");
  return (
    <div className={`message ${isUser ? "user" : "assistant"}`}>
      {isUser ? <User size={18} aria-hidden="true" /> : <Bot size={18} aria-hidden="true" />}
      <div className="message-content">
        <div className="message-bubble">
          {bubbleContent}
          {message.streaming && message.content ? <span className="typing-cursor" aria-hidden="true" /> : null}
        </div>
        {message.citations?.length ? (
          <div className="citation-list">
            {message.citations.map((citation) => (
              <button
                className={`citation-pill ${evidenceQualityClass(citation.evidence_quality)}`}
                key={citation.chunk_id ?? `${citation.paper_id}-${citation.page_number}`}
                onClick={() => onOpenCitation?.(citation)}
                title={citation.chunk_id ? `Open evidence chunk ${citation.chunk_id}` : "Open evidence"}
                type="button"
              >
                <span className="citation-main">
                  {citation.title || citation.paper_id}
                  {citation.page_number ? `, p. ${citation.page_number}` : ""}
                </span>
                {citation.chunk_id ? <span className="citation-chunk">{citation.chunk_id}</span> : null}
                <span className="citation-quality">{formatEvidenceQuality(citation.evidence_quality)}</span>
              </button>
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

function formatEvidenceQuality(quality) {
  if (quality === "high") return "High";
  if (quality === "medium") return "Medium";
  if (quality === "low") return "Low";
  return "Unknown";
}

function evidenceQualityClass(quality) {
  if (quality === "high") return "quality-high";
  if (quality === "medium") return "quality-medium";
  if (quality === "low") return "quality-low";
  return "quality-unknown";
}

function formatScore(score) {
  return Number.isFinite(score) ? score.toFixed(2) : "n/a";
}

function formatRetrievalSources(sources) {
  return sources?.length ? sources.join(" + ") : "n/a";
}

function isWeakEvidence(citation) {
  const score = citation?.rerank_score ?? citation?.score;
  return citation?.evidence_quality === "low" || citation?.evidence_quality === "unknown" || (Number.isFinite(score) && score < 0.5);
}

function HighlightedText({ text, terms }) {
  if (!text) return null;
  const highlightTerms = Array.from(new Set((terms ?? []).filter(Boolean))).sort((a, b) => b.length - a.length);
  if (!highlightTerms.length) return text;

  const pattern = new RegExp(`(${highlightTerms.map(escapeRegExp).join("|")})`, "gi");
  return text.split(pattern).map((part, index) => {
    const isMatch = highlightTerms.some((term) => term.toLowerCase() === part.toLowerCase());
    return isMatch ? <mark key={`${part}-${index}`}>{part}</mark> : part;
  });
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export default App;
