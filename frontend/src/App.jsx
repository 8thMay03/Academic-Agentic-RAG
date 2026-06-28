import {
  ArrowDownToLine,
  ArrowLeft,
  Bot,
  CalendarDays,
  ExternalLink,
  FileText,
  MessageSquare,
  RefreshCw,
  Search,
  Send,
  User,
} from "lucide-react";
import { useEffect, useState } from "react";
import {
  chatWithPaper,
  downloadPapers,
  indexDownloadedPdf,
  listDownloadedPdfs,
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

function paperFromDownloadedPdf(pdf) {
  const paperId = pdf.filename.replace(/\.pdf$/i, "");
  return {
    type: "local",
    paperId,
    filename: pdf.filename,
    title: pdf.filename,
    subtitle: `${formatBytes(pdf.size_bytes)} · ${formatDateTime(pdf.modified_at)}`,
    path: pdf.path,
  };
}

function paperFromOnlinePaper(paper) {
  return {
    type: "online",
    paperId: paper.paper_id,
    filename: null,
    title: paper.title,
    subtitle: `${paper.published} · ${paper.authors?.join(", ") || "Unknown authors"}`,
    abstract: paper.abstract,
    pdfUrl: paper.pdfUrl,
    arxivUrl: paper.arxivUrl,
  };
}

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [maxResults, setMaxResults] = useState(5);
  const [sortBy, setSortBy] = useState("submittedDate");
  const [onlinePapers, setOnlinePapers] = useState([]);
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [activePaper, setActivePaper] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [question, setQuestion] = useState("");

  const [searchState, setSearchState] = useState({ loading: false, error: "" });
  const [pdfListState, setPdfListState] = useState({ loading: false, error: "" });
  const [prepareState, setPrepareState] = useState({ loading: false, error: "", message: "" });
  const [chatState, setChatState] = useState({ loading: false, error: "" });

  useEffect(() => {
    void refreshDownloadedPdfs();
  }, []);

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

  async function openLocalPdf(pdf) {
    const paper = paperFromDownloadedPdf(pdf);
    setActivePaper(paper);
    setChatMessages([]);
    setPrepareState({ loading: true, error: "", message: "Indexing PDF for chat..." });

    try {
      const response = await indexDownloadedPdf(pdf.filename);
      setActivePaper({ ...paper, paperId: response.paper_id });
      setPrepareState({
        loading: false,
        error: "",
        message: `Ready for chat: ${response.chunks_indexed} chunks indexed.`,
      });
    } catch (error) {
      setPrepareState({ loading: false, error: error.message, message: "" });
    }
  }

  async function openOnlinePaper(paper) {
    const normalized = paperFromOnlinePaper(paper);
    setActivePaper(normalized);
    setChatMessages([]);

    if (!paper.pdfUrl) {
      setPrepareState({ loading: false, error: "This paper does not have a PDF URL.", message: "" });
      return;
    }

    setPrepareState({ loading: true, error: "", message: "Downloading and indexing PDF..." });

    try {
      const downloadResponse = await downloadPapers([paper.pdfUrl]);
      const downloadedPath = downloadResponse.files?.[0];
      if (!downloadedPath) {
        throw new Error(downloadResponse.errors?.[0]?.error ?? "PDF download failed.");
      }

      await refreshDownloadedPdfs();
      const filename = downloadedPath.split("/").pop();
      const indexResponse = await indexDownloadedPdf(filename);
      setActivePaper({
        ...normalized,
        type: "online-indexed",
        paperId: indexResponse.paper_id,
        filename,
        path: downloadedPath,
      });
      setPrepareState({
        loading: false,
        error: downloadResponse.errors?.[0]?.error ?? "",
        message: `Ready for chat: ${indexResponse.chunks_indexed} chunks indexed.`,
      });
    } catch (error) {
      setPrepareState({ loading: false, error: error.message, message: "" });
    }
  }

  function closeChat() {
    setActivePaper(null);
    setChatMessages([]);
    setQuestion("");
    setPrepareState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !activePaper || prepareState.loading) return;

    setChatMessages((messages) => [...messages, { role: "user", content: trimmedQuestion }]);
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      const response = await chatWithPaper({
        question: trimmedQuestion,
        paperIds: [activePaper.paperId],
        topK: 5,
        scoreThreshold: 0.65,
      });
      setChatMessages((messages) => [
        ...messages,
        {
          role: "assistant",
          content: response.answer,
          citations: response.citations ?? [],
        },
      ]);
      setChatState({ loading: false, error: "" });
    } catch (error) {
      setChatState({ loading: false, error: error.message });
    }
  }

  if (activePaper) {
    return (
      <ChatWorkspace
        activePaper={activePaper}
        chatMessages={chatMessages}
        chatState={chatState}
        onAsk={handleAsk}
        onBack={closeChat}
        prepareState={prepareState}
        question={question}
        setQuestion={setQuestion}
      />
    );
  }

  return (
    <main className="home-shell">
      <header className="home-header">
        <div>
          <h1>AI Research Assistant</h1>
          <p>Open an existing PDF or search arXiv, then chat with the selected paper.</p>
        </div>
      </header>

      <section className="home-grid">
        <section className="online-panel" aria-label="Search online PDFs">
          <div className="section-title-row prominent">
            <div>
              <h2>Search Online PDFs</h2>
              <p>Find arXiv papers and open one to chat.</p>
            </div>
            <Search size={20} aria-hidden="true" />
          </div>

          <form className="search-form" onSubmit={handleSearch}>
            <label className="field-label" htmlFor="paper-query">
              Search query
            </label>
            <div className="search-input-row">
              <Search size={18} aria-hidden="true" />
              <input
                id="paper-query"
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Agentic RAG"
                value={query}
              />
              <button disabled={searchState.loading || !query.trim()} type="submit">
                {searchState.loading ? "Searching" : "Search"}
              </button>
            </div>

            <div className="search-controls">
              <label>
                Results
                <input
                  max="20"
                  min="1"
                  onChange={(event) => setMaxResults(Number(event.target.value))}
                  type="number"
                  value={maxResults}
                />
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

          <div className="online-result-list">
            {onlinePapers.length === 0 && !searchState.loading ? (
              <div className="empty-state compact">
                <FileText size={22} aria-hidden="true" />
                <span>Search results will appear here.</span>
              </div>
            ) : null}

            {onlinePapers.map((paper) => (
              <OnlinePaperCard key={paper.paper_id} onClick={() => openOnlinePaper(paper)} paper={paper} />
            ))}
          </div>
        </section>

        <section className="library-panel" aria-label="Available PDFs">
          <div className="section-title-row prominent">
            <div>
              <h2>PDFs in System</h2>
              <p>{downloadedPdfs.length ? `${downloadedPdfs.length} local files ready` : "No PDFs found yet"}</p>
            </div>
            <button aria-label="Refresh PDFs" disabled={pdfListState.loading} onClick={refreshDownloadedPdfs} type="button">
              <RefreshCw size={16} aria-hidden="true" />
            </button>
          </div>

          {pdfListState.error ? <div className="error-box">{pdfListState.error}</div> : null}

          <div className="pdf-card-grid">
            {downloadedPdfs.length === 0 && !pdfListState.loading ? (
              <div className="empty-state">
                <FileText size={26} aria-hidden="true" />
                <span>No downloaded PDFs found.</span>
              </div>
            ) : null}

            {downloadedPdfs.map((pdf) => (
              <PdfCard key={pdf.path} onClick={() => openLocalPdf(pdf)} pdf={pdf} />
            ))}

            {pdfListState.loading ? (
              <div className="pdf-card muted">
                <span>Loading PDFs...</span>
              </div>
            ) : null}
          </div>
        </section>
      </section>
    </main>
  );
}

function ChatWorkspace({
  activePaper,
  chatMessages,
  chatState,
  onAsk,
  onBack,
  prepareState,
  question,
  setQuestion,
}) {
  const chatDisabled = prepareState.loading || Boolean(prepareState.error);

  return (
    <main className="chat-workspace">
      <section className="chat-paper-panel" aria-label="Selected paper">
        <button className="back-button" onClick={onBack} type="button">
          <ArrowLeft size={17} aria-hidden="true" />
          Back
        </button>

        <div className="selected-paper-card">
          <div className="paper-kind">{activePaper.type === "local" ? "Local PDF" : "Online PDF"}</div>
          <h1>{activePaper.title}</h1>
          <p>{activePaper.subtitle}</p>

          <div className="action-row">
            {activePaper.arxivUrl ? (
              <a href={activePaper.arxivUrl} rel="noreferrer" target="_blank">
                <ExternalLink size={16} aria-hidden="true" />
                arXiv
              </a>
            ) : null}
            {activePaper.pdfUrl ? (
              <a href={activePaper.pdfUrl} rel="noreferrer" target="_blank">
                <FileText size={16} aria-hidden="true" />
                PDF
              </a>
            ) : null}
            {activePaper.path ? (
              <span className="index-badge neutral">
                <ArrowDownToLine size={13} aria-hidden="true" />
                Local file
              </span>
            ) : null}
          </div>

          {prepareState.message ? <div className="success-box">{prepareState.message}</div> : null}
          {prepareState.error ? <div className="error-box">{prepareState.error}</div> : null}

          {activePaper.abstract ? (
            <section className="abstract-section">
              <h2>Abstract</h2>
              <p>{activePaper.abstract}</p>
            </section>
          ) : null}

          {activePaper.path ? <div className="local-path-box">{activePaper.path}</div> : null}
        </div>
      </section>

      <section className="chat-main-panel" aria-label="Chat with selected PDF">
        <div className="panel-header">
          <div>
            <h2>Chat with paper</h2>
            <p>{activePaper.title}</p>
          </div>
          <MessageSquare size={20} aria-hidden="true" />
        </div>

        <div className="chat-log">
          {chatMessages.length === 0 ? (
            <div className="empty-state compact">
              <Bot size={22} aria-hidden="true" />
              <span>{chatDisabled ? "Waiting for PDF indexing." : "Ask a grounded question about this paper."}</span>
            </div>
          ) : null}

          {chatMessages.map((message, index) => (
            <ChatMessage key={`${message.role}-${index}`} message={message} />
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
            disabled={chatDisabled || chatState.loading}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={chatDisabled ? "Indexing must finish before chat" : "Ask about method, results, limitations..."}
            value={question}
          />
          <button disabled={chatDisabled || !question.trim() || chatState.loading} type="submit">
            <Send size={17} aria-hidden="true" />
            <span>Ask</span>
          </button>
        </form>
      </section>
    </main>
  );
}

function PdfCard({ onClick, pdf }) {
  return (
    <button className="pdf-card" onClick={onClick} type="button">
      <FileText size={22} aria-hidden="true" />
      <span className="pdf-card-title">{pdf.filename}</span>
      <span className="pdf-card-meta">
        {formatBytes(pdf.size_bytes)} · {formatDateTime(pdf.modified_at)}
      </span>
      <span className="pdf-card-path">{pdf.path}</span>
    </button>
  );
}

function OnlinePaperCard({ onClick, paper }) {
  return (
    <button className="online-paper-card" onClick={onClick} type="button">
      <span className="paper-title">{paper.title}</span>
      <span className="paper-meta">
        <CalendarDays size={14} aria-hidden="true" />
        {paper.published}
      </span>
      <span className="author-line">{paper.authors?.join(", ") || "Unknown authors"}</span>
      <span className="online-paper-action">Download, index, and chat</span>
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
              <span key={citation.chunk_id ?? `${citation.paper_id}-${citation.page_number}`}>
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
