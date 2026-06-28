import {
  ArrowDownToLine,
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
import { useEffect, useMemo, useState } from "react";
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
    paper_id: paperId,
    title: pdf.filename,
    authors: [],
    abstract: "Local downloaded PDF. Select it to index the file, then ask questions in chat.",
    published: formatDateTime(pdf.modified_at),
    pdfPath: pdf.path,
    source: "local",
  };
}

function App() {
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [maxResults, setMaxResults] = useState(5);
  const [sortBy, setSortBy] = useState("submittedDate");
  const [papers, setPapers] = useState([]);
  const [selectedPaperId, setSelectedPaperId] = useState(null);
  const [searchState, setSearchState] = useState({ loading: false, error: "" });
  const [downloadState, setDownloadState] = useState({ loading: false, message: "", error: "" });
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [pdfListState, setPdfListState] = useState({ loading: false, error: "" });
  const [selectedPdfFilename, setSelectedPdfFilename] = useState(null);
  const [pdfIndexState, setPdfIndexState] = useState({});
  const [chatMessages, setChatMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [chatState, setChatState] = useState({ loading: false, error: "" });

  const selectedDownloadedPdf = useMemo(
    () => downloadedPdfs.find((pdf) => pdf.filename === selectedPdfFilename),
    [downloadedPdfs, selectedPdfFilename],
  );

  const selectedPaper = useMemo(() => {
    if (selectedDownloadedPdf) {
      return paperFromDownloadedPdf(selectedDownloadedPdf);
    }
    return papers.find((paper) => paper.paper_id === selectedPaperId) ?? papers[0];
  }, [papers, selectedDownloadedPdf, selectedPaperId]);

  const selectedPdfIndexState = selectedPdfFilename
    ? pdfIndexState[selectedPdfFilename] ?? { loading: false, error: "", chunksIndexed: null }
    : null;

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
    setDownloadState({ loading: false, message: "", error: "" });
    setChatMessages([]);

    try {
      const response = await searchPapers({ query, maxResults, sortBy });
      const normalizedPapers = (response.papers ?? []).map(normalizePaper);
      setPapers(normalizedPapers);
      setSelectedPaperId(normalizedPapers[0]?.paper_id ?? null);
      setSelectedPdfFilename(null);
    } catch (error) {
      setSearchState({ loading: false, error: error.message });
      return;
    }

    setSearchState({ loading: false, error: "" });
  }

  async function handleDownloadSelected() {
    if (!selectedPaper?.pdfUrl) return;
    setDownloadState({ loading: true, message: "", error: "" });

    try {
      const response = await downloadPapers([selectedPaper.pdfUrl]);
      const cachedCount = response.cached_files?.length ?? 0;
      const file = response.files?.[0] ?? "Downloaded";
      setDownloadState({
        loading: false,
        message: cachedCount ? `Cached: ${file}` : `Saved: ${file}`,
        error: response.errors?.[0]?.error ?? "",
      });
      await refreshDownloadedPdfs();
    } catch (error) {
      setDownloadState({ loading: false, message: "", error: error.message });
    }
  }

  async function handleSelectDownloadedPdf(pdf) {
    setSelectedPdfFilename(pdf.filename);
    setSelectedPaperId(null);
    setChatMessages([]);
    setDownloadState({ loading: false, message: "", error: "" });

    setPdfIndexState((state) => ({
      ...state,
      [pdf.filename]: { loading: true, error: "", chunksIndexed: null },
    }));

    try {
      const response = await indexDownloadedPdf(pdf.filename);
      setPdfIndexState((state) => ({
        ...state,
        [pdf.filename]: {
          loading: false,
          error: "",
          chunksIndexed: response.chunks_indexed,
        },
      }));
    } catch (error) {
      setPdfIndexState((state) => ({
        ...state,
        [pdf.filename]: { loading: false, error: error.message, chunksIndexed: null },
      }));
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !selectedPaper) return;

    const userMessage = { role: "user", content: trimmedQuestion };
    setChatMessages((messages) => [...messages, userMessage]);
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      const response = await chatWithPaper({
        question: trimmedQuestion,
        paperIds: [selectedPaper.paper_id],
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

  return (
    <main className="app-shell">
      <section className="search-panel" aria-label="Paper search">
        <div className="brand-row">
          <div>
            <h1>AI Research Assistant</h1>
            <p>Search arXiv papers, inspect details, and ask grounded questions.</p>
          </div>
        </div>

        <form className="search-form" onSubmit={handleSearch}>
          <label className="field-label" htmlFor="paper-query">
            Search papers
          </label>
          <div className="search-input-row">
            <Search size={18} aria-hidden="true" />
            <input
              id="paper-query"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Agentic RAG"
            />
            <button type="submit" disabled={searchState.loading || !query.trim()}>
              {searchState.loading ? "Searching" : "Search"}
            </button>
          </div>

          <div className="search-controls">
            <label>
              Results
              <input
                type="number"
                min="1"
                max="20"
                value={maxResults}
                onChange={(event) => setMaxResults(Number(event.target.value))}
              />
            </label>
            <label>
              Sort
              <select value={sortBy} onChange={(event) => setSortBy(event.target.value)}>
                <option value="submittedDate">Submitted date</option>
                <option value="lastUpdatedDate">Last updated</option>
                <option value="relevance">Relevance</option>
              </select>
            </label>
          </div>
        </form>

        {searchState.error ? <div className="error-box">{searchState.error}</div> : null}

        <div className="result-list" aria-label="Search results">
          {papers.length === 0 && !searchState.loading ? (
            <div className="empty-state">
              <FileText size={22} aria-hidden="true" />
              <span>No papers loaded yet.</span>
            </div>
          ) : null}

          {papers.map((paper) => (
            <button
              className={`paper-result ${paper.paper_id === selectedPaper?.paper_id ? "active" : ""}`}
              key={paper.paper_id}
              onClick={() => {
                setSelectedPaperId(paper.paper_id);
                setSelectedPdfFilename(null);
                setChatMessages([]);
                setDownloadState({ loading: false, message: "", error: "" });
              }}
              type="button"
            >
              <span className="paper-title">{paper.title}</span>
              <span className="paper-meta">
                <CalendarDays size={14} aria-hidden="true" />
                {paper.published}
              </span>
              <span className="author-line">{paper.authors?.join(", ") || "Unknown authors"}</span>
            </button>
          ))}
        </div>

        <DownloadedPdfList
          error={pdfListState.error}
          loading={pdfListState.loading}
          onRefresh={refreshDownloadedPdfs}
          onSelect={handleSelectDownloadedPdf}
          pdfs={downloadedPdfs}
          selectedFilename={selectedPdfFilename}
          states={pdfIndexState}
        />
      </section>

      <section className="detail-panel" aria-label="Paper detail">
        {selectedPaper ? (
          <PaperDetail
            downloadState={downloadState}
            indexState={selectedPdfIndexState}
            onDownload={handleDownloadSelected}
            paper={selectedPaper}
          />
        ) : (
          <div className="detail-empty">
            <FileText size={34} aria-hidden="true" />
            <h2>Select a paper</h2>
            <p>Run a search and choose a result to view the abstract, links, and chat context.</p>
          </div>
        )}
      </section>

      <section className="chat-panel" aria-label="Chat with paper">
        <div className="panel-header">
          <div>
            <h2>Chat</h2>
            <p>{selectedPaper ? selectedPaper.title : "Select a paper to begin"}</p>
          </div>
          <MessageSquare size={20} aria-hidden="true" />
        </div>

        <div className="chat-log">
          {chatMessages.length === 0 ? (
            <div className="empty-state compact">
              <Bot size={21} aria-hidden="true" />
              <span>Ask a question grounded in the selected paper.</span>
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

        <form className="chat-form" onSubmit={handleAsk}>
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={selectedPaper ? "Ask about method, results, limitations..." : "Select a paper first"}
            disabled={!selectedPaper || chatState.loading || selectedPdfIndexState?.loading}
          />
          <button
            type="submit"
            disabled={!selectedPaper || !question.trim() || chatState.loading || selectedPdfIndexState?.loading}
          >
            <Send size={17} aria-hidden="true" />
            <span>Ask</span>
          </button>
        </form>
      </section>
    </main>
  );
}

function DownloadedPdfList({ error, loading, onRefresh, onSelect, pdfs, selectedFilename, states }) {
  return (
    <section className="downloaded-section" aria-label="Downloaded PDFs">
      <div className="section-title-row">
        <div>
          <h2>Downloaded PDFs</h2>
          <p>{pdfs.length ? `${pdfs.length} local files` : "No local PDFs yet"}</p>
        </div>
        <button aria-label="Refresh downloaded PDFs" disabled={loading} onClick={onRefresh} type="button">
          <RefreshCw size={16} aria-hidden="true" />
        </button>
      </div>

      {error ? <div className="error-box">{error}</div> : null}

      <div className="downloaded-list">
        {pdfs.length === 0 && !loading ? (
          <div className="empty-state small">
            <FileText size={20} aria-hidden="true" />
            <span>No downloaded PDFs found.</span>
          </div>
        ) : null}

        {pdfs.map((pdf) => (
          <button
            className={`downloaded-item ${pdf.filename === selectedFilename ? "active" : ""}`}
            key={pdf.path}
            onClick={() => onSelect(pdf)}
            type="button"
          >
            <div>
              <span className="downloaded-name">{pdf.filename}</span>
              <span className="downloaded-meta">
                {formatBytes(pdf.size_bytes)} · {formatDateTime(pdf.modified_at)}
              </span>
            </div>
            <span className="downloaded-path">{pdf.path}</span>
            <PdfIndexBadge state={states[pdf.filename]} />
          </button>
        ))}

        {loading ? (
          <div className="downloaded-item muted">
            <span>Loading PDFs...</span>
          </div>
        ) : null}
      </div>
    </section>
  );
}

function PdfIndexBadge({ state }) {
  if (!state) return <span className="index-badge neutral">Not indexed</span>;
  if (state.loading) return <span className="index-badge neutral">Indexing...</span>;
  if (state.error) return <span className="index-badge error">Index failed</span>;
  if (state.chunksIndexed !== null) return <span className="index-badge success">{state.chunksIndexed} chunks</span>;
  return null;
}

function PaperDetail({ downloadState, indexState, onDownload, paper }) {
  return (
    <article className="paper-detail">
      <div className="detail-topline">
        <span>{paper.paper_id}</span>
        <span>{paper.published}</span>
      </div>
      <h2>{paper.title}</h2>
      <p className="authors">{paper.authors?.join(", ") || "Unknown authors"}</p>

      <div className="action-row">
        {paper.arxivUrl ? (
          <a href={paper.arxivUrl} rel="noreferrer" target="_blank">
            <ExternalLink size={16} aria-hidden="true" />
            arXiv
          </a>
        ) : null}
        {paper.pdfUrl ? (
          <a href={paper.pdfUrl} rel="noreferrer" target="_blank">
            <FileText size={16} aria-hidden="true" />
            PDF
          </a>
        ) : null}
        {paper.source === "local" ? null : (
          <button disabled={!paper.pdfUrl || downloadState.loading} onClick={onDownload} type="button">
            <ArrowDownToLine size={16} aria-hidden="true" />
            {downloadState.loading ? "Saving" : "Download"}
          </button>
        )}
      </div>

      {downloadState.message ? <div className="success-box">{downloadState.message}</div> : null}
      {downloadState.error ? <div className="error-box">{downloadState.error}</div> : null}
      {indexState?.loading ? <div className="success-box">Indexing PDF for chat...</div> : null}
      {indexState?.chunksIndexed !== null ? (
        <div className="success-box">Ready for chat: {indexState.chunksIndexed} chunks indexed.</div>
      ) : null}
      {indexState?.error ? <div className="error-box">{indexState.error}</div> : null}
      {paper.pdfPath ? <div className="local-path-box">{paper.pdfPath}</div> : null}

      <section className="abstract-section">
        <h3>Abstract</h3>
        <p>{paper.abstract || "No abstract available."}</p>
      </section>
    </article>
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
