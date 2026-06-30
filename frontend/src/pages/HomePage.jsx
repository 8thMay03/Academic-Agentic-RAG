import { FileText, MessageSquare, RefreshCw, Upload } from "lucide-react";
import { useRef } from "react";
import { formatBytes, formatDateTime } from "../utils/format.js";
import { displayTitleFromFilename } from "../utils/paper.js";

export default function HomePage({
  papers,
  loading,
  error,
  uploadState,
  onOpenPaper,
  onStartChat,
  onRefresh,
  onUpload,
}) {
  const fileInputRef = useRef(null);

  return (
    <div className="home">
      <header className="home-header">
        <div className="home-brand">
          <span className="brand-mark" aria-hidden="true">
            <FileText size={18} />
          </span>
          <div>
            <p className="brand-eyebrow">Thư viện nghiên cứu</p>
            <h1>Paper đã lưu</h1>
          </div>
        </div>
        <div className="home-actions">
          <button className="btn-ghost" disabled={loading} onClick={onRefresh} type="button">
            <RefreshCw size={16} aria-hidden="true" />
            Làm mới
          </button>
          <button className="btn-ghost" disabled={uploadState.loading} onClick={() => fileInputRef.current?.click()} type="button">
            <Upload size={16} aria-hidden="true" />
            Tải PDF lên
          </button>
          <button className="btn-primary" onClick={onStartChat} type="button">
            <MessageSquare size={16} aria-hidden="true" />
            Chat với AI
          </button>
          <input
            accept="application/pdf,.pdf"
            className="sr-only"
            multiple
            onChange={(event) => {
              void onUpload(event.target.files);
              event.target.value = "";
            }}
            ref={fileInputRef}
            type="file"
          />
        </div>
      </header>

      {uploadState.message ? <div className="banner banner-success">{uploadState.message}</div> : null}
      {uploadState.error ? <div className="banner banner-error">{uploadState.error}</div> : null}
      {error ? <div className="banner banner-error">{error}</div> : null}

      <main className="home-main">
        {loading ? (
          <div className="state-panel">
            <div className="spinner" aria-hidden="true" />
            <p>Đang tải danh sách paper...</p>
          </div>
        ) : papers.length === 0 ? (
          <div className="state-panel">
            <FileText size={36} strokeWidth={1.25} aria-hidden="true" />
            <h2>Chưa có paper nào</h2>
            <p>Tải file PDF lên hoặc thêm vào thư mục <code>backend/data/pdfs</code> rồi làm mới.</p>
            <button className="btn-primary" disabled={uploadState.loading} onClick={() => fileInputRef.current?.click()} type="button">
              <Upload size={16} aria-hidden="true" />
              Tải PDF đầu tiên
            </button>
          </div>
        ) : (
          <ul className="paper-grid">
            {papers.map((paper) => (
              <li key={paper.path}>
                <button className="paper-card" onClick={() => onOpenPaper(paper)} type="button">
                  <span className="paper-card-icon" aria-hidden="true">
                    <FileText size={22} />
                  </span>
                  <span className="paper-card-body">
                    <span className="paper-card-title">{displayTitleFromFilename(paper.filename)}</span>
                    <span className="paper-card-filename">{paper.filename}</span>
                    <span className="paper-card-meta">
                      {formatBytes(paper.size_bytes)} · {formatDateTime(paper.modified_at)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}
