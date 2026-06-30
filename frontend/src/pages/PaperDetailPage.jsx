import { ArrowLeft, FileText, MessageSquare } from "lucide-react";
import { getPdfFileUrl } from "../api.js";
import { formatBytes, formatDateTime } from "../utils/format.js";
import { displayTitleFromFilename } from "../utils/paper.js";

export default function PaperDetailPage({ paper, onBack, onChatWithPaper }) {
  const pdfUrl = getPdfFileUrl(paper.filename);
  const title = displayTitleFromFilename(paper.filename);

  return (
    <div className="paper-detail">
      <header className="paper-detail-header">
        <button className="btn-icon" onClick={onBack} type="button" aria-label="Quay lại">
          <ArrowLeft size={18} aria-hidden="true" />
        </button>
        <div className="paper-detail-heading">
          <p className="brand-eyebrow">Chi tiết paper</p>
          <h1>{title}</h1>
          <p className="paper-detail-meta">
            {paper.filename} · {formatBytes(paper.size_bytes)} · Cập nhật {formatDateTime(paper.modified_at)}
          </p>
        </div>
        <button className="btn-primary" onClick={() => onChatWithPaper(paper)} type="button">
          <MessageSquare size={16} aria-hidden="true" />
          Hỏi AI về paper này
        </button>
      </header>

      <div className="paper-detail-body">
        <aside className="paper-detail-sidebar">
          <div className="detail-card">
            <FileText size={20} aria-hidden="true" />
            <h2>Thông tin file</h2>
            <dl>
              <div>
                <dt>Tên file</dt>
                <dd>{paper.filename}</dd>
              </div>
              <div>
                <dt>Kích thước</dt>
                <dd>{formatBytes(paper.size_bytes)}</dd>
              </div>
              <div>
                <dt>Cập nhật</dt>
                <dd>{formatDateTime(paper.modified_at)}</dd>
              </div>
            </dl>
          </div>
        </aside>
        <section className="paper-viewer" aria-label="Xem PDF">
          <iframe src={`${pdfUrl}#view=FitH`} title={title} />
        </section>
      </div>
    </div>
  );
}
