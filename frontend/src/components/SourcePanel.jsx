import { Trash2, X } from "lucide-react";

import { displayTitleFromFilename, paperIdFromFilename } from "../utils/paper.js";

export default function SourcePanel({
  activeChat,
  downloadedPdfs,
  onAddPdf,
  onClose,
  onRemoveSource,
  sourceIds,
  sourceState,
}) {
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
