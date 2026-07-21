import { X } from "lucide-react";

import { getPdfFileUrl } from "../api.js";

export default function PdfPreviewModal({ onClose, source }) {
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
