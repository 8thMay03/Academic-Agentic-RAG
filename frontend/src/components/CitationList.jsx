import { citationBadges, evidenceQualityClass } from "../utils/format.js";

export default function CitationList({ citations, onOpenCitation }) {
  if (!citations?.length) return null;

  return (
    <div className="citation-list">
      {citations.map((citation, index) => {
        const isWebCitation = Boolean(citation.url);
        const badges = citationBadges(citation);

        return (
          <button
            aria-label={isWebCitation ? `Mở nguồn web: ${citation.title || citation.url}` : undefined}
            className={`citation-pill ${isWebCitation ? "citation-pill-web" : ""} ${evidenceQualityClass(citation.evidence_quality)}`}
            key={citation.chunk_id ?? `${citation.paper_id}-${citation.page_number}`}
            onClick={() => onOpenCitation(citation)}
            title={isWebCitation ? citation.url : undefined}
            type="button"
          >
            <span className="citation-index">[{index + 1}]</span>
            {citation.title || citation.paper_id}
            {citation.page_number ? ` · tr.${citation.page_number}` : ""}
            <span className="citation-badges" aria-hidden="true">
              {badges.map((badge) => (
                <span className={`citation-badge ${badge.className}`} key={badge.label}>
                  {badge.label}
                </span>
              ))}
            </span>
          </button>
        );
      })}
    </div>
  );
}
