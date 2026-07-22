import { CircleAlert, HelpCircle, ShieldCheck } from "lucide-react";

import { claimCitationHighlights, formatClaimStatus } from "../utils/format.js";

export default function ClaimCitationMap({ citations, onOpenCitation, trace }) {
  const highlights = claimCitationHighlights(trace, citations);
  if (!highlights.length) return null;

  return (
    <section className="claim-map" aria-label="Claim citation map">
      <div className="claim-map-head">
        <ShieldCheck size={14} aria-hidden="true" />
        <span>Grounding by claim</span>
      </div>
      <ol className="claim-map-list">
        {highlights.map((item, index) => {
          const StatusIcon = iconForStatus(item.status);
          return (
            <li className={`claim-map-item claim-map-${item.status}`} key={`${item.claim}-${index}`}>
              <div className="claim-map-claim">
                <StatusIcon size={14} aria-hidden="true" />
                <span>{item.claim}</span>
              </div>
              <div className="claim-map-meta">
                <span className="claim-map-status">{formatClaimStatus(item.status)}</span>
                {item.reason ? <span>{item.reason}</span> : null}
              </div>
              {item.sources.length ? (
                <div className="claim-map-sources">
                  {item.sources.map(({ citation, displayIndex }) => (
                    <button
                      className="claim-map-source"
                      key={citation.chunk_id}
                      onClick={() => onOpenCitation(citation)}
                      title={citation.url ?? citation.title ?? citation.paper_id}
                      type="button"
                    >
                      [{displayIndex}] {citation.title || citation.paper_id || citation.chunk_id}
                      {citation.page_number ? ` · tr.${citation.page_number}` : ""}
                    </button>
                  ))}
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function iconForStatus(status) {
  if (status === "supported") return ShieldCheck;
  if (status === "contradicted") return CircleAlert;
  return HelpCircle;
}
