import { X } from "lucide-react";

import { formatDateTime, formatRunUsageSummary } from "../utils/format.js";
import AgentActivity from "./AgentActivity.jsx";

export default function AgentRunPanel({ activeTab, findings, onClose, onSelectTab, runs, runState }) {
  const hasRuns = runs.length > 0;
  const hasFindings = findings.length > 0;

  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label="Agent run history">
      <section className="agent-run-panel">
        <header className="agent-run-panel-header">
          <div>
            <h2>Agent memory</h2>
            <p>{hasFindings ? `${findings.length} findings từ ${runs.length} runs.` : `${runs.length} lượt chạy đã lưu.`}</p>
          </div>
          <button className="btn-icon" onClick={onClose} type="button" aria-label="Đóng">
            <X size={18} aria-hidden="true" />
          </button>
        </header>

        <div className="agent-run-tabs" role="tablist" aria-label="Agent memory views">
          <button
            aria-selected={activeTab === "findings"}
            className={activeTab === "findings" ? "active" : ""}
            onClick={() => onSelectTab("findings")}
            role="tab"
            type="button"
          >
            Findings
          </button>
          <button
            aria-selected={activeTab === "runs"}
            className={activeTab === "runs" ? "active" : ""}
            onClick={() => onSelectTab("runs")}
            role="tab"
            type="button"
          >
            Runs
          </button>
        </div>

        {runState.error ? <div className="banner banner-error">{runState.error}</div> : null}
        {runState.loading ? (
          <div className="run-empty">
            <span className="spinner" aria-hidden="true" />
            Đang tải agent memory...
          </div>
        ) : null}
        {!runState.loading && activeTab === "findings" && !hasFindings ? (
          <div className="run-empty">Không có findings.</div>
        ) : null}
        {!runState.loading && activeTab === "findings" && hasFindings ? (
          <ol className="finding-list">
            {[...findings].reverse().map((finding) => (
              <li className="finding-item" key={finding.finding_id}>
                <div className="finding-head">
                  <span className={`finding-confidence confidence-${finding.confidence}`}>{finding.confidence}</span>
                  <span className="agent-run-time">{formatDateTime(finding.created_at)}</span>
                </div>
                <p className="finding-summary">{finding.summary}</p>
                <div className="agent-run-meta">
                  <span>{finding.source_ids?.length ?? 0} sources</span>
                  <span>{finding.citation_ids?.length ?? 0} citations</span>
                </div>
                <p className="finding-question">{finding.question}</p>
              </li>
            ))}
          </ol>
        ) : null}
        {!runState.loading && activeTab === "runs" && !hasRuns ? <div className="run-empty">Không có run history.</div> : null}
        {!runState.loading && activeTab === "runs" && hasRuns ? (
          <ol className="agent-run-list">
            {[...runs].reverse().map((run) => (
              <AgentRunItem key={run.run_id} run={run} />
            ))}
          </ol>
        ) : null}
      </section>
    </div>
  );
}

function AgentRunItem({ run }) {
  const usageDetails = formatRunUsageSummary(run.usage);
  return (
    <li className="agent-run-item">
      <div className="agent-run-head">
        <span className="agent-run-question">{run.question}</span>
        <span className="agent-run-time">{formatDateTime(run.created_at)}</span>
      </div>
      <p className="agent-run-answer">{run.answer}</p>
      <div className="agent-run-meta">
        <span>{run.citations?.length ?? 0} citations</span>
        <span>{run.trace?.length ?? 0} steps</span>
      </div>
      {usageDetails.length ? (
        <div className="agent-run-usage">
          {usageDetails.map((detail) => (
            <span key={detail}>{detail}</span>
          ))}
        </div>
      ) : null}
      {run.trace?.length ? <AgentActivity trace={run.trace} stopReason={run.stop_reason} /> : null}
    </li>
  );
}
