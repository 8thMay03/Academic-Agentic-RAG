export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "Không rõ";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDateTime(value) {
  if (!value) return "Không rõ";
  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatEvidenceQuality(quality) {
  if (quality === "high") return "Cao";
  if (quality === "medium") return "Trung bình";
  if (quality === "low") return "Thấp";
  if (quality === "web") return "Web";
  return "Không rõ";
}

export function evidenceQualityClass(quality) {
  if (quality === "high") return "quality-high";
  if (quality === "medium") return "quality-medium";
  if (quality === "low") return "quality-low";
  if (quality === "web") return "quality-web";
  return "quality-unknown";
}

export function citationBadges(citation) {
  const badges = [citationSourceBadge(citation)];
  const retrievalSources = new Set(citation.retrieval_sources ?? []);
  if (retrievalSources.has("vector") || retrievalSources.has("keyword")) {
    badges.push({ label: "Indexed", className: "badge-indexed" });
  }
  if (citation.ingestion_status === "downloaded") {
    badges.push({ label: "Downloaded", className: "badge-indexed" });
  }
  if (citation.trust_level === "high") {
    badges.push({ label: "High trust", className: "badge-verified" });
  }
  if (citation.evidence_quality === "low") {
    badges.push({ label: "Low confidence", className: "badge-low" });
  } else if (["high", "medium", "web"].includes(citation.evidence_quality)) {
    badges.push({ label: "Verified", className: "badge-verified" });
  }
  return badges;
}

function citationSourceBadge(citation) {
  const retrievalSources = new Set(citation.retrieval_sources ?? []);
  const url = citation.url ?? citation.source_url ?? citation.pdf_url ?? "";
  if (citation.source_type === "arxiv" || url.includes("arxiv.org") || /^\d{4}\./.test(citation.paper_id ?? "")) {
    return { label: "arXiv", className: "badge-arxiv" };
  }
  if (citation.source_type === "web_pdf") {
    return { label: "Web PDF", className: "badge-web" };
  }
  if (citation.source_type === "web_page" || url || retrievalSources.has("web")) {
    return { label: "Web", className: "badge-web" };
  }
  if (citation.source_type === "local_pdf") {
    return { label: "Local PDF", className: "badge-local" };
  }
  return { label: "Local", className: "badge-local" };
}

export function agentTraceDisplay(step) {
  const stage = step.stage ?? "unknown";
  return {
    label: agentStageLabel(stage),
    detail: agentStepDetail(step),
    badge: agentStepBadge(step),
    tone: agentStepTone(step),
  };
}

function agentStageLabel(stage) {
  const labels = {
    classify_intent: "Intent",
    local_retrieve: "Local retrieval",
    quality_gate: "Quality gate",
    plan: "Planning",
    plan_recovery: "Recovery plan",
    execute_tool: "Tool call",
    observe: "Observation",
    draft_answer: "Drafting",
    generate_answer: "Generation",
    verify_answer: "Verification",
  };
  return labels[stage] ?? stage;
}

function agentStepDetail(step) {
  const details = [];
  if (step.tool_name) details.push(formatToolName(step.tool_name));
  if (typeof step.chunk_count === "number") details.push(`${step.chunk_count} chunks`);
  if (typeof step.step_count === "number") details.push(`${step.step_count} steps`);
  if (typeof step.paper_count === "number") details.push(`${step.paper_count} papers`);
  if (typeof step.artifact_count === "number") details.push(`${step.artifact_count} artifacts`);
  if (typeof step.chunks_indexed === "number") details.push(`${step.chunks_indexed} indexed`);
  if (typeof step.snippets_ingested === "number") details.push(`${step.snippets_ingested} snippets indexed`);
  if (step.source_type) details.push(formatSourceType(step.source_type));
  if (step.ingestion_status) details.push(step.ingestion_status);
  if (step.trust_level) details.push(`${step.trust_level} trust`);
  if (step.source_url) details.push(shortUrl(step.source_url));
  if (typeof step.issue_count === "number") details.push(`${step.issue_count} issues`);
  if (typeof step.unsupported_claim_count === "number") details.push(`${step.unsupported_claim_count} unsupported claims`);
  if (typeof step.context_chars === "number") details.push(`${step.context_chars} context chars`);
  if (typeof step.answer_chars === "number") details.push(`${step.answer_chars} answer chars`);
  if (typeof step.sufficient === "boolean") details.push(step.sufficient ? "sufficient context" : "needs more context");
  if (step.reason) details.push(step.reason);
  if (step.status) details.push(step.status);
  if (step.suggested_action) details.push(formatSuggestedAction(step.suggested_action));
  return details.join(" · ");
}

function agentStepBadge(step) {
  if (step.source_type) return formatSourceType(step.source_type);
  if (step.tool_name) return formatToolName(step.tool_name);
  if (step.suggested_action) return formatSuggestedAction(step.suggested_action);
  if (typeof step.sufficient === "boolean") return step.sufficient ? "Sufficient" : "Needs context";
  if (step.status) return step.status;
  return "";
}

function agentStepTone(step) {
  if (step.success === false) return "error";
  if (step.stage === "verify_answer" && step.suggested_action && step.suggested_action !== "finalize") return "warn";
  if (step.stage === "quality_gate" && step.sufficient === false) return "warn";
  if (step.stage === "verify_answer" || step.sufficient === true || step.success === true) return "success";
  return "neutral";
}

function formatToolName(toolName) {
  const labels = {
    local_retrieve: "Local",
    web_search: "Web",
    web_snippet_ingest: "Index snippets",
    arxiv_search: "arXiv",
    pdf_download: "Download PDF",
    pdf_index: "Index PDF",
  };
  return labels[toolName] ?? toolName;
}

function formatSuggestedAction(action) {
  const labels = {
    finalize: "Finalize",
    retrieve_more: "Retrieve more",
    revise_answer: "Revise",
    answer_unknown: "Answer unknown",
  };
  return labels[action] ?? action;
}

function formatSourceType(sourceType) {
  const labels = {
    arxiv: "arXiv",
    local_pdf: "Local PDF",
    snippet: "Snippet",
    web_page: "Web page",
    web_pdf: "Web PDF",
  };
  return labels[sourceType] ?? sourceType;
}

function shortUrl(value) {
  try {
    const url = new URL(value);
    return `${url.hostname}${url.pathname}`.replace(/\/$/, "");
  } catch {
    return value;
  }
}
