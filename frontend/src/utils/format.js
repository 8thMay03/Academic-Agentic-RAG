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

export function claimCitationHighlights(trace = [], citations = []) {
  const verificationStep = [...trace].reverse().find((step) => step.stage === "verify_answer" && step.claim_citation_map?.length);
  if (!verificationStep) return [];

  const citationsByChunkId = new Map(
    citations
      .filter((citation) => citation?.chunk_id)
      .map((citation, index) => [citation.chunk_id, { citation, displayIndex: index + 1 }]),
  );

  return verificationStep.claim_citation_map
    .filter((item) => item?.claim)
    .map((item) => {
      const sources = (item.supporting_chunk_ids ?? [])
        .map((chunkId) => citationsByChunkId.get(chunkId))
        .filter(Boolean);
      return {
        claim: item.claim,
        status: item.status ?? "unknown",
        reason: item.reason ?? "",
        sources,
      };
    });
}

export function formatClaimStatus(status) {
  const labels = {
    supported: "Supported",
    contradicted: "Contradicted",
    insufficient: "Insufficient",
    unknown: "Unknown",
  };
  return labels[status] ?? status;
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

export function formatAgentStopReason(stopReason) {
  const labels = {
    answered_with_sufficient_context: "Answered with sufficient context",
    answered_after_recovery: "Answered after recovery",
    no_relevant_context: "No relevant context",
    web_search_disabled: "Web search disabled",
    planner_no_valid_steps: "Planner had no valid steps",
    step_limit_reached: "Step limit reached",
    tool_limit_reached: "Tool limit reached",
    tool_execution_failed: "Tool execution failed",
    verification_failed: "Verification failed",
    verification_failed_answer_unknown: "Verification failed, answered unknown",
  };
  return labels[stopReason] ?? stopReason;
}

export function formatRunUsageSummary(usage) {
  if (!usage) return [];
  const details = [];
  if (typeof usage.latency_ms === "number" && usage.latency_ms > 0) details.push(`${Math.round(usage.latency_ms)} ms`);
  if (typeof usage.total_tokens === "number" && usage.total_tokens > 0) details.push(`${usage.total_tokens} tokens`);
  if (typeof usage.embedding_tokens === "number" && usage.embedding_tokens > 0) details.push(`${usage.embedding_tokens} embedding tokens`);
  if (typeof usage.estimated_cost_usd === "number" && usage.estimated_cost_usd > 0) {
    details.push(`$${usage.estimated_cost_usd.toFixed(4)}`);
  }
  if (typeof usage.tool_call_count === "number" && usage.tool_call_count > 0) {
    details.push(`${usage.tool_call_count} tool calls`);
  }
  if (usage.models?.length) details.push(usage.models.join(", "));
  return details;
}

function agentStageLabel(stage) {
  const labels = {
    classify_intent: "Intent",
    query_planning: "Query planning",
    query_decomposition: "Decomposition",
    retrieval_planning: "Retrieval plan",
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
  if (step.planner_source) details.push(`planner: ${step.planner_source}`);
  if (step.selected_tools?.length) details.push(`tools: ${step.selected_tools.map(formatToolName).join(", ")}`);
  if (typeof step.chunk_count === "number") details.push(`${step.chunk_count} chunks`);
  if (typeof step.suspicious_context_count === "number") details.push(`${step.suspicious_context_count} suspicious chunks`);
  if (typeof step.query_count === "number") details.push(`${step.query_count} queries`);
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
  if (typeof step.supported_claim_count === "number") details.push(`${step.supported_claim_count} supported claims`);
  if (typeof step.contradicted_claim_count === "number" && step.contradicted_claim_count > 0) {
    details.push(`${step.contradicted_claim_count} contradicted claims`);
  }
  if (typeof step.insufficient_claim_count === "number" && step.insufficient_claim_count > 0) {
    details.push(`${step.insufficient_claim_count} insufficient claims`);
  }
  if (typeof step.context_chars === "number") details.push(`${step.context_chars} context chars`);
  if (typeof step.answer_chars === "number") details.push(`${step.answer_chars} answer chars`);
  if (typeof step.latency_ms === "number") details.push(`${Math.round(step.latency_ms)} ms`);
  if (typeof step.input_tokens === "number" || typeof step.output_tokens === "number") {
    details.push(`${step.input_tokens ?? 0}/${step.output_tokens ?? 0} tokens`);
  }
  if (typeof step.estimated_cost_usd === "number" && step.estimated_cost_usd > 0) {
    details.push(`$${step.estimated_cost_usd.toFixed(4)}`);
  }
  if (typeof step.embedding_tokens === "number") details.push(`${step.embedding_tokens} embedding tokens`);
  if (typeof step.embedding_estimated_cost_usd === "number" && step.embedding_estimated_cost_usd > 0) {
    details.push(`embed $${step.embedding_estimated_cost_usd.toFixed(4)}`);
  }
  if (typeof step.sufficient === "boolean") details.push(step.sufficient ? "sufficient context" : "needs more context");
  if (step.stop_condition) details.push(`stop: ${step.stop_condition}`);
  if (step.reason) details.push(step.reason);
  if (step.status) details.push(step.status);
  if (step.suggested_action) details.push(formatSuggestedAction(step.suggested_action));
  return details.join(" · ");
}

function agentStepBadge(step) {
  if (typeof step.suspicious_context_count === "number" && step.suspicious_context_count > 0) return "Suspicious";
  if (step.source_type) return formatSourceType(step.source_type);
  if (step.tool_name) return formatToolName(step.tool_name);
  if (step.suggested_action) return formatSuggestedAction(step.suggested_action);
  if (typeof step.sufficient === "boolean") return step.sufficient ? "Sufficient" : "Needs context";
  if (step.status) return step.status;
  return "";
}

function agentStepTone(step) {
  if (step.success === false) return "error";
  if (typeof step.suspicious_context_count === "number" && step.suspicious_context_count > 0) return "warn";
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
