import {
  CheckCircle2,
  CircleAlert,
  Circle,
  Eye,
  FileDown,
  FileSearch,
  ListChecks,
  PencilLine,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  LoaderCircle,
  Wrench,
} from "lucide-react";

import { agentTraceDisplay } from "../utils/format.js";

export default function AgentActivity({ trace, active = false }) {
  const phases = buildPhases(trace, active);
  const headline = activityHeadline(trace, active);

  return (
    <div className="agent-activity" aria-label="Agent activity">
      <div className="agent-activity-head">
        <span className="agent-live-dot" aria-hidden="true" />
        <span>{headline}</span>
      </div>

      <div className="agent-todo-card">
        <div className="agent-todo-title">
          <ListChecks size={14} aria-hidden="true" />
          <span>To-dos {phases.filter((phase) => phase.status === "done").length}/{phases.length}</span>
        </div>
        <ol className="agent-todo-list">
          {phases.map((phase) => {
            const PhaseIcon = phase.status === "done" ? CheckCircle2 : phase.status === "active" ? LoaderCircle : Circle;
            return (
              <li className={`agent-todo-item agent-todo-${phase.status}`} key={phase.id}>
                <PhaseIcon size={14} aria-hidden="true" />
                <span>{phase.label}</span>
              </li>
            );
          })}
        </ol>
      </div>

      <div className="agent-timeline">
        {trace.map((step, index) => {
        const display = agentTraceDisplay(step);
        const Icon = iconForStep(step);
        const eventText = eventSummary(step, display);

        return (
          <div className={`agent-step agent-step-${display.tone}`} key={`${step.stage}-${index}`}>
            <span className="agent-step-icon" aria-hidden="true">
              <Icon size={14} />
            </span>
            <div className="agent-step-copy">
              <span className="agent-step-stage">{eventText}</span>
              {display.detail ? <span className="agent-step-detail">{display.detail}</span> : null}
              {step.queries?.length ? (
                <details className="agent-tool-result">
                  <summary>Queries</summary>
                  <ol className="agent-query-list">
                    {step.queries.map((query, queryIndex) => (
                      <li key={`${query}-${queryIndex}`}>{query}</li>
                    ))}
                  </ol>
                </details>
              ) : null}
              {step.tool_result ? (
                <details className="agent-tool-result">
                  <summary>Tool result</summary>
                  <pre>{formatToolResult(step.tool_result)}</pre>
                </details>
              ) : null}
              {hasStepData(step) ? (
                <details className="agent-tool-result">
                  <summary>Step data</summary>
                  <pre>{formatStepData(step)}</pre>
                </details>
              ) : null}
            </div>
            {display.badge ? <span className="agent-step-badge">{display.badge}</span> : null}
          </div>
        );
        })}
      </div>
    </div>
  );
}

function formatToolResult(toolResult) {
  return JSON.stringify(toolResult, null, 2);
}

function formatStepData(step) {
  return JSON.stringify(stepData(step), null, 2);
}

function hasStepData(step) {
  return Object.keys(stepData(step)).length > 0;
}

function stepData(step) {
  const hiddenKeys = new Set(["stage", "tool_result"]);
  return Object.fromEntries(
    Object.entries(step).filter(([, value]) => value !== null && value !== undefined).filter(([key]) => !hiddenKeys.has(key)),
  );
}

function buildPhases(trace, active) {
  const phaseDefinitions = [
    { id: "plan", label: "Plan queries", stages: ["classify_intent", "query_planning", "query_decomposition", "retrieval_planning", "plan"] },
    { id: "explore", label: "Explore sources", stages: ["local_retrieve", "execute_tool"] },
    { id: "read", label: "Read evidence", stages: ["observe"] },
    { id: "draft", label: "Draft answer", stages: ["draft_answer", "generate_answer"] },
    { id: "verify", label: "Verify grounding", stages: ["quality_gate", "verify_answer"] },
  ];
  const seenStages = new Set(trace.map((step) => step.stage));
  const latestPhaseId = phaseForStage(trace.at(-1)?.stage);

  return phaseDefinitions.map((phase) => {
    const hasStarted = phase.stages.some((stage) => seenStages.has(stage));
    const status = active && phase.id === latestPhaseId ? "active" : hasStarted ? "done" : "pending";
    return { ...phase, status };
  });
}

function phaseForStage(stage) {
  if (["classify_intent", "query_planning", "query_decomposition", "retrieval_planning", "plan", "plan_recovery"].includes(stage)) {
    return "plan";
  }
  if (["local_retrieve", "execute_tool"].includes(stage)) return "explore";
  if (stage === "observe") return "read";
  if (["draft_answer", "generate_answer"].includes(stage)) return "draft";
  if (["quality_gate", "verify_answer"].includes(stage)) return "verify";
  return "";
}

function activityHeadline(trace, active) {
  const latest = trace.at(-1);
  if (!latest) return "Preparing agent run";
  if (!active) return latest.stage === "verify_answer" ? "Completed agent run" : "Agent run paused";

  const labels = {
    plan: "Planning the search",
    explore: "Exploring sources",
    read: "Reading evidence",
    draft: "Drafting the answer",
    verify: "Checking groundedness",
  };
  return labels[phaseForStage(latest.stage)] ?? "Working through the task";
}

function eventSummary(step, display) {
  if (step.stage === "query_decomposition") return `Planned ${step.query_count ?? step.queries?.length ?? 0} search queries`;
  if (step.stage === "local_retrieve") return `Explored local index${typeof step.chunk_count === "number" ? ` · ${step.chunk_count} chunks` : ""}`;
  if (step.stage === "execute_tool") return `Ran ${display.label.toLowerCase()}${step.tool_name ? ` · ${formatToolNameForEvent(step.tool_name)}` : ""}`;
  if (step.stage === "observe") return `Observed ${step.tool_name ? formatToolNameForEvent(step.tool_name) : "tool result"}`;
  if (step.stage === "quality_gate") return step.sufficient ? "Confirmed enough evidence" : "Found evidence gap";
  if (step.stage === "verify_answer") return step.suggested_action === "finalize" ? "Completed verification" : "Flagged answer for revision";
  if (step.stage === "draft_answer") return "Prepared answer context";
  if (step.stage === "generate_answer") return "Generated answer draft";
  if (step.stage === "plan_recovery") return "Planned recovery search";
  return `Started ${display.label.toLowerCase()}`;
}

function formatToolNameForEvent(toolName) {
  const labels = {
    local_retrieve: "local retrieval",
    web_search: "web search",
    web_snippet_ingest: "snippet indexing",
    arxiv_search: "arXiv search",
    pdf_download: "PDF download",
    pdf_index: "PDF indexing",
  };
  return labels[toolName] ?? toolName;
}

function iconForStep(step) {
  const toolIcons = {
    arxiv_search: FileSearch,
    local_retrieve: Search,
    pdf_download: FileDown,
    pdf_index: FileSearch,
    web_search: Search,
    web_snippet_ingest: FileSearch,
  };
  if (step.stage === "execute_tool" && step.tool_name) return toolIcons[step.tool_name] ?? Wrench;

  const stageIcons = {
    draft_answer: PencilLine,
    generate_answer: Sparkles,
    classify_intent: ListChecks,
    local_retrieve: Search,
    observe: Eye,
    plan: ListChecks,
    plan_recovery: RotateCcw,
    quality_gate: ShieldCheck,
    verify_answer: step.suggested_action === "finalize" ? CheckCircle2 : CircleAlert,
  };
  return stageIcons[step.stage] ?? Wrench;
}
