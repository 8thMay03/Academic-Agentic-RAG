import {
  CheckCircle2,
  CircleAlert,
  Eye,
  FileDown,
  FileSearch,
  ListChecks,
  PencilLine,
  RotateCcw,
  Search,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";

import { agentTraceDisplay } from "../utils/format.js";

export default function AgentActivity({ trace }) {
  return (
    <div className="agent-activity" aria-label="Agent activity">
      {trace.map((step, index) => {
        const display = agentTraceDisplay(step);
        const Icon = iconForStep(step);

        return (
          <div className={`agent-step agent-step-${display.tone}`} key={`${step.stage}-${index}`}>
            <span className="agent-step-icon" aria-hidden="true">
              <Icon size={14} />
            </span>
            <div className="agent-step-copy">
              <span className="agent-step-stage">{display.label}</span>
              {display.detail ? <span className="agent-step-detail">{display.detail}</span> : null}
              {step.tool_result ? (
                <details className="agent-tool-result">
                  <summary>Tool result</summary>
                  <pre>{formatToolResult(step.tool_result)}</pre>
                </details>
              ) : null}
            </div>
            {display.badge ? <span className="agent-step-badge">{display.badge}</span> : null}
          </div>
        );
      })}
    </div>
  );
}

function formatToolResult(toolResult) {
  return JSON.stringify(toolResult, null, 2);
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
