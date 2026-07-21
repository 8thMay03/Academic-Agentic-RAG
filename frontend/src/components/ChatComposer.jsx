import { Send } from "lucide-react";

export default function ChatComposer({
  agentOptions,
  canChat,
  chatLoading,
  onAgentOptionsChange,
  onQuestionChange,
  onSubmit,
  question,
}) {
  return (
    <div className="composer-shell">
      <div className="agent-options" aria-label="Agent controls">
        <label className="agent-toggle">
          <input
            checked={agentOptions.enableWebSearch}
            disabled={chatLoading}
            onChange={(event) => onAgentOptionsChange({ enableWebSearch: event.target.checked })}
            type="checkbox"
          />
          <span>Web</span>
        </label>
        <label className="agent-toggle">
          <input
            checked={agentOptions.enableResearchIngest}
            disabled={chatLoading}
            onChange={(event) => onAgentOptionsChange({ enableResearchIngest: event.target.checked })}
            type="checkbox"
          />
          <span>arXiv ingest</span>
        </label>
        <label className="agent-toggle">
          <input
            checked={agentOptions.autoDownloadPdfs}
            disabled={chatLoading || !agentOptions.enableResearchIngest}
            onChange={(event) => onAgentOptionsChange({ autoDownloadPdfs: event.target.checked })}
            type="checkbox"
          />
          <span>Auto PDF</span>
        </label>
        <label className="agent-number">
          <span>Sources</span>
          <input
            disabled={chatLoading}
            max="10"
            min="1"
            onChange={(event) => onAgentOptionsChange({ topK: Number(event.target.value) })}
            type="number"
            value={agentOptions.topK}
          />
        </label>
        <label className="agent-number">
          <span>Steps</span>
          <input
            disabled={chatLoading}
            max="10"
            min="1"
            onChange={(event) => onAgentOptionsChange({ maxAgentSteps: Number(event.target.value) })}
            type="number"
            value={agentOptions.maxAgentSteps}
          />
        </label>
      </div>

      <form className="chat-composer" onSubmit={onSubmit}>
        <div className="composer-box">
          <textarea
            disabled={!canChat || chatLoading}
            onChange={(event) => onQuestionChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder={canChat ? "Nhắn tin cho AI..." : "Tạo cuộc trò chuyện để bắt đầu"}
            rows={1}
            value={question}
          />
          <button
            aria-label="Gửi"
            className="composer-send"
            disabled={!canChat || !question.trim() || chatLoading}
            type="submit"
          >
            <Send size={18} aria-hidden="true" />
          </button>
        </div>
        <p className="composer-hint">Enter để gửi · Shift+Enter để xuống dòng</p>
      </form>
    </div>
  );
}
