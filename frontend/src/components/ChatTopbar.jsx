import { Database, History, Menu, Sparkles, Trash2 } from "lucide-react";

export default function ChatTopbar({
  activeChat,
  chatLoading,
  onClearHistory,
  onOpenRuns,
  onOpenSidebar,
  onOpenSources,
}) {
  return (
    <header className="chat-topbar">
      <button className="btn-icon" onClick={onOpenSidebar} type="button" aria-label="Mở sidebar">
        <Menu size={18} aria-hidden="true" />
      </button>
      <div className="chat-topbar-title">
        <Sparkles size={16} aria-hidden="true" />
        <span>{activeChat?.title ?? "Research Assistant"}</span>
      </div>
      <div className="chat-topbar-actions">
        {activeChat ? (
          <>
            <button className="btn-ghost btn-sm" disabled={!activeChat.messages.length || chatLoading} onClick={onClearHistory} type="button">
              <Trash2 size={14} aria-hidden="true" />
              Xóa tin nhắn
            </button>
            <button className="btn-ghost btn-sm" onClick={onOpenRuns} type="button">
              <History size={14} aria-hidden="true" />
              Agent runs
            </button>
            <button className="btn-ghost btn-sm" onClick={onOpenSources} type="button">
              <Database size={14} aria-hidden="true" />
              Nguồn tùy chọn ({activeChat.sources.length})
            </button>
          </>
        ) : null}
      </div>
    </header>
  );
}
