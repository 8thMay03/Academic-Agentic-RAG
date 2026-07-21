import { ArrowLeft, MessageSquare, PanelLeftClose, Plus, Trash2 } from "lucide-react";

import { formatDateTime } from "../utils/format.js";

export default function ChatSidebar({
  activeChat,
  onBackHome,
  onCollapse,
  onCreateChat,
  onDeleteThread,
  onOpenThread,
  threads,
  threadsState,
}) {
  return (
    <aside className="chat-sidebar" aria-label="Danh sách cuộc trò chuyện">
      <div className="sidebar-top">
        <button className="btn-icon" onClick={onBackHome} type="button" aria-label="Về trang chủ">
          <ArrowLeft size={18} aria-hidden="true" />
        </button>
        <button className="sidebar-new-chat" onClick={onCreateChat} type="button">
          <Plus size={16} aria-hidden="true" />
          Cuộc trò chuyện mới
        </button>
        <button className="btn-icon sidebar-collapse" onClick={onCollapse} type="button" aria-label="Thu gọn sidebar">
          <PanelLeftClose size={18} aria-hidden="true" />
        </button>
      </div>

      {threadsState.error ? <div className="sidebar-error">{threadsState.error}</div> : null}

      <nav className="thread-list">
        {threads.length === 0 && !threadsState.loading ? (
          <p className="thread-empty">Chưa có cuộc trò chuyện nào.</p>
        ) : null}
        {threads.map((thread) => (
          <div className={`thread-item ${activeChat?.chat_id === thread.chat_id ? "active" : ""}`} key={thread.chat_id}>
            <button className="thread-button" onClick={() => onOpenThread(thread.chat_id)} type="button">
              <MessageSquare size={15} aria-hidden="true" />
              <span className="thread-title">{thread.title}</span>
              <span className="thread-meta">{formatDateTime(thread.updated_at)}</span>
            </button>
            <button
              aria-label={`Xóa ${thread.title}`}
              className="thread-delete"
              onClick={() => onDeleteThread(thread)}
              type="button"
            >
              <Trash2 size={14} aria-hidden="true" />
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}
