import { useState } from "react";
import { ArrowLeft, Check, MessageSquare, PanelLeftClose, Pencil, Plus, Trash2, X } from "lucide-react";

import { formatDateTime } from "../utils/format.js";

export default function ChatSidebar({
  activeChat,
  onBackHome,
  onCollapse,
  onCreateChat,
  onDeleteThread,
  onRenameThread,
  onOpenThread,
  threads,
  threadsState,
}) {
  const [editingThreadId, setEditingThreadId] = useState("");
  const [draftTitle, setDraftTitle] = useState("");
  const [savingThreadId, setSavingThreadId] = useState("");

  function beginRename(thread) {
    setEditingThreadId(thread.chat_id);
    setDraftTitle(thread.title ?? "");
  }

  function cancelRename() {
    setEditingThreadId("");
    setDraftTitle("");
  }

  async function submitRename(event, thread) {
    event.preventDefault();
    const nextTitle = draftTitle.trim();
    if (!nextTitle) return;
    if (nextTitle === thread.title) {
      cancelRename();
      return;
    }

    setSavingThreadId(thread.chat_id);
    try {
      await onRenameThread(thread.chat_id, nextTitle);
      cancelRename();
    } catch {
      // The parent owns the visible error state; keep the input open so the title can be corrected.
    } finally {
      setSavingThreadId("");
    }
  }

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
        {threads.map((thread) => {
          const isEditing = editingThreadId === thread.chat_id;
          const isSaving = savingThreadId === thread.chat_id;

          return (
            <div
              className={`thread-item ${activeChat?.chat_id === thread.chat_id ? "active" : ""}`}
              key={thread.chat_id}
            >
              {isEditing ? (
                <form className="thread-rename-form" onSubmit={(event) => submitRename(event, thread)}>
                  <input
                    aria-label={`Đổi tên ${thread.title}`}
                    autoFocus
                    className="thread-rename-input"
                    disabled={isSaving}
                    onChange={(event) => setDraftTitle(event.target.value)}
                    onFocus={(event) => event.target.select()}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") cancelRename();
                    }}
                    value={draftTitle}
                  />
                  <button
                    aria-label="Lưu tên mới"
                    className="thread-action"
                    disabled={isSaving || !draftTitle.trim()}
                    type="submit"
                  >
                    <Check size={14} aria-hidden="true" />
                  </button>
                  <button
                    aria-label="Hủy đổi tên"
                    className="thread-action"
                    disabled={isSaving}
                    onClick={cancelRename}
                    type="button"
                  >
                    <X size={14} aria-hidden="true" />
                  </button>
                </form>
              ) : (
                <>
                  <button className="thread-button" onClick={() => onOpenThread(thread.chat_id)} type="button">
                    <MessageSquare size={15} aria-hidden="true" />
                    <span className="thread-title">{thread.title}</span>
                    <span className="thread-meta">{formatDateTime(thread.updated_at)}</span>
                  </button>
                  <button
                    aria-label={`Đổi tên ${thread.title}`}
                    className="thread-action"
                    onClick={() => beginRename(thread)}
                    type="button"
                  >
                    <Pencil size={14} aria-hidden="true" />
                  </button>
                  <button
                    aria-label={`Xóa ${thread.title}`}
                    className="thread-delete"
                    onClick={() => onDeleteThread(thread)}
                    type="button"
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
