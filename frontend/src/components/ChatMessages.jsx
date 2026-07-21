import { Bot, User } from "lucide-react";
import { useEffect, useRef } from "react";

import AgentActivity from "./AgentActivity.jsx";
import CitationList from "./CitationList.jsx";

export default function ChatMessages({ activeChat, canChat, chatState, onOpenCitation, sourceState }) {
  const logRef = useRef(null);

  useEffect(() => {
    const log = logRef.current;
    if (!log) return;
    log.scrollTop = log.scrollHeight;
  }, [activeChat.messages, chatState.loading]);

  if (activeChat.messages.length === 0) {
    return (
      <div className="chat-welcome chat-welcome-inline">
        <div className="welcome-icon">
          <Bot size={24} aria-hidden="true" />
        </div>
        <h2>Hỏi về paper của bạn</h2>
        <p>
          {canChat
            ? "AI sẽ truy xuất toàn bộ tài liệu local, rồi dùng web nếu local context chưa đủ."
            : "Tạo cuộc trò chuyện để bắt đầu hỏi AI."}
        </p>
        {sourceState.error ? <div className="banner banner-error">{sourceState.error}</div> : null}
        {sourceState.message ? <div className="banner banner-success">{sourceState.message}</div> : null}
      </div>
    );
  }

  return (
    <div className="chat-log" ref={logRef}>
      {activeChat.messages.map((message, index) => (
        <ChatMessage key={`${message.role}-${message.created_at}-${index}`} message={message} onOpenCitation={onOpenCitation} />
      ))}
      {chatState.error ? <div className="banner banner-error">{chatState.error}</div> : null}
    </div>
  );
}

function ChatMessage({ message, onOpenCitation }) {
  const isUser = message.role === "user";
  const content = message.content || (message.streaming ? "" : "");

  return (
    <article className={`chat-message ${isUser ? "user" : "assistant"}`}>
      <div className="message-avatar" aria-hidden="true">
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="message-body">
        <div className="message-role">{isUser ? "Bạn" : "AI"}</div>
        <div className="message-text">
          {content || (message.streaming ? <span className="typing-dots">Đang suy nghĩ</span> : null)}
          {message.streaming && content ? <span className="typing-cursor" aria-hidden="true" /> : null}
        </div>
        {!isUser && message.trace?.length ? <AgentActivity trace={message.trace} /> : null}
        <CitationList citations={message.citations} onOpenCitation={onOpenCitation} />
      </div>
    </article>
  );
}
