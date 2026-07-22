import { Bot, User } from "lucide-react";
import { useEffect, useRef } from "react";
import rehypeKatex from "rehype-katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import AgentActivity from "./AgentActivity.jsx";
import CitationList from "./CitationList.jsx";
import { buildAssistantSegments, normalizeMathDelimiters } from "../utils/assistantMarkdown.js";

const MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkMath];
const MARKDOWN_REHYPE_PLUGINS = [rehypeKatex];

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
        <h2>Nhập câu hỏi của bạn</h2>
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
          {content ? <MessageContent content={content} isUser={isUser} /> : null}
          {!content && message.streaming ? <span className="typing-dots">Đang suy nghĩ</span> : null}
          {message.streaming && content ? <span className="typing-cursor" aria-hidden="true" /> : null}
        </div>
        {!isUser && message.trace?.length ? <AgentActivity trace={message.trace} active={Boolean(message.streaming)} /> : null}
        <CitationList citations={message.citations} onOpenCitation={onOpenCitation} />
      </div>
    </article>
  );
}

function MessageContent({ content, isUser }) {
  if (isUser) return content;

  return buildAssistantSegments(content).map((segment, index) => {
    if (segment.type === "table") {
      return <AssistantTable key={`table-${index}`} headers={segment.headers} rows={segment.rows} />;
    }

    return (
      <ReactMarkdown key={`markdown-${index}`} remarkPlugins={MARKDOWN_REMARK_PLUGINS} rehypePlugins={MARKDOWN_REHYPE_PLUGINS}>
        {segment.content}
      </ReactMarkdown>
    );
  });
}

function AssistantTable({ headers, rows }) {
  return (
    <table>
      <thead>
        <tr>
          {headers.map((header) => (
            <th key={header}>
              <MarkdownCell content={header} />
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={`${row[0]}-${rowIndex}`}>
            {headers.map((header, cellIndex) => (
              <td key={`${header}-${cellIndex}`}>
                <MarkdownCell content={row[cellIndex] ?? ""} />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MarkdownCell({ content }) {
  return (
    <ReactMarkdown remarkPlugins={MARKDOWN_REMARK_PLUGINS} rehypePlugins={MARKDOWN_REHYPE_PLUGINS}>
      {normalizeMathDelimiters(content)}
    </ReactMarkdown>
  );
}
