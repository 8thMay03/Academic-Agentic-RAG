import { Bot, User } from "lucide-react";
import { useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
          {content ? <MessageContent content={content} isUser={isUser} /> : null}
          {!content && message.streaming ? <span className="typing-dots">Đang suy nghĩ</span> : null}
          {message.streaming && content ? <span className="typing-cursor" aria-hidden="true" /> : null}
        </div>
        {!isUser && message.trace?.length ? <AgentActivity trace={message.trace} /> : null}
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
      <ReactMarkdown key={`markdown-${index}`} remarkPlugins={[remarkGfm]}>
        {segment.content}
      </ReactMarkdown>
    );
  });
}

function normalizeAssistantMarkdown(content) {
  let text = content.replace(/\r\n/g, "\n").trim();
  text = normalizeFlattenedTables(text);
  text = text.replace(/\s+---\s+/g, "\n\n---\n\n");
  text = text.replace(/\s+(#{1,6}\s+)/g, "\n\n$1");
  text = text.replace(/\s+(-\s+(?=(\*\*)?[A-ZÀ-Ỹ0-9]))/g, "\n$1");
  text = text.replace(/\s+(\d+\.\s+(?=[A-ZÀ-Ỹ0-9]))/g, "\n$1");
  text = text.replace(/\s*(\|[^|\n]+(?:\|[^|\n]+)+\|)\s*/g, "\n$1\n");
  text = repairBrokenTableLines(text);
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function normalizeFlattenedTables(content) {
  return content.replace(
    /\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|\s*\|\s*[-:\s]+\|\s*[-:\s]+\|\s*((?:\|?\s*\d+\.\s+[^|]+?\s*\|\s*[^|]+?\s*\|\s*)+)/g,
    (_, firstHeader, secondHeader, rowText) => {
      const rows = [...rowText.matchAll(/\|?\s*(\d+\.\s+[^|]+?)\s*\|\s*([^|]+?)\s*\|/g)];
      if (!rows.length) return _;

      return [
        "",
        `| ${firstHeader.trim()} | ${secondHeader.trim()} |`,
        "| --- | --- |",
        ...rows.map((row) => `| ${cleanTableCell(row[1])} | ${cleanTableCell(row[2])} |`),
        "",
      ].join("\n");
    },
  );
}

function repairBrokenTableLines(content) {
  const lines = content.split("\n");
  const repairedLines = [];

  for (let index = 0; index < lines.length; index += 1) {
    const header = parsePipeCells(lines[index]);
    const separatorIndex = nextNonEmptyLineIndex(lines, index + 1);
    if (!header || header.length !== 2 || separatorIndex === -1 || !isTwoColumnSeparator(lines[separatorIndex])) {
      repairedLines.push(lines[index]);
      continue;
    }

    const rows = [];
    let rowIndex = nextNonEmptyLineIndex(lines, separatorIndex + 1);
    while (rowIndex !== -1) {
      const row = parseNumberedTwoColumnRow(lines[rowIndex]);
      if (!row) break;
      rows.push(row);
      rowIndex = nextNonEmptyLineIndex(lines, rowIndex + 1);
    }

    if (!rows.length) {
      repairedLines.push(lines[index]);
      continue;
    }

    repairedLines.push(`| ${header[0]} | ${header[1]} |`);
    repairedLines.push("| --- | --- |");
    rows.forEach((row) => repairedLines.push(`| ${row[0]} | ${row[1]} |`));
    index = rowIndex === -1 ? lines.length : rowIndex - 1;
  }

  return repairedLines.join("\n");
}

function parsePipeCells(line) {
  const trimmedLine = line.trim();
  if (!trimmedLine.startsWith("|") || !trimmedLine.endsWith("|")) return null;
  const cells = trimmedLine
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
  return cells.length >= 2 ? cells : null;
}

function isTwoColumnSeparator(line) {
  const cells = parsePipeCells(line);
  return cells?.length === 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function parseNumberedTwoColumnRow(line) {
  const cells = line
    .trim()
    .replace(/^\|/, "")
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
  if (cells.length < 2 || !/^\d+\.\s+/.test(cells[0])) return null;
  return [cells[0], cells.slice(1).join(" | ")];
}

function nextNonEmptyLineIndex(lines, startIndex) {
  for (let index = startIndex; index < lines.length; index += 1) {
    const trimmedLine = lines[index].trim();
    if (trimmedLine && trimmedLine !== "|" && trimmedLine !== "||") return index;
  }
  return -1;
}

function cleanTableCell(cell) {
  return cell.replace(/\s*\|\s*$/g, "").trim();
}

function buildAssistantSegments(content) {
  const text = normalizeAssistantMarkdown(content);
  const lines = text.split("\n");
  const segments = [];
  let markdownBuffer = [];

  for (let index = 0; index < lines.length; index += 1) {
    const table = readTableAt(lines, index);
    if (!table) {
      markdownBuffer.push(lines[index]);
      continue;
    }

    pushMarkdownSegment(segments, markdownBuffer);
    markdownBuffer = [];
    segments.push({ type: "table", headers: table.headers, rows: table.rows });
    index = table.endIndex;
  }

  pushMarkdownSegment(segments, markdownBuffer);
  return segments.length ? segments : [{ type: "markdown", content: text }];
}

function readTableAt(lines, startIndex) {
  const headers = parsePipeCells(lines[startIndex]);
  if (!headers || headers.length < 2) return null;

  const separatorIndex = nextNonEmptyLineIndex(lines, startIndex + 1);
  if (separatorIndex === -1 || !isTwoColumnSeparator(lines[separatorIndex])) return null;

  const rows = [];
  let endIndex = separatorIndex;
  let rowIndex = separatorIndex + 1;

  while (rowIndex < lines.length) {
    const line = lines[rowIndex];
    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine === "|" || trimmedLine === "||") {
      rowIndex += 1;
      continue;
    }

    const pipeCells = parsePipeCells(line);
    const numberedRow = parseNumberedTwoColumnRow(line);
    const row = pipeCells?.length >= 2 ? pipeCells : numberedRow;
    if (!row) break;

    rows.push([cleanTableCell(row[0]), cleanTableCell(row.slice(1).join(" | "))]);
    endIndex = rowIndex;
    rowIndex += 1;
  }

  if (!rows.length) return null;
  return { headers: headers.slice(0, 2).map(cleanTableCell), rows, endIndex };
}

function pushMarkdownSegment(segments, markdownBuffer) {
  const content = markdownBuffer.join("\n").trim();
  if (content) segments.push({ type: "markdown", content });
}

function AssistantTable({ headers, rows }) {
  return (
    <table>
      <thead>
        <tr>
          {headers.map((header) => (
            <th key={header}>{header}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, rowIndex) => (
          <tr key={`${row[0]}-${rowIndex}`}>
            {headers.map((header, cellIndex) => (
              <td key={`${header}-${cellIndex}`}>{row[cellIndex] ?? ""}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
