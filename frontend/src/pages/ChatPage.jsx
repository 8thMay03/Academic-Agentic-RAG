import { useEffect, useMemo, useRef, useState } from "react";
import {
  addChatSource,
  clearChatHistory,
  createChatSession,
  deleteChatSession,
  getChatSession,
  indexDownloadedPdf,
  listAgentRuns,
  listChatThreads,
  listDownloadedPdfs,
  listResearchFindings,
  removeChatSource,
  streamChatWithPaper,
  updateChatSessionTitle,
} from "../api.js";
import AgentRunPanel from "../components/AgentRunPanel.jsx";
import ChatComposer from "../components/ChatComposer.jsx";
import ChatEmptyState from "../components/ChatEmptyState.jsx";
import ChatMessages from "../components/ChatMessages.jsx";
import ChatSidebar from "../components/ChatSidebar.jsx";
import ChatTopbar from "../components/ChatTopbar.jsx";
import ConfirmDialog from "../components/ConfirmDialog.jsx";
import PdfPreviewModal from "../components/PdfPreviewModal.jsx";
import SourcePanel from "../components/SourcePanel.jsx";
import { sourceFromPdf } from "../utils/paper.js";

const DEFAULT_CHAT_TITLES = new Set(["Cuộc trò chuyện mới", "New chat"]);

function chatTitleFromQuestion(value) {
  return value.trim().replace(/\s+/g, " ").slice(0, 160);
}

function shouldTitleFromFirstQuestion(chat) {
  return Boolean(chat) && DEFAULT_CHAT_TITLES.has(chat.title) && (chat.messages?.length ?? 0) === 0;
}

export default function ChatPage({ onBackHome, initialPaper }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [threads, setThreads] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [question, setQuestion] = useState("");
  const [threadsState, setThreadsState] = useState({ loading: false, error: "" });
  const [chatState, setChatState] = useState({ loading: false, error: "" });
  const [sourceState, setSourceState] = useState({ loading: false, error: "", message: "" });
  const [agentOptions, setAgentOptions] = useState({
    enableWebSearch: true,
    enableResearchIngest: true,
    autoDownloadPdfs: true,
    topK: 5,
    maxAgentSteps: 6,
  });
  const [sourcePanelOpen, setSourcePanelOpen] = useState(false);
  const [runPanelOpen, setRunPanelOpen] = useState(false);
  const [agentRuns, setAgentRuns] = useState([]);
  const [researchFindings, setResearchFindings] = useState([]);
  const [runPanelTab, setRunPanelTab] = useState("findings");
  const [runState, setRunState] = useState({ loading: false, error: "" });
  const [downloadedPdfs, setDownloadedPdfs] = useState([]);
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const [paperPreview, setPaperPreview] = useState(null);
  const initialPaperHandled = useRef(false);

  const sourceIds = useMemo(
    () => new Set((activeChat?.sources ?? []).map((source) => source.paper_id)),
    [activeChat],
  );

  useEffect(() => {
    void refreshThreads();
    void refreshPdfs();
  }, []);

  useEffect(() => {
    if (!initialPaper || initialPaperHandled.current) return;
    initialPaperHandled.current = true;
    void startChatWithPaper(initialPaper);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPaper]);

  async function refreshThreads() {
    setThreadsState({ loading: true, error: "" });
    try {
      const response = await listChatThreads();
      setThreads(response.chats ?? []);
      setThreadsState({ loading: false, error: "" });
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function refreshPdfs() {
    try {
      const pdfs = await listDownloadedPdfs();
      setDownloadedPdfs(pdfs ?? []);
    } catch {
      setDownloadedPdfs([]);
    }
  }

  async function createNewChat() {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    try {
      const session = await createChatSession("Cuộc trò chuyện mới");
      setActiveChat(session);
      setQuestion("");
      setSourcePanelOpen(false);
      await refreshThreads();
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function openChat(chatId) {
    setSourceState({ loading: false, error: "", message: "" });
    setChatState({ loading: false, error: "" });
    setRunPanelOpen(false);
    setAgentRuns([]);
    setResearchFindings([]);
    setRunPanelTab("findings");
    setRunState({ loading: false, error: "" });
    try {
      const session = await getChatSession(chatId);
      setActiveChat(session);
      setQuestion("");
      if (window.innerWidth < 900) setSidebarOpen(false);
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  async function deleteChat(chatId) {
    setThreadsState({ loading: true, error: "" });
    try {
      await deleteChatSession(chatId);
      setDeleteCandidate(null);
      if (activeChat?.chat_id === chatId) {
        setActiveChat(null);
        setQuestion("");
      }
      await refreshThreads();
    } catch (error) {
      setThreadsState({ loading: false, error: error.message });
    }
  }

  function setChatTitleLocally(chatId, title) {
    setActiveChat((chat) => (chat?.chat_id === chatId ? { ...chat, title } : chat));
    setThreads((chatThreads) =>
      chatThreads.map((thread) => (thread.chat_id === chatId ? { ...thread, title } : thread)),
    );
  }

  async function renameChat(chatId, title) {
    const nextTitle = chatTitleFromQuestion(title);
    if (!nextTitle) {
      throw new Error("Tên cuộc trò chuyện không được để trống.");
    }

    const previousTitle =
      threads.find((thread) => thread.chat_id === chatId)?.title ??
      (activeChat?.chat_id === chatId ? activeChat.title : "");

    setChatTitleLocally(chatId, nextTitle);
    try {
      const session = await updateChatSessionTitle(chatId, nextTitle);
      setActiveChat((chat) => (chat?.chat_id === chatId ? { ...chat, ...session, messages: chat.messages } : chat));
      setThreads((chatThreads) =>
        chatThreads.map((thread) => (thread.chat_id === chatId ? { ...thread, ...session } : thread)),
      );
      setThreadsState({ loading: false, error: "" });
      return session;
    } catch (error) {
      if (previousTitle) setChatTitleLocally(chatId, previousTitle);
      setThreadsState({ loading: false, error: error.message });
      throw error;
    }
  }

  async function ensureActiveChat() {
    if (activeChat) return activeChat;
    const session = await createChatSession("Cuộc trò chuyện mới");
    setActiveChat(session);
    await refreshThreads();
    return session;
  }

  async function startChatWithPaper(pdf) {
    setSourceState({ loading: true, error: "", message: `Đang index ${pdf.filename}...` });
    setChatState({ loading: false, error: "" });
    try {
      const chat = await ensureActiveChat();
      const indexResponse = await indexDownloadedPdf(pdf.filename);
      const session = await addChatSource(chat.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: "" });
      setSourcePanelOpen(false);
      if (window.innerWidth < 900) setSidebarOpen(false);
      await refreshThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function addPdfToChat(pdf) {
    setSourceState({ loading: true, error: "", message: `Đang index ${pdf.filename}...` });
    try {
      const chat = await ensureActiveChat();
      const indexResponse = await indexDownloadedPdf(pdf.filename);
      const session = await addChatSource(chat.chat_id, sourceFromPdf(pdf, indexResponse.paper_id));
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: `${pdf.filename} đã được thêm.` });
      await refreshThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function removeSource(source) {
    if (!activeChat) return;
    setSourceState({ loading: true, error: "", message: "" });
    try {
      const session = await removeChatSource(activeChat.chat_id, source.paper_id);
      setActiveChat(session);
      setSourceState({ loading: false, error: "", message: "Đã gỡ nguồn khỏi cuộc trò chuyện." });
      await refreshThreads();
    } catch (error) {
      setSourceState({ loading: false, error: error.message, message: "" });
    }
  }

  async function handleClearHistory() {
    if (!activeChat) return;
    setChatState({ loading: true, error: "" });
    try {
      await clearChatHistory(activeChat.chat_id);
      const session = await getChatSession(activeChat.chat_id);
      setActiveChat(session);
      setChatState({ loading: false, error: "" });
      await refreshThreads();
    } catch (error) {
      setChatState({ loading: false, error: error.message });
    }
  }

  async function openRunHistory() {
    if (!activeChat) return;
    setRunPanelOpen(true);
    setRunPanelTab("findings");
    setRunState({ loading: true, error: "" });
    try {
      const [runsResponse, findingsResponse] = await Promise.all([
        listAgentRuns(activeChat.chat_id),
        listResearchFindings(activeChat.chat_id),
      ]);
      setAgentRuns(runsResponse.runs ?? []);
      setResearchFindings(findingsResponse.findings ?? []);
      setRunState({ loading: false, error: "" });
    } catch (error) {
      setRunState({ loading: false, error: error.message });
    }
  }

  async function handleAsk(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || !activeChat || sourceState.loading) return;

    const optimisticUser = {
      role: "user",
      content: trimmedQuestion,
      citations: [],
      created_at: new Date().toISOString(),
    };
    const optimisticAssistant = {
      role: "assistant",
      content: "",
      citations: [],
      trace: [],
      stop_reason: null,
      created_at: new Date(Date.now() + 1).toISOString(),
      streaming: true,
    };
    const chatId = activeChat.chat_id;
    const shouldRenameChat = shouldTitleFromFirstQuestion(activeChat);
    const firstQuestionTitle = shouldRenameChat ? chatTitleFromQuestion(trimmedQuestion) : "";

    setActiveChat((chat) =>
      chat?.chat_id === chatId
        ? { ...chat, messages: [...chat.messages, optimisticUser, optimisticAssistant] }
        : chat,
    );
    setQuestion("");
    setChatState({ loading: true, error: "" });

    try {
      if (firstQuestionTitle) {
        try {
          await renameChat(chatId, firstQuestionTitle);
        } catch {
          // The answer flow should still continue if only the title update fails.
        }
      }

      const { stopReason } = await streamChatWithPaper({
        question: trimmedQuestion,
        chatId,
        paperIds: activeChat.sources.length ? activeChat.sources.map((source) => source.paper_id) : undefined,
        topK: agentOptions.topK,
        scoreThreshold: 0.25,
        maxAgentSteps: agentOptions.maxAgentSteps,
        enableWebSearch: agentOptions.enableWebSearch,
        enableResearchIngest: agentOptions.enableResearchIngest,
        autoDownloadPdfs: agentOptions.autoDownloadPdfs,
        onToken: (token) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === optimisticAssistant.created_at
                  ? { ...message, content: `${message.content}${token}` }
                  : message,
              ),
            };
          });
        },
        onAgentStep: (step) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === optimisticAssistant.created_at
                  ? { ...message, trace: [...(message.trace ?? []), step] }
                  : message,
              ),
            };
          });
        },
        onCitations: (citations) => {
          setActiveChat((chat) => {
            if (!chat || chat.chat_id !== chatId) return chat;
            return {
              ...chat,
              messages: chat.messages.map((message) =>
                message.created_at === optimisticAssistant.created_at
                  ? { ...message, citations, streaming: false }
                  : message,
              ),
            };
          });
        },
      });
      setActiveChat((chat) => {
        if (!chat || chat.chat_id !== chatId) return chat;
        return {
          ...chat,
          messages: chat.messages.map((message) =>
            message.created_at === optimisticAssistant.created_at
              ? { ...message, stop_reason: stopReason, streaming: false }
              : message,
          ),
        };
      });
      const session = await getChatSession(chatId);
      setActiveChat((chat) => (chat?.chat_id === chatId ? session : chat));
      setChatState({ loading: false, error: "" });
      await refreshThreads();
    } catch (error) {
      setActiveChat((chat) => {
        if (!chat || chat.chat_id !== chatId) return chat;
        return {
          ...chat,
          messages: chat.messages.map((message) =>
            message.created_at === optimisticAssistant.created_at ? { ...message, streaming: false } : message,
          ),
        };
      });
      setChatState({ loading: false, error: error.message });
    }
  }

  function openCitation(citation) {
    if (citation.url) {
      window.open(citation.url, "_blank", "noopener,noreferrer");
      return;
    }

    const source = activeChat?.sources.find(
      (candidate) =>
        candidate.paper_id === citation.paper_id ||
        candidate.title === citation.title ||
        candidate.filename === citation.title,
    );
    const filename = source?.filename ?? (citation.title?.toLowerCase().endsWith(".pdf") ? citation.title : null);
    if (!filename) {
      setChatState({ loading: false, error: "Không mở được trích dẫn vì thiếu file PDF." });
      return;
    }
    setPaperPreview({
      ...(source ?? { paper_id: citation.paper_id, title: citation.title || citation.paper_id }),
      filename,
      pageNumber: citation.page_number ?? citation.page,
    });
  }

  const canChat = Boolean(activeChat) && !sourceState.loading;

  return (
    <div className={`chat-shell ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <ChatSidebar
        activeChat={activeChat}
        onBackHome={onBackHome}
        onCollapse={() => setSidebarOpen(false)}
        onCreateChat={createNewChat}
        onDeleteThread={setDeleteCandidate}
        onOpenThread={openChat}
        onRenameThread={renameChat}
        threads={threads}
        threadsState={threadsState}
      />

      <main className="chat-main">
        <ChatTopbar
          activeChat={activeChat}
          chatLoading={chatState.loading}
          onClearHistory={handleClearHistory}
          onOpenRuns={openRunHistory}
          onOpenSidebar={() => setSidebarOpen(true)}
          onOpenSources={() => setSourcePanelOpen(true)}
        />

        <div className="chat-workspace">
          <div className="chat-body">
            {!activeChat ? (
              <ChatEmptyState onCreateChat={createNewChat} />
            ) : (
              <ChatMessages
                activeChat={activeChat}
                canChat={canChat}
                chatState={chatState}
                onOpenCitation={openCitation}
                sourceState={sourceState}
              />
            )}
          </div>

          {activeChat ? (
            <ChatComposer
              agentOptions={agentOptions}
              canChat={canChat}
              chatLoading={chatState.loading}
              onAgentOptionsChange={(patch) => setAgentOptions((options) => ({ ...options, ...patch }))}
              onQuestionChange={setQuestion}
              onSubmit={handleAsk}
              question={question}
            />
          ) : null}
        </div>
      </main>

      {sourcePanelOpen ? (
        <SourcePanel
          activeChat={activeChat}
          downloadedPdfs={downloadedPdfs}
          onAddPdf={addPdfToChat}
          onClose={() => setSourcePanelOpen(false)}
          onRemoveSource={removeSource}
          sourceIds={sourceIds}
          sourceState={sourceState}
        />
      ) : null}

      {runPanelOpen ? (
        <AgentRunPanel
          activeTab={runPanelTab}
          findings={researchFindings}
          onSelectTab={setRunPanelTab}
          runs={agentRuns}
          runState={runState}
          onClose={() => setRunPanelOpen(false)}
        />
      ) : null}

      {deleteCandidate ? (
        <ConfirmDialog
          confirmLabel="Xóa"
          loading={threadsState.loading}
          message="Cuộc trò chuyện và danh sách nguồn sẽ bị xóa. File PDF local vẫn được giữ."
          onCancel={() => setDeleteCandidate(null)}
          onConfirm={() => deleteChat(deleteCandidate.chat_id)}
          title="Xóa cuộc trò chuyện?"
        />
      ) : null}

      {paperPreview ? (
        <PdfPreviewModal onClose={() => setPaperPreview(null)} source={paperPreview} />
      ) : null}
    </div>
  );
}

