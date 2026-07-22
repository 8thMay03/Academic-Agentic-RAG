import { lazy, Suspense, useEffect, useState } from "react";
import { listDownloadedPdfs, uploadLocalPdfs } from "./api.js";

const ChatPage = lazy(() => import("./pages/ChatPage.jsx"));
const HomePage = lazy(() => import("./pages/HomePage.jsx"));
const PaperDetailPage = lazy(() => import("./pages/PaperDetailPage.jsx"));

export default function App() {
  const [view, setView] = useState("home");
  const [papers, setPapers] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [chatInitialPaper, setChatInitialPaper] = useState(null);

  const [listState, setListState] = useState({ loading: true, error: "" });
  const [uploadState, setUploadState] = useState({ loading: false, error: "", message: "" });

  useEffect(() => {
    void refreshPapers();
  }, []);

  async function refreshPapers() {
    setListState({ loading: true, error: "" });
    try {
      const response = await listDownloadedPdfs();
      setPapers(response ?? []);
      setListState({ loading: false, error: "" });
    } catch (error) {
      setListState({ loading: false, error: error.message });
    }
  }

  async function handleUpload(files) {
    const pdfFiles = Array.from(files ?? []).filter(
      (file) => file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf"),
    );
    if (!pdfFiles.length) {
      setUploadState({ loading: false, error: "Chọn ít nhất một file PDF.", message: "" });
      return;
    }

    setUploadState({ loading: true, error: "", message: `Đang tải ${pdfFiles.length} file...` });
    try {
      await uploadLocalPdfs(pdfFiles);
      await refreshPapers();
      setUploadState({
        loading: false,
        error: "",
        message: `Đã tải lên ${pdfFiles.length} paper.`,
      });
    } catch (error) {
      setUploadState({ loading: false, error: error.message, message: "" });
    }
  }

  function openPaper(paper) {
    setSelectedPaper(paper);
    setView("paper");
  }

  function startChat() {
    setChatInitialPaper(null);
    setView("chat");
  }

  async function chatWithPaper(paper) {
    setChatInitialPaper(paper);
    setView("chat");
  }

  function goHome() {
    setView("home");
    setSelectedPaper(null);
    setChatInitialPaper(null);
    void refreshPapers();
  }

  if (view === "chat") {
    return (
      <Suspense fallback={<RouteLoading />}>
        <ChatPage initialPaper={chatInitialPaper} onBackHome={goHome} />
      </Suspense>
    );
  }

  if (view === "paper" && selectedPaper) {
    return (
      <Suspense fallback={<RouteLoading />}>
        <PaperDetailPage
          onBack={goHome}
          onChatWithPaper={chatWithPaper}
          paper={selectedPaper}
        />
      </Suspense>
    );
  }

  return (
    <Suspense fallback={<RouteLoading />}>
      <HomePage
        error={listState.error}
        loading={listState.loading}
        onOpenPaper={openPaper}
        onRefresh={refreshPapers}
        onStartChat={startChat}
        onUpload={handleUpload}
        papers={papers}
        uploadState={uploadState}
      />
    </Suspense>
  );
}

function RouteLoading() {
  return (
    <div className="route-loading" role="status" aria-live="polite">
      Đang tải
    </div>
  );
}
