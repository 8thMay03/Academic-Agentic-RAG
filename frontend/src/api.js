const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers ?? {}),
    },
    ...options,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const message = payload?.detail ?? `Request failed with status ${response.status}`;
    throw new Error(Array.isArray(message) ? JSON.stringify(message) : message);
  }
  return payload;
}

export async function searchPapers({ query, maxResults, sortBy }) {
  return request("/search", {
    method: "POST",
    body: JSON.stringify({
      query,
      max_results: maxResults,
      sort_by: sortBy,
    }),
  });
}

export async function downloadPapers(pdfUrls) {
  return request("/papers/download", {
    method: "POST",
    body: JSON.stringify({ pdf_urls: pdfUrls }),
  });
}

export async function listDownloadedPdfs() {
  return request("/papers/pdfs");
}

export async function indexDownloadedPdf(filename) {
  return request("/papers/pdfs/index", {
    method: "POST",
    body: JSON.stringify({ filename }),
  });
}

export async function chatWithPaper({ question, chatId, paperIds, topK, scoreThreshold }) {
  return request("/chat", {
    method: "POST",
    body: JSON.stringify({
      question,
      chat_id: chatId,
      paper_ids: paperIds,
      top_k: topK,
      score_threshold: scoreThreshold,
    }),
  });
}

export async function getChatHistory(paperId) {
  return request(`/chat/history/${encodeURIComponent(paperId)}`);
}

export async function listChatThreads() {
  return request("/chat/history");
}

export async function createChatSession(title) {
  return request("/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title }),
  });
}

export async function getChatSession(chatId) {
  return request(`/chat/sessions/${encodeURIComponent(chatId)}`);
}

export async function deleteChatSession(chatId) {
  return request(`/chat/sessions/${encodeURIComponent(chatId)}`, {
    method: "DELETE",
  });
}

export async function addChatSource(chatId, source) {
  return request(`/chat/sessions/${encodeURIComponent(chatId)}/sources`, {
    method: "POST",
    body: JSON.stringify(source),
  });
}

export async function removeChatSource(chatId, paperId) {
  return request(`/chat/sessions/${encodeURIComponent(chatId)}/sources/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
}

export async function clearChatHistory(paperId) {
  return request(`/chat/history/${encodeURIComponent(paperId)}`, {
    method: "DELETE",
  });
}

export function getPdfFileUrl(filename) {
  return `${API_BASE_URL}/papers/pdfs/${encodeURIComponent(filename)}/content`;
}
