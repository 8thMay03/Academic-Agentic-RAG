export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "Không rõ";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDateTime(value) {
  if (!value) return "Không rõ";
  return new Intl.DateTimeFormat("vi-VN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function formatEvidenceQuality(quality) {
  if (quality === "high") return "Cao";
  if (quality === "medium") return "Trung bình";
  if (quality === "low") return "Thấp";
  if (quality === "web") return "Web";
  return "Không rõ";
}

export function evidenceQualityClass(quality) {
  if (quality === "high") return "quality-high";
  if (quality === "medium") return "quality-medium";
  if (quality === "low") return "quality-low";
  if (quality === "web") return "quality-web";
  return "quality-unknown";
}
