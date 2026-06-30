export function paperIdFromFilename(filename) {
  return filename.replace(/\.pdf$/i, "");
}

export function displayTitleFromFilename(filename) {
  const stem = filename.replace(/\.pdf$/i, "");
  return stem.replace(/[-_]/g, " ");
}

export function sourceFromPdf(pdf, paperId = paperIdFromFilename(pdf.filename)) {
  return {
    paper_id: paperId,
    title: displayTitleFromFilename(pdf.filename),
    filename: pdf.filename,
    path: pdf.path,
  };
}
