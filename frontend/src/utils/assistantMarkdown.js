export function buildAssistantSegments(content) {
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

export function normalizeMathDelimiters(content) {
  return content
    .replace(/\\\[((?:.|\n)*?)\\\]/g, (_, formula) => `\n\n$$\n${formula.trim()}\n$$\n\n`)
    .replace(/\\\(((?:.|\n)*?)\\\)/g, (_, formula) => `$${formula.trim()}$`);
}

function normalizeAssistantMarkdown(content) {
  let text = content.replace(/\r\n/g, "\n").trim();
  text = normalizeFullyFlattenedTables(text);
  text = normalizeFlattenedTables(text);
  text = text.replace(/(^|\n)\s*---\s*(?=\n|$)/g, "\n\n---\n\n");
  text = text.replace(/\s+(#{1,6}\s+)/g, "\n\n$1");
  text = text.replace(/\s+(-\s+(?=(\*\*)?[A-ZÀ-Ỹ0-9]))/g, "\n$1");
  text = text.replace(/\s+(\d+\.\s+(?=[A-ZÀ-Ỹ0-9]))/g, "\n$1");
  text = text.replace(/\s*(\|[^|\n]+(?:\|[^|\n]+)+\|)\s*/g, "\n$1\n");
  text = splitEmbeddedTableRows(text);
  text = repairBrokenTableLines(text);
  text = normalizeMathDelimiters(text);
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function normalizeFullyFlattenedTables(content) {
  return content
    .split("\n")
    .map((line) => {
      const trimmedLine = line.trim();
      if (!trimmedLine.includes("|") || !trimmedLine.endsWith("|")) return line;

      const firstPipeIndex = trimmedLine.indexOf("|");
      const prefix = trimmedLine.slice(0, firstPipeIndex).trim();
      const tablePart = trimmedLine.slice(firstPipeIndex);
      const cells = splitTableCells(tablePart).map(cleanTableCell).filter(Boolean);
      const separatorStart = cells.findIndex((cell) => /^:?-{3,}:?$/.test(cell));
      if (separatorStart < 2) return line;

      const columnCount = separatorStart;
      const separators = cells.slice(separatorStart, separatorStart + columnCount);
      if (separators.length !== columnCount || !separators.every((cell) => /^:?-{3,}:?$/.test(cell))) return line;

      const bodyCells = cells.slice(separatorStart + columnCount);
      if (bodyCells.length < columnCount) return line;

      const rows = [];
      for (let index = 0; index < bodyCells.length; index += columnCount) {
        rows.push(padTableRow(bodyCells.slice(index, index + columnCount), columnCount));
      }

      const tableLines = [
        formatTableRow(cells.slice(0, columnCount)),
        formatTableRow(Array.from({ length: columnCount }, () => "---")),
        ...rows.map(formatTableRow),
      ];
      return [prefix, tableLines.join("\n")].filter(Boolean).join("\n");
    })
    .join("\n");
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
    if (!header || header.length < 2 || separatorIndex === -1 || !isSeparatorRow(lines[separatorIndex])) {
      repairedLines.push(lines[index]);
      continue;
    }

    const rows = [];
    let rowIndex = nextNonEmptyLineIndex(lines, separatorIndex + 1);
    while (rowIndex !== -1) {
      const row = parseLooseTableRow(lines[rowIndex], header.length);
      if (!row) break;
      rows.push(row);
      rowIndex = nextNonEmptyLineIndex(lines, rowIndex + 1);
    }

    if (!rows.length) {
      repairedLines.push(lines[index]);
      continue;
    }

    repairedLines.push(formatTableRow(header));
    repairedLines.push(formatTableRow(header.map(() => "---")));
    rows.forEach((row) => repairedLines.push(formatTableRow(padTableRow(row, header.length))));
    index = rowIndex === -1 ? lines.length : rowIndex - 1;
  }

  return repairedLines.join("\n");
}

function parsePipeCells(line) {
  const trimmedLine = line.trim();
  if (!trimmedLine.startsWith("|") || !trimmedLine.endsWith("|")) return null;
  const cells = splitTableCells(trimmedLine).map((cell) => cleanTableCell(cell)).filter(Boolean);
  return cells.length >= 2 ? cells : null;
}

function isSeparatorRow(line) {
  const cells = parsePipeCells(line);
  return Boolean(cells?.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell)));
}

function parseLooseTableRow(line, columnCount) {
  const pipeCells = parsePipeCells(line);
  if (pipeCells?.length >= 2) return pipeCells;

  const cells = splitTableCells(line.trim().replace(/^\|/, "")).map((cell) => cleanTableCell(cell)).filter(Boolean);
  if (cells.length < 2) return null;
  if (cells.length === columnCount) return cells;
  if (/^\d+\.\s+/.test(cells[0])) return cells;
  return null;
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

function formatTableRow(cells) {
  return `| ${cells.map(cleanTableCell).join(" | ")} |`;
}

function padTableRow(row, columnCount) {
  if (row.length === columnCount) return row;
  if (row.length > columnCount) return [...row.slice(0, columnCount - 1), row.slice(columnCount - 1).join(" | ")];
  return [...row, ...Array.from({ length: columnCount - row.length }, () => "")];
}

function splitEmbeddedTableRows(content) {
  return content
    .split("\n")
    .flatMap((line) => {
      const trimmedLine = line.trim();
      if (!trimmedLine || trimmedLine.startsWith("|")) return [line];

      const firstPipeIndex = findFirstTablePipe(trimmedLine);
      if (firstPipeIndex <= 0) return [line];

      const prefix = trimmedLine.slice(0, firstPipeIndex).trim();
      const tablePart = trimmedLine.slice(firstPipeIndex).trim();
      if (!prefix || !parsePipeCells(tablePart)) return [line];
      return [prefix, tablePart];
    })
    .join("\n");
}

function findFirstTablePipe(line) {
  for (let index = 0; index < line.length; index += 1) {
    if (line[index] === "|" && line.lastIndexOf("|") > index) return index;
  }
  return -1;
}

function splitTableCells(line) {
  const trimmedLine = line.trim();
  const content = trimmedLine.startsWith("|") ? trimmedLine.slice(1, trimmedLine.endsWith("|") ? -1 : undefined) : trimmedLine;
  const cells = [];
  let current = "";
  let dollarMath = false;
  let escapedInlineMath = false;
  let escapedDisplayMath = false;

  for (let index = 0; index < content.length; index += 1) {
    const char = content[index];
    const nextChar = content[index + 1];
    const previousChar = content[index - 1];

    if (char === "\\" && nextChar === "(") {
      escapedInlineMath = true;
      current += char;
      continue;
    }
    if (char === "\\" && nextChar === "[") {
      escapedDisplayMath = true;
      current += char;
      continue;
    }
    if (previousChar === "\\" && char === ")" && escapedInlineMath) escapedInlineMath = false;
    if (previousChar === "\\" && char === "]" && escapedDisplayMath) escapedDisplayMath = false;
    if (char === "$" && previousChar !== "\\") dollarMath = !dollarMath;

    if (char === "|" && !dollarMath && !escapedInlineMath && !escapedDisplayMath) {
      cells.push(current);
      current = "";
      continue;
    }
    current += char;
  }

  cells.push(current);
  return cells;
}

function readTableAt(lines, startIndex) {
  const headers = parsePipeCells(lines[startIndex]);
  if (!headers || headers.length < 2) return null;

  const separatorIndex = nextNonEmptyLineIndex(lines, startIndex + 1);
  if (separatorIndex === -1 || !isSeparatorRow(lines[separatorIndex])) return null;

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

    const row = parseLooseTableRow(line, headers.length);
    if (!row) break;

    rows.push(padTableRow(row.map(cleanTableCell), headers.length));
    endIndex = rowIndex;
    rowIndex += 1;
  }

  if (!rows.length) return null;
  return { headers: headers.map(cleanTableCell), rows, endIndex };
}

function pushMarkdownSegment(segments, markdownBuffer) {
  const content = markdownBuffer.join("\n").trim();
  if (content) segments.push({ type: "markdown", content });
}
