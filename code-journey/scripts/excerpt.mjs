// Remove only the whitespace prefix shared by all nonblank lines.
// Tabs retain their original meaning; compare literal prefixes, not visual columns.
export function dedent(text) {
  const lines = text.split("\n");
  const nonblank = lines.filter((line) => line.trim());
  if (!nonblank.length) return text;
  let prefix = nonblank[0].match(/^[\t ]*/)[0];
  for (const line of nonblank.slice(1)) {
    while (prefix && !line.startsWith(prefix)) prefix = prefix.slice(0, -1);
  }
  return lines.map((line) => line.trim() ? line.slice(prefix.length) : "").join("\n");
}

function completeMetadata(text, language) {
  // Quoted delimiters and comments do not change the metadata's nesting.
  const unquoted = text.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'/g, "");
  const clean = unquoted.replace(language === "rust" ? /\/\/[^\n]*/g : /#[^\n]*/g, "");
  const start = clean.search(/[([{]/);
  if (start < 0 || clean.slice(0, start).includes("\n")) return false;
  const stack = [];
  const pairs = { ")": "(", "]": "[", "}": "{" };
  for (let i = start; i < clean.length; i++) {
    const char = clean[i];
    if ("([{".includes(char)) stack.push(char);
    else if (pairs[char]) {
      if (stack.pop() !== pairs[char]) return false;
      if (!stack.length) return !clean.slice(i + 1).trim();
    }
  }
  return false;
}

// Include the documentation and metadata attached to the selected declaration.
// Stop between complete attachments; blank paragraphs inside them stay intact.
function declarationStart(lines, index, language) {
  const indent = lines[index].match(/^[\t ]*/)[0];
  const aligned = (line) => line.startsWith(indent) && !/^[\t ]/.test(line.slice(indent.length));
  while (index > 0) {
    const previous = lines[index - 1];
    const text = previous.trim();
    if (aligned(previous) && (language === "rust"
      ? /^\/\/\/(?!\/)|^#\[.*\]$|^\/\*\*.*\*\/$/.test(text)
      : /^@/.test(text))) {
      index--;
      continue;
    }
    const block = language === "rust" && text.endsWith("*/");
    const metadata = /^[)\]]+\s*$/.test(text);
    if (!block && !metadata) break;
    let start = index - 1;
    while (start >= 0) {
      const line = lines[start];
      const opener = block ? /^\/\*\*/ : language === "rust" ? /^#\[/ : /^@/;
      if (aligned(line) && opener.test(line.trim())) {
        if (!block && !completeMetadata(lines.slice(start, index).join("\n"), language)) start = -1;
        break;
      }
      // The nearest opening comment must be a doc comment for this element.
      if (block && line.includes("/*")) { start = -1; break; }
      start--;
    }
    if (start < 0) break;
    index = start;
  }
  return index;
}

export function extractExcerpt(source, { begin, end, limit = 48, language }) {
  const offset = source.indexOf(begin);
  if (offset < 0) throw new Error("Missing source anchor: " + begin);
  const stop = end ? source.indexOf(end, offset + begin.length) : source.length;
  if (stop < 0) throw new Error("Missing end anchor: " + end);
  const lines = source.split("\n");
  const anchorLine = source.slice(0, offset).split("\n").length - 1;
  const startLine = declarationStart(lines, anchorLine, language);
  const context = lines.slice(startLine, anchorLine);
  // The excerpt limit applies to the selected body, leaving leading docs intact.
  const body = source.slice(offset, stop).trimEnd().split("\n").slice(0, limit);
  const originalText = [...context, ...body].join("\n");
  return { line: startLine + 1, originalText, text: dedent(originalText) };
}
