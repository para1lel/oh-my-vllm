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
