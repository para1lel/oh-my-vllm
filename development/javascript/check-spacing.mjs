import { parse } from "acorn";
import { readFileSync, writeFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

function isDefinition(node) {
  if (node.type === "ExportNamedDeclaration" || node.type === "ExportDefaultDeclaration") {
    return node.declaration && isDefinition(node.declaration);
  }
  if (["FunctionDeclaration", "ClassDeclaration", "MethodDefinition"].includes(node.type)) return true;

  const isFunction = (value) => value && ["ArrowFunctionExpression", "FunctionExpression"].includes(value.type);

  if (node.type === "VariableDeclaration") return node.declarations.some((entry) => isFunction(entry.init));
  if (["Property", "PropertyDefinition"].includes(node.type)) return isFunction(node.value);
  return node.type === "ExpressionStatement" && node.expression.type === "AssignmentExpression"
    && isFunction(node.expression.right);
}

export function missingDefinitionSpacing(source, typescript = false) {
  const comments = [];
  const syntax = parse(typescript ? stripTypeScriptTypes(source, { mode: "strip" }) : source,
    { ecmaVersion: "latest", sourceType: "module", locations: true, onComment: comments });
  const insertions = new Map();

  function inspect(entries) {
    for (let index = 1; index < entries.length; index++) {
      const previous = entries[index - 1];
      const current = entries[index];
      if (!isDefinition(previous) && !isDefinition(current)) continue;
      let gap = source.slice(previous.end, current.start);
      for (const comment of comments.filter((entry) => entry.start >= previous.end && entry.end <= current.start).reverse()) {
        const start = comment.start - previous.end;
        const end = comment.end - previous.end;
        gap = gap.slice(0, start) + "COMMENT" + gap.slice(end);
      }
      if (/\n[ \t]*\n/.test(gap)) continue;
      const leading = comments.filter((comment) => comment.start >= previous.end
        && comment.end <= current.start && comment.loc.start.line > previous.loc.end.line);
      let start = current.start;
      if (leading.length && /^\s*$/.test(source.slice(leading.at(-1).end, current.start))) {
        start = leading[0].start;
      }
      const lineStart = source.lastIndexOf("\n", start - 1) + 1;
      if (lineStart >= previous.end) start = lineStart;
      insertions.set(start, source[start - 1] === "\n" ? "\n" : "\n\n");
    }
  }

  function visit(node) {
    for (const key of ["body", "properties", "consequent"]) {
      if (Array.isArray(node[key])) inspect(node[key]);
    }
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) {
        for (const child of value) if (child && typeof child.type === "string") visit(child);
      } else if (value && typeof value.type === "string") visit(value);
    }
  }

  visit(syntax);
  return insertions;
}

function main(arguments_) {
  const fix = arguments_[0] === "--fix";
  let failed = false;
  for (const path of arguments_.slice(Number(fix))) {
    let source = readFileSync(path, "utf8");
    const insertions = missingDefinitionSpacing(source, path.endsWith(".ts"));
    if (fix && insertions.size) {
      for (const [start, blank] of [...insertions].sort((left, right) => right[0] - left[0])) {
        source = source.slice(0, start) + blank + source.slice(start);
      }
      writeFileSync(path, source);
    } else {
      for (const start of insertions.keys()) {
        console.error(path + ":" + source.slice(0, start).split("\n").length + ": add a blank line before this definition");
      }
      failed ||= insertions.size > 0;
    }
  }
  process.exitCode = Number(failed);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main(process.argv.slice(2));
