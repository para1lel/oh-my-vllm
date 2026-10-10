import { createHash } from "node:crypto";
import { cp, mkdir, readFile, readdir, writeFile, chmod } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createCodeHighlighter } from "./highlight.mjs";
import { sourceNotes } from "../src/source-notes.mjs";
import { specifications } from "../src/source-specs.mjs";
import { chapters, coverage } from "../src/curriculum.mjs";
import { extractExcerpt } from "./excerpt.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(root, "..");
const cache = resolve(root, ".tools");
const dist = resolve(root, "dist");
const generated = resolve(root, "generated");
await Promise.all([mkdir(cache, { recursive: true }), mkdir(dist, { recursive: true }),
  mkdir(generated, { recursive: true })]);

const assets = [
  ["tweego.zip", "https://github.com/tmedwards/tweego/releases/download/v2.1.1/tweego-2.1.1-linux-x64.zip",
    "2ed11c3aa66c88ec5382b51061c7782152632b88916777428ab57a66feb7e5aa"],
  ["sugarcube.zip", "https://github.com/tmedwards/sugarcube-2/releases/download/v2.37.3/sugarcube-2.37.3-for-twine-2.1-local.zip",
    "da00a8c15ec4e88a9e231a3ff6c516c57055f84231bb999f869ed34ade353dab"],
  ["FiraCodeNerdFontMono-Regular.ttf", "https://raw.githubusercontent.com/ryanoasis/nerd-fonts/v3.5.1/patched-fonts/FiraCode/FiraCodeNerdFontMono-Regular.ttf",
    "53eebbaaa09c483ebb27fdd5cf21073cc4b760308e5d666728e9db2f41616c00"],
  ["FiraCode-LICENSE", "https://raw.githubusercontent.com/ryanoasis/nerd-fonts/v3.5.1/patched-fonts/FiraCode/LICENSE",
    "1d41e10031ab125302780a05ec4c91d218e47db0c7e37cf315cce5e608cdc25c"],
];
for (const [name, url, sha] of assets) {
  const path = resolve(cache, name);
  let bytes;
  try { bytes = await readFile(path); } catch {
    // curl validates TLS and follows the release redirect. Never execute unchecked bytes.
    execFileSync("curl", ["-fLsS", "--retry", "2", "--max-time", "120", url, "-o", path]);
    bytes = await readFile(path);
  }
  if (createHash("sha256").update(bytes).digest("hex") !== sha) {
    throw new Error("Asset checksum mismatch: " + name);
  }
}
execFileSync("unzip", ["-oq", resolve(cache, "tweego.zip"), "-d", cache]);
execFileSync("unzip", ["-oq", resolve(cache, "sugarcube.zip"), "-d", resolve(cache, "storyformats")]);
await chmod(resolve(cache, "tweego"), 0o755);

const font = resolve(root, "node_modules/@fontsource/lxgw-wenkai");
await mkdir(resolve(dist, "assets/wenkai/files"), { recursive: true });
for (const extension of ["woff2", "woff"]) {
  const name = "lxgw-wenkai-latin-500-normal." + extension;
  await cp(resolve(font, "files", name), resolve(dist, "assets/wenkai/files", name));
}
await cp(resolve(font, "500.css"), resolve(dist, "assets/wenkai/font.css"));
await cp(resolve(font, "LICENSE"), resolve(dist, "assets/wenkai/LICENSE"));
await cp(resolve(root, "node_modules/katex/dist"), resolve(dist, "assets/katex"), { recursive: true });
await cp(resolve(root, "node_modules/katex/LICENSE"), resolve(dist, "assets/katex/LICENSE"));
await cp(resolve(cache, "FiraCodeNerdFontMono-Regular.ttf"), resolve(dist, "assets/FiraCodeNerdFontMono-Regular.ttf"));
await cp(resolve(cache, "FiraCode-LICENSE"), resolve(dist, "assets/FiraCode-LICENSE"));
await cp(resolve(cache, "storyformats/sugarcube-2/LICENSE"), resolve(dist, "assets/SugarCube-LICENSE"));
await cp(resolve(root, "node_modules/shiki/LICENSE"), resolve(dist, "assets/Shiki-LICENSE"));
await cp(resolve(root, "node_modules/@shikijs/themes/LICENSE"), resolve(dist, "assets/Shiki-Themes-LICENSE"));

const snippets = {};
const highlighter = await createCodeHighlighter();
for (const [key, [path, begin, end, limit = 48]] of Object.entries(specifications)) {
  const source = await readFile(resolve(repo, path), "utf8");
  const language = sourceLanguage(path);
  const excerpt = extractExcerpt(source, { begin, end, limit, language });
  const { text } = excerpt;
  const notes = sourceNotes[key];
  if (!notes || !notes.entries.length || notes.entries.some((entry) => entry.length !== 3 || entry.some((value) => !value.trim()))) {
    throw new Error("Missing source explanation: " + key);
  }
  for (const [, field] of text.matchAll(/^\s*pub\s+(\w+)\s*:/gm)) {
    if (!notes.entries.some(([name]) => name === field)) throw new Error("Undocumented field: " + key + "." + field);
  }
  if (/^class /m.test(text)) {
    for (const [, field] of text.matchAll(/^    (\w+): /gm)) {
      if (!notes.entries.some(([name]) => name === field)) throw new Error("Undocumented field: " + key + "." + field);
    }
  }
  snippets[key] = {
    path,
    ...excerpt, language,
    tokens: highlighter.highlight(text, language),
    sha256: createHash("sha256").update(source).digest("hex"),
  };
}
const previewText = snippets.aligned.text.split("\n")
  .filter((line) => line.includes("let block =") || line.includes("let last_boundary ="))
  .map((line) => line.trim()).join("\n");
const alignmentPreview = {
  text: previewText, language: "rust", tokens: highlighter.highlight(previewText, "rust"),
};
const sourceCoverage = await buildSourceCoverage(highlighter);
highlighter.dispose();
const trace = JSON.parse(execFileSync(resolve(repo, "scripts/with-env.sh"), [
  "cargo", "run", "--quiet", "--locked", "--manifest-path", resolve(root, "trace/Cargo.toml"),
], { cwd: repo, encoding: "utf8" }));
const data = {
  snippets, alignmentPreview, trace, sourceNotes, chapters,
  coverage: sourceCoverage,
  sourceRevision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim(),
};
await writeFile(resolve(generated, "data.js"), "setup.journeyData = " + JSON.stringify(data) + ";\n");
execFileSync(resolve(cache, "tweego"), [
  "-f", "sugarcube-2", "--head", resolve(root, "src/head.html"),
  "-o", resolve(dist, "index.html"), resolve(root, "src/story.twee"),
  resolve(root, "src/style.css"), resolve(root, "src/setup.js"), resolve(generated, "data.js"),
  resolve(root, "src/chapters"),
], { cwd: root, env: { ...process.env, TWEEGO_PATH: resolve(cache, "storyformats") } });
console.log("Built genuine SugarCube 2.37.3 with real Rust scheduler traces.");

async function buildSourceCoverage(highlighter) {
  const directory = resolve(dist, "sources");
  await mkdir(directory, { recursive: true });
  await cp(resolve(root, "src/style.css"), resolve(dist, "style.css"));
  const records = [];
  const files = new Map();
  for (const [chapter, path, anchor, responsibility, decision] of coverage) {
    if (!chapters[chapter]) throw new Error("Unknown coverage chapter: " + chapter);
    const source = await readFile(resolve(repo, path), "utf8");
    const offset = source.indexOf(anchor);
    if (offset < 0) throw new Error("Missing coverage anchor: " + path + " / " + anchor);
    const line = source.slice(0, offset).split("\n").length;
    const name = path.replaceAll("/", "__") + ".html";
    const sha256 = createHash("sha256").update(source).digest("hex");
    const record = { chapter, path, line, sha256, responsibility, decision, url: "sources/" + name + "#L" + line };
    records.push(record);
    if (!files.has(path)) files.set(path, { source, name, entries: [] });
    files.get(path).entries.push(record);
  }
  // Include every substantive first-party runtime file. Frozen comparison
  // providers and test modules are deliberately outside the default call chain.
  for (const directory of ["crates", "python/oh_my_vllm"]) {
    for (const entry of await readdir(resolve(repo, directory), { recursive: true, withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const path = resolve(entry.parentPath, entry.name).slice(repo.length + 1);
      if (!/\.(rs|py|cu|cuh)$/.test(path) || /\/tests[/.]|\/tilelang_reference\//.test(path)) continue;
      if (!files.has(path)) {
        const text = await readFile(resolve(repo, path), "utf8");
        if (path.endsWith("/__init__.py") && !text.replace(/"""[\s\S]*?"""|#.*|\s/g, "")) continue;
        throw new Error("Runtime file lacks chapter coverage: " + path);
      }
    }
  }
  const escape = (text) => text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");
  for (const [path, { source, name, entries }] of files) {
    const language = sourceLanguage(path);
    const lines = highlighter.highlight(source, language).map((tokens, index) =>
      '<span class="source-line" id="L' + (index + 1) + '"><span class="line-number" aria-hidden="true">' + (index + 1) + '</span>' + tokens.map((token) =>
        '<span class="syntax-token" style="--syntax-light:' + token.light + ';--syntax-dark:' + token.dark + '">' + escape(token.content) + '</span>').join("") + '</span>').join("\n");
    await writeFile(resolve(directory, name), `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(path)} / 源码</title><link rel="stylesheet" href="../assets/wenkai/font.css"><link rel="stylesheet" href="../style.css"><script>try{document.documentElement.dataset.theme=localStorage.getItem('journey-theme')||'light'}catch{}</script></head><body><main id="story" class="source-viewer"><article><h1>完整源码</h1><p class="source-caption">${escape(path)} / SHA-256 ${entries[0].sha256}</p><ul class="navigation-list">${entries.map((entry) => '<li>如果想读懂此文件的调用背景, 返回 <a href="../index.html?chapter=' + entry.chapter + '">' + escape(chapters[entry.chapter].title) + '</a>. <p class="small">' + escape(entry.responsibility + '. ' + entry.decision) + '</p></li>').join("")}</ul><button class="text-button" id="source-theme" type="button">切换亮色 / 暗色</button><div class="code-block"><pre tabindex="0" aria-label="完整源码, 可横向滚动"><code class="language-${language}">${lines}</code></pre></div></article></main><script>document.getElementById('source-theme').onclick=()=>{const next=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=next;try{localStorage.setItem('journey-theme',next)}catch{}}</script></body></html>`);
  }
  return records;
}

function sourceLanguage(path) {
  if (path.endsWith(".rs")) return "rust";
  if (/\.(cu|cuh)$/.test(path)) return "cpp";
  return "python";
}
