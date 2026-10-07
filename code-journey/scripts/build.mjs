import { createHash } from "node:crypto";
import { cp, mkdir, readFile, writeFile, chmod } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createCodeHighlighter } from "./highlight.mjs";
import { sourceNotes, readingSyntax } from "../src/source-notes.mjs";

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

const specifications = {
  request: ["crates/scheduler/src/request.rs", "pub struct Request {", "\nimpl Request {"],
  tokenHistory: ["crates/scheduler/src/request.rs", "    pub token_ids: Vec<u32>,", "    /// Unverified MTP draft tokens appended"],
  computedCounters: ["crates/scheduler/src/request.rs", "    pub num_computed_tokens: usize,", "    /// Maximum number of output tokens"],
  aligned: ["crates/scheduler/src/lib.rs", "    fn aligned_prefill(", "    pub fn new("],
  queues: ["crates/scheduler/src/lib.rs", "        let mut token_budget", "        let mut still_running"],
  waiting: ["crates/scheduler/src/lib.rs", "        // ── phase 2:", "            let has_running"],
  output: ["crates/scheduler/src/output.rs", "pub struct ScheduledRequest", "// ── step output"],
  batchOutput: ["crates/scheduler/src/output.rs", "pub struct SchedulerOutput", "// ── worker feedback"],
  workerResult: ["crates/scheduler/src/output.rs", "pub struct WorkerOutput", null],
  validate: ["crates/scheduler/src/lib.rs", "            let final_chunk", "            let computed_after"],
  commit: ["crates/scheduler/src/lib.rs", "            req.num_computed_tokens =\n", "            let full_blocks"],
  worker: ["python/oh_my_vllm/worker/model_runner.py", "    def execute_model(", "        plans = []"],
  plan: ["python/oh_my_vllm/worker/batch_plan.py", "def plan_request(", null],
};
const snippets = {};
const highlighter = await createCodeHighlighter();
for (const [key, [path, begin, end]] of Object.entries(specifications)) {
  const source = await readFile(resolve(repo, path), "utf8");
  const start = source.indexOf(begin);
  if (start < 0) throw new Error("Missing source anchor: " + key);
  let stop = end ? source.indexOf(end, start + begin.length) : source.length;
  if (stop < 0) {
    if (key === "commit") stop = source.indexOf("        // Remove finished", start);
    if (stop < 0) throw new Error("Missing end anchor: " + key);
  }
  const lines = source.slice(start, stop).trimEnd().split("\n").slice(0, key === "plan" ? 24 : 48);
  const text = lines.join("\n");
  const notes = sourceNotes[key];
  if (!notes || !notes.entries.length || notes.entries.some((entry) => entry.length !== 3 || entry.some((value) => !value.trim()))) {
    throw new Error("Missing source explanation: " + key);
  }
  for (const [, field] of text.matchAll(/^\s*pub\s+(\w+)\s*:/gm)) {
    if (!notes.entries.some(([name]) => name === field)) throw new Error("Undocumented field: " + key + "." + field);
  }
  const language = path.endsWith(".rs") ? "rust" : "python";
  snippets[key] = {
    path,
    line: source.slice(0, start).split("\n").length,
    text, language,
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
highlighter.dispose();
const trace = JSON.parse(execFileSync(resolve(repo, "scripts/with-env.sh"), [
  "cargo", "run", "--quiet", "--locked", "--manifest-path", resolve(root, "trace/Cargo.toml"),
], { cwd: repo, encoding: "utf8" }));
const data = {
  snippets, alignmentPreview, trace, sourceNotes, readingSyntax,
  sourceRevision: execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim(),
};
await writeFile(resolve(generated, "data.js"), "setup.journeyData = " + JSON.stringify(data) + ";\n");
execFileSync(resolve(cache, "tweego"), [
  "-f", "sugarcube-2", "--head", resolve(root, "src/head.html"),
  "-o", resolve(dist, "index.html"), resolve(root, "src/story.twee"),
  resolve(root, "src/style.css"), resolve(root, "src/setup.js"), resolve(generated, "data.js"),
], { cwd: root, env: { ...process.env, TWEEGO_PATH: resolve(cache, "storyformats") } });
console.log("Built genuine SugarCube 2.37.3 with real Rust scheduler traces.");
