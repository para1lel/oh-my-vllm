/* Story state belongs to SugarCube; the helper only renders code and trace data. */
Config.history.maxStates = 100;
Config.passages.nobr = true;
Config.passages.transitionOut = 0;
Config.saves.isAllowed = () => false;

setup.lessons = {};
const storageKey = "oh-my-vllm-journey-v2";
setup.load = function () {
  setup.lessons = setup.journeyData.chapters;
  session.delete("state");
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || "null"); } catch {}
  if (!saved || typeof saved !== "object" || Array.isArray(saved)) {
    let old = {};
    try { old = JSON.parse(localStorage.getItem("oh-my-vllm-journey-v1") || "{}"); } catch {}
    const mapped = { Token: "Basics", Position: "Basics", Prefill: "Basics", Ownership: "Ownership", Request: "Requests", Queue: "Budget", Pages: "Cache", Experiment: "Budget", Worker: "Protocol", Commit: "Requests" };
    const oldList = Array.isArray(old?.visited) ? old.visited : [];
    saved = { ...old, read: [], visited: oldList.map((id) => mapped[id]).filter(Boolean),
      last: mapped[old?.last] || "Start", goal: mapped[old?.goal] || "Basics" };
  }
  const valid = (list) => Array.isArray(list) ? [...new Set(list.filter((name) => Object.hasOwn(setup.lessons, name)))] : [];
  const v = State.variables;
  v.read = valid(saved.read); v.visited = valid(saved.visited);
  setup.readMemory = v.read.slice(); setup.visitedMemory = v.visited.slice();
  v.goal = Object.hasOwn(setup.lessons, saved.goal) ? saved.goal : "Basics";
  v.interest = ["basics", "request", "gpu"].includes(saved.interest) ? saved.interest : "basics";
  v.expLength = [784, 785, 1568, 1569, 32768].includes(saved.expLength) ? saved.expLength : 32768;
  v.expIndex = Number.isInteger(saved.expIndex) ? Math.max(-1, Math.min(saved.expIndex, setup.journeyData.trace[v.expLength].length - 1)) : -1;
  const requested = new URLSearchParams(location.search).get("chapter");
  if (Object.hasOwn(setup.lessons, requested)) {
    v.goal = requested; Config.passages.start = setup.nextMissing(requested);
    const url = new URL(location.href); url.searchParams.delete("chapter");
    history.replaceState(null, "", url);
  } else if (Object.hasOwn(setup.lessons, saved.last)) {
    Config.passages.start = setup.nextMissing(saved.last);
  } else Config.passages.start = "Start";
};
setup.save = function () {
  const v = State.variables;
  try {
    localStorage.setItem(storageKey, JSON.stringify({
      read: v.read, visited: v.visited, goal: v.goal, interest: v.interest,
      expLength: v.expLength, expIndex: v.expIndex,
      last: setup.lessons[State.passage] ? State.passage : v.lastLesson || "Start",
    }));
  } catch {}
};
setup.nextMissing = function (target) {
  for (const required of setup.lessons[target].needs) {
    if (!State.variables.read.includes(required)) return setup.nextMissing(required);
  }
  return target;
};
setup.complete = function () {
  const current = State.passage;
  if (setup.lessons[current] && !State.variables.read.includes(current)) State.variables.read.push(current);
  setup.readMemory = [...new Set([...setup.readMemory, ...State.variables.read])];
};
setup.navigate = function (target, interest, markComplete = false) {
  if (markComplete) setup.complete();
  if (interest) {
    State.variables.interest = interest;
    }
  if (setup.lessons[target]) State.variables.goal = target;
  setup.save();
  Engine.play(setup.lessons[target] ? setup.nextMissing(target) : target);
};
Macro.add("route", {
  handler() {
    const [text, target, interest, markComplete = false] = this.args;
    if (!["Map", "Start"].includes(target) && !setup.lessons[target]) return this.error("Unknown lesson: " + target);
    const link = document.createElement("a");
    link.href = "#";
    link.className = "link-internal";
    link.textContent = text;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setup.navigate(target, interest, markComplete);
    });
    this.output.appendChild(link);
  },
});
Macro.add("icode", {
  handler() {
    // Text insertion prevents Twine markup (e.g. Python //) changing prose DOM.
    const code = document.createElement("code");
    code.textContent = this.args[0];
    this.output.appendChild(code);
  },
});
setup.conditionLink = function (list, condition, target, label, interest, markComplete = false) {
  const item = document.createElement("li");
  item.append(document.createTextNode(condition + " "));
  new Wikifier(item, '<<route ' + JSON.stringify(label || setup.lessons[target]?.title || target) + ' ' + JSON.stringify(target) + ' ' + JSON.stringify(interest || null) + ' ' + JSON.stringify(markComplete) + '>>');
  item.append(document.createTextNode(".")); list.appendChild(item);
};
Macro.add("chapterHeader", { handler() {
  const chapter = setup.lessons[State.passage];
  const title = document.createElement("h1"); title.textContent = "第 " + chapter.number + " 章. " + chapter.title;
  const lead = document.createElement("p"); lead.className = "chapter-lead"; lead.textContent = chapter.question;
  this.output.append(title, lead);
} });
Macro.add("chapterChoices", { handler() {
  const id = State.passage, chapter = setup.lessons[id];
  const list = document.createElement("ul"); list.className = "navigation-list chapter-choices";
  const goal = State.variables.goal;
  if (goal !== id && !State.variables.read.includes(goal))
    setup.conditionLink(list, "如果已理解本章并想继续之前选择的主题, 阅读", goal, null, null, true);
  const next = Object.keys(setup.lessons)[chapter.number];
  if (next && next !== goal) setup.conditionLink(list, "如果已理解本章并想沿推理流程继续, 阅读", next, null, null, true);
  if (!next) setup.conditionLink(list, "如果已理解本章并想查看整套阅读进度, 打开", "Map", "阅读地图", null, true);
  const previous = chapter.needs[0];
  if (previous) setup.conditionLink(list, "如果想复习本章用到的前置知识, 阅读", previous);
  setup.conditionLink(list, "如果想选择其他主题或查看进度, 打开", "Map", "阅读地图");
  this.output.appendChild(list);
} });
Macro.add("readingMap", { handler() {
  const list = document.createElement("ul"); list.className = "navigation-list map-list";
  for (const [id, chapter] of Object.entries(setup.lessons)) {
    setup.conditionLink(list, "如果想了解" + chapter.question.replace(/[?]$/, "") + ", 阅读", id, chapter.title);
    const item = list.lastElementChild;
    const status = document.createElement("span"); status.className = "read-status";
    status.textContent = State.variables.read.includes(id) ? "已读" : State.variables.visited.includes(id) ? "阅读中" : "待阅读";
    item.append(document.createTextNode(" "), status);
  }
  this.output.appendChild(list);
} });
Macro.add("coverage", { handler() {
  const chapter = this.args[0] || State.passage;
  const section = document.createElement("section"); section.className = "source-coverage";
  const heading = document.createElement("h2"); heading.textContent = "本章源码与调用位置";
  const intro = document.createElement("p"); intro.className = "small";
  intro.textContent = "以下链接打开当前构建的完整源码. 行号定位到本章入口, 文件摘要用于核对版本.";
  const list = document.createElement("ul"); list.className = "navigation-list coverage-list";
  for (const source of setup.journeyData.coverage.filter((item) => item.chapter === chapter)) {
    const item = document.createElement("li");
    item.append(document.createTextNode("如果想查看 " + source.responsibility + " 的完整实现, 打开 "));
    const link = document.createElement("a"); link.href = source.url; link.textContent = source.path + ":" + source.line;
    item.append(link, document.createTextNode("."));
    const detail = document.createElement("p"); detail.className = "small"; detail.textContent = source.decision;
    item.appendChild(detail); list.appendChild(item);
  }
  section.append(heading, intro, list); this.output.appendChild(section);
} });
setup.copyCode = async function (text) {
  if (navigator.clipboard?.writeText) {
    try { await navigator.clipboard.writeText(text); return; } catch {}
  }
  // The intranet preview uses HTTP, where Clipboard API may be unavailable.
  const active = document.activeElement;
  const selection = window.getSelection();
  const ranges = Array.from({ length: selection.rangeCount }, (_, index) => selection.getRangeAt(index).cloneRange());
  const input = document.createElement("textarea");
  input.value = text;
  input.readOnly = true;
  input.className = "copy-buffer";
  input.setAttribute("aria-hidden", "true");
  input.tabIndex = -1;
  document.body.appendChild(input);
  let copied;
  try { input.select(); copied = document.execCommand("copy"); }
  finally {
    input.remove();
    selection.removeAllRanges();
    for (const range of ranges) selection.addRange(range);
    active?.focus({ preventScroll: true });
  }
  if (!copied) throw new Error("Clipboard copy failed");
};
setup.codeBlock = function (source, preview = false) {
  const block = document.createElement("div");
  block.className = "code-block" + (preview ? " excerpt-preview" : "");
  const language = source.language === "rust" ? "Rust" : "Python";
  const toolbar = document.createElement("div");
  toolbar.className = "code-toolbar";
  const label = document.createElement("span");
  label.className = "code-language";
  label.textContent = language;
  const copy = document.createElement("button");
  copy.className = "code-copy";
  copy.type = "button";
  copy.textContent = "复制代码";
  copy.setAttribute("aria-label", "复制 " + language + " 代码");
  const status = document.createElement("span");
  status.className = "sr-only";
  status.setAttribute("role", "status");
  let reset;
  copy.onclick = async () => {
    clearTimeout(reset);
    try {
      await setup.copyCode(source.text);
      copy.textContent = "已复制";
      status.textContent = "已复制 " + language + " 代码.";
    } catch {
      copy.textContent = "复制失败";
      status.textContent = "复制失败, 请选中代码复制.";
    }
    reset = setTimeout(() => { copy.textContent = "复制代码"; status.textContent = ""; }, 1800);
  };
  toolbar.append(label, copy, status);
  const pre = document.createElement("pre");
  pre.tabIndex = 0;
  pre.setAttribute("aria-label", language + " 代码, 可横向滚动");
  const code = document.createElement("code");
  code.className = "language-" + source.language;
  for (const [index, line] of source.tokens.entries()) {
    if (index) code.appendChild(document.createTextNode("\n"));
    const row = document.createElement("span");
    row.className = "source-line";
    if (!preview) {
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = source.line + index;
      row.appendChild(number);
    }
    for (const token of line) {
      const span = document.createElement("span");
      span.className = "syntax-token";
      span.textContent = token.content;
      for (const theme of ["light", "dark"]) {
        span.style.setProperty("--syntax-" + theme, token[theme]);
        const style = token[theme + "Style"] || 0;
        span.style.setProperty("--syntax-" + theme + "-style", style & 1 ? "italic" : "normal");
        span.style.setProperty("--syntax-" + theme + "-weight", style & 2 ? "700" : "400");
        span.style.setProperty("--syntax-" + theme + "-decoration", style & 4 ? "underline" : "none");
      }
      row.appendChild(span);
    }
    code.appendChild(row);
  }
  pre.appendChild(code);
  block.append(toolbar, pre);
  return block;
};
setup.sourceGuide = function (key, withSyntax = false) {
  const notes = setup.journeyData.sourceNotes[key];
  const section = document.createElement("section");
  section.className = "field-guide";
  section.dataset.sourceKey = key;
  const heading = document.createElement("h3");
  heading.textContent = notes.title;
  const intro = document.createElement("p");
  intro.textContent = notes.intro;
  section.append(heading, intro);
  if (withSyntax) {
    const syntax = document.createElement("p");
    syntax.className = "syntax-guide";
    syntax.textContent = setup.journeyData.readingSyntax[setup.journeyData.snippets[key].language];
    section.appendChild(syntax);
  }
  const table = document.createElement("table");
  table.className = "field-table";
  table.setAttribute("aria-label", notes.title);
  const labels = ["字段 / 变量", "含义", "用途与例子"];
  const head = table.createTHead().insertRow();
  for (const label of labels) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.appendChild(cell);
  }
  const body = table.createTBody();
  for (const entry of notes.entries) {
    const row = body.insertRow();
    entry.forEach((text, index) => {
      const cell = row.insertCell();
      cell.dataset.label = labels[index];
      const content = index === 0 ? document.createElement("code") : document.createElement("span");
      content.textContent = text;
      cell.appendChild(content);
    });
  }
  section.appendChild(table);
  return section;
};
Macro.add("explain", {
  handler() {
    const key = this.args[0];
    if (!setup.journeyData.sourceNotes[key]) return this.error("Unknown explanation: " + key);
    this.output.appendChild(setup.sourceGuide(key));
  },
});
Macro.add("source", {
  handler() {
    const [key, title, includeGuide = true] = this.args;
    const source = setup.journeyData.snippets[key];
    if (!source) return this.error("Unknown source: " + key);
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = title;
    details.appendChild(summary);
    const caption = document.createElement("p");
    caption.className = "source-caption";
    caption.textContent = source.path + ":" + source.line + " / SHA-256 " + source.sha256.slice(0, 12);
    details.appendChild(caption);
    details.appendChild(setup.codeBlock(source));
    if (includeGuide) details.appendChild(setup.sourceGuide(key, true));
    else {
      const syntax = document.createElement("p");
      syntax.className = "syntax-guide";
      syntax.textContent = setup.journeyData.readingSyntax[source.language];
      details.appendChild(syntax);
    }
    this.output.appendChild(details);
  },
});
Macro.add("alignmentPreview", {
  handler() {
    this.output.appendChild(setup.codeBlock(setup.journeyData.alignmentPreview, true));
  },
});
Macro.add("math", {
  handler() {
    const element = document.createElement("div");
    element.className = "math-block";
    try { katex.render(this.args[0], element, { displayMode: true, throwOnError: true, trust: false }); }
    catch (error) { return this.error(error.message); }
    this.output.appendChild(element);
  },
});
Macro.add("imath", {
  handler() {
    const element = document.createElement("span");
    element.className = "inline-math";
    try { katex.render(this.args[0], element, { displayMode: false, throwOnError: true, trust: false }); }
    catch (error) { return this.error(error.message); }
    this.output.appendChild(element);
  },
});

setup.renderExperiment = function (container) {
  const v = State.variables;
  const steps = setup.journeyData.trace[v.expLength];
  const step = v.expIndex >= 0 ? steps[v.expIndex] : { start: 0, count: 0, computed: 0, outputs: 0 };
  const completed = v.expIndex === steps.length - 1;
  const percentage = Math.round(step.computed / v.expLength * 100);
  const options = container.querySelector(".case-options");
  options.replaceChildren();
  for (const length of [784, 785, 1568, 1569, 32768]) {
    const button = document.createElement("button");
    button.className = "text-button";
    button.textContent = length;
    button.type = "button";
    button.setAttribute("aria-pressed", String(length === v.expLength));
    button.addEventListener("click", () => {
      v.expLength = length; v.expIndex = -1;
      setup.save(); setup.renderExperiment(container);
    });
    options.appendChild(button);
  }
  const first = steps[0].count;
  const expression = first < v.expLength
    ? "\\left\\lfloor\\frac{" + v.expLength + "}{784}\\right\\rfloor \\times 784 = " + first
    : "n_{\\mathrm{first}} = " + first;
  katex.render(expression, container.querySelector(".math-block"), { displayMode: true, throwOnError: true });
  const track = container.querySelector(".track");
  track.setAttribute("aria-valuenow", String(step.computed));
  track.setAttribute("aria-valuemax", String(v.expLength));
  container.querySelector(".track-fill").style.width = (step.computed / v.expLength * 100) + "%";
  container.querySelector("[data-value='percent']").textContent = percentage + "%";
  container.querySelector("[data-value='remaining-percent']").textContent = (100 - percentage) + "%";
  for (const key of ["computed", "count", "outputs"]) container.querySelector("[data-value='" + key + "']").textContent = step[key];
  const status = container.querySelector(".lab-status");
  if (v.expIndex < 0) status.textContent = "请求还在等待. 点击按钮, 执行第一轮调度.";
  else if (!completed) status.textContent = "第一轮停在第 " + (step.computed / 784) + " 个完整页的末尾. 剩下 " + (v.expLength - step.computed) + " 个 token 会在下一轮处理.";
  else status.textContent = "提示词的 " + v.expLength + " 个 token 已计算完. worker 按契约返回 1 个输出 token.";
  const button = container.querySelector("[data-action='step']");
  button.disabled = completed;
  button.textContent = completed ? "已得到首 token" : "执行下一步";
  setup.save();
};
$(document).on(":passagestart", () => {
  // History restores variables before rendering; merge progress before macros run.
  State.variables.read = [...new Set([...setup.readMemory, ...State.variables.read])];
  State.variables.visited = [...new Set([...setup.visitedMemory, ...State.variables.visited])];
});
$(document).on(":passageend", (event) => {
  const v = State.variables;
  v.read = [...new Set([...setup.readMemory, ...v.read])];
  v.visited = [...new Set([...setup.visitedMemory, ...v.visited])];
  if (setup.lessons[State.passage]) {
    if (!v.visited.includes(State.passage)) v.visited.push(State.passage);
    v.lastLesson = State.passage;
  }
  setup.readMemory = v.read.slice();
  setup.visitedMemory = v.visited.slice();
  document.getElementById("journey-home").onclick = (event) => { event.preventDefault(); Engine.play("Start"); };
  document.getElementById("journey-map").onclick = (event) => { event.preventDefault(); Engine.play("Map"); };
  const theme = document.getElementById("theme-toggle");
  theme.textContent = document.documentElement.dataset.theme === "dark" ? "亮色" : "暗色";
  theme.setAttribute("aria-label", document.documentElement.dataset.theme === "dark" ? "切换到亮色模式" : "切换到暗色模式");
  theme.onclick = () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("journey-theme", next); } catch {}
    theme.textContent = next === "dark" ? "亮色" : "暗色";
    theme.setAttribute("aria-label", next === "dark" ? "切换到亮色模式" : "切换到暗色模式");
  };
  document.getElementById("read-count").textContent = "已读 " + v.read.length + " / 12 章";
  document.getElementById("restart").onclick = () => {
    v.read = []; v.visited = []; setup.readMemory = []; setup.visitedMemory = []; v.goal = "Basics"; v.interest = "basics";
    v.expLength = 32768; v.expIndex = -1; v.lastLesson = "Start";
    try { localStorage.removeItem(storageKey); localStorage.removeItem("oh-my-vllm-journey-v1"); } catch {}
    State.reset();
    Config.passages.start = "Start";
    setup.load();
    Engine.play("Start");
  };
  const back = document.getElementById("journey-back");
  if (back) {
    back.disabled = State.length <= 1;
    back.closest("li").hidden = State.length <= 1;
    back.onclick = () => Engine.backward();
  }
  const container = event.content.querySelector("[data-experiment]");
  if (container) {
    setup.renderExperiment(container);
    container.querySelector("[data-action='step']").onclick = () => {
      v.expIndex++; setup.renderExperiment(container);
    };
    container.querySelector("[data-action='reset']").onclick = () => {
      v.expIndex = -1; setup.renderExperiment(container);
    };
  }
  setup.save();
  document.documentElement.lang = "zh-CN";
  const heading = event.content.querySelector("article h1");
  if (heading) { heading.setAttribute("tabindex", "-1"); heading.focus({ preventScroll: true }); }
});
