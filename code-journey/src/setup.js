/* Story state belongs to SugarCube; the helper only renders code and trace data. */
Config.history.maxStates = 100;
Config.passages.nobr = true;
Config.passages.transitionOut = 0;
Config.saves.isAllowed = () => false;

setup.lessons = {
  Start: { title: "从一条请求开始", needs: [] },
  Token: { title: "文字怎样变成 token?", needs: [] },
  Position: { title: "一个 token, 两种编号", needs: ["Token"] },
  Prefill: { title: "先读完, 再接着写", needs: ["Position"] },
  Ownership: { title: "谁决定, 谁计算?", needs: ["Prefill"] },
  Request: { title: "请求带着哪些东西?", needs: ["Prefill"] },
  Queue: { title: "这一轮, 轮到谁?", needs: ["Request"] },
  Pages: { title: "为什么是 784?", needs: ["Queue"] },
  Experiment: { title: "预算为什么留下一截?", needs: ["Pages"] },
  Worker: { title: "worker 怎样返回结果?", needs: ["Experiment", "Ownership"] },
  Commit: { title: "第一个 token 回到了哪里?", needs: ["Worker"] },
};
const storageKey = "oh-my-vllm-journey-v1";
setup.load = function () {
  // Use one checked persistence schema instead of SugarCube session auto-restore.
  session.delete("state");
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(storageKey) || "{}") || {}; } catch {}
  if (!saved || typeof saved !== "object") saved = {};
  const valid = (list) => Array.isArray(list) ? list.filter((name) => Object.hasOwn(setup.lessons, name)) : [];
  State.variables.read = valid(saved.read);
  State.variables.visited = valid(saved.visited);
  setup.readMemory = State.variables.read.slice();
  setup.visitedMemory = State.variables.visited.slice();
  State.variables.goal = Object.hasOwn(setup.lessons, saved.goal) ? saved.goal : "Request";
  State.variables.interest = ["basics", "request", "gpu"].includes(saved.interest) ? saved.interest : "basics";
  State.variables.expLength = [784, 785, 1568, 1569, 32768].includes(saved.expLength) ? saved.expLength : 32768;
  State.variables.expIndex = Number.isInteger(saved.expIndex) ? Math.max(-1, Math.min(saved.expIndex, setup.journeyData.trace[State.variables.expLength].length - 1)) : -1;
  if (Object.hasOwn(setup.lessons, saved.last)) {
    Config.passages.start = setup.nextMissing(saved.last);
  }
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
setup.navigate = function (target, interest) {
  setup.complete();
  if (interest) {
    State.variables.interest = interest;
    }
  if (target !== "Map") State.variables.goal = target;
  setup.save();
  Engine.play(target === "Map" ? target : setup.nextMissing(target));
};
Macro.add("route", {
  handler() {
    const [text, target, interest] = this.args;
    if (target !== "Map" && !setup.lessons[target]) return this.error("Unknown lesson: " + target);
    const link = document.createElement("a");
    link.href = "#";
    link.className = "link-internal";
    link.textContent = text;
    link.addEventListener("click", (event) => {
      event.preventDefault();
      setup.navigate(target, interest);
    });
    this.output.appendChild(link);
  },
});
Macro.add("nextReading", {
  handler() {
    const goal = State.variables.goal;
    let target;
    if (!State.variables.read.includes(goal) && goal !== State.passage) target = goal;
    else target = this.args[0];
    new Wikifier(this.output, '<<route "' + this.args[1] + '" "' + target + '">>');
  },
});
Macro.add("source", {
  handler() {
    const [key, title] = this.args;
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
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    for (const [index, line] of source.text.split("\n").entries()) {
      const row = document.createElement("span");
      row.className = "source-line";
      const number = document.createElement("span");
      number.className = "line-number";
      number.setAttribute("aria-hidden", "true");
      number.textContent = source.line + index;
      row.append(number, document.createTextNode(line));
      code.appendChild(row);
    }
    pre.appendChild(code);
    details.appendChild(pre);
    this.output.appendChild(details);
  },
});
Macro.add("alignmentPreview", {
  handler() {
    const pre = document.createElement("pre");
    pre.className = "excerpt-preview";
    const code = document.createElement("code");
    code.textContent = setup.journeyData.snippets.aligned.text.split("\n").filter((line) => line.includes("let block =") || line.includes("let last_boundary =")).map((line) => line.trim()).join("\n");
    pre.appendChild(code); this.output.appendChild(pre);
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
  if (State.passage === "Start" && !v.read.includes("Start")) v.read.push("Start");
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
  document.getElementById("read-count").textContent = "已读 " + v.read.length + " 篇";
  document.getElementById("restart").onclick = () => {
    v.read = []; v.visited = []; setup.readMemory = []; setup.visitedMemory = []; v.goal = "Request"; v.interest = "basics";
    v.expLength = 32768; v.expIndex = -1; v.lastLesson = "Start";
    try { localStorage.removeItem(storageKey); } catch {}
    State.reset();
    Config.passages.start = "Start";
    setup.load();
    Engine.play("Start");
  };
  const back = document.getElementById("journey-back");
  if (back) {
    back.disabled = State.length <= 1;
    back.hidden = State.length <= 1;
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
