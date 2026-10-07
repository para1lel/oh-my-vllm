import { test, expect } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";

import { createHash } from "node:crypto";

const evidence = process.env.JOURNEY_QA_DIR || "/tmp/oh-my-vllm-journey-qa";
const link = (page, name) => page.getByRole("link", { name, exact: true });
async function snapshot(page, name, codeBlock) {
  await mkdir(evidence, { recursive: true });
  await expect(page.locator(".passage-out")).toHaveCount(0);
  await page.locator("#passages h1").waitFor();
  await page.evaluate(async () => {
    await document.fonts.load('500 17px "LXGW WenKai"');
    await document.fonts.load('400 14px "Fira Code Nerd Font"');
    await document.fonts.ready;
    window.scrollTo(0, 0);
  });
  await page.screenshot({ path: evidence + "/" + name + ".png", fullPage: true, animations: "disabled" });
  if (name === "light-reading" || name === "dark-reading" || name === "light-experiment") {
    await page.screenshot({ path: evidence + "/" + name + "-viewport.png", fullPage: false, animations: "disabled" });
  }
  if (codeBlock) await codeBlock.screenshot({ path: evidence + "/" + name + "-code.png", animations: "disabled" });
}
async function mainRoute(page) {
  await page.goto("/");
  await link(page, "跟着请求往前走").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("分词: 文字, 词表与 token ID");
  await expect(page.getByRole("heading", { level: 1 })).toBeFocused();
  for (const text of [
    "我读完了, 接着看位置", "我读完了, 看看模型怎样接着写",
    "我读完了, 继续走选中的路线", "我读完了, 让请求进入队列",
    "我读完了, 看看缓存的分块单位", "我读完了, 亲手执行两轮调度",
  ]) await link(page, text).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("分块实验: 32768 个 token 如何分两轮计算");
  await expect(page.getByRole("heading", { level: 1 })).toBeFocused();
}

test("genuine Twine, prerequisite routing, real Rust trace and worker branch", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  await page.goto("/");
  await expect(page).toHaveTitle("oh-my-vllm 代码之旅");
  await expect(page.locator("tw-storydata")).toHaveAttribute("format", "SugarCube");
  await expect(page.locator("tw-storydata")).toHaveAttribute("format-version", "2.37.3");
  await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1);
  await expect(page.locator("#journey-header")).toHaveText("oh-my-vllm阅读地图暗色");
  await snapshot(page, "light-reading");
  await mainRoute(page);
  const value = (key) => page.locator("[data-value='" + key + "']");
  await expect(value("computed")).toHaveText("0");
  await page.getByRole("button", { name: "执行下一步", exact: true }).click();
  await expect(value("computed")).toHaveText("32144");
  await expect(value("count")).toHaveText("32144");
  await expect(value("outputs")).toHaveText("0");
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await expect(page.locator(".katex")).toHaveCount(1);
  await snapshot(page, "light-experiment");
  await page.getByRole("button", { name: "执行下一步", exact: true }).click();
  await expect(value("computed")).toHaveText("32768");
  await expect(value("count")).toHaveText("624");
  await expect(value("outputs")).toHaveText("1");
  await expect(page.getByRole("button", { name: "已得到首 token" })).toBeDisabled();
  for (const [length, counts] of [[784, [784]], [785, [784, 1]], [1568, [1568]], [1569, [1568, 1]]]) {
    await page.getByRole("button", { name: String(length), exact: true }).click();
    for (const count of counts) {
      await page.getByRole("button", { name: "执行下一步", exact: true }).click();
      await expect(value("count")).toHaveText(String(count));
    }
    await expect(value("computed")).toHaveText(String(length));
    await expect(value("outputs")).toHaveText("1");
  }
  await page.getByText("查看 aligned_prefill 源码", { exact: true }).click();
  await expect(page.locator("details pre")).toContainText("let block = self.kv.block_size();");
  await expect(page.locator("details .source-caption")).toContainText("crates/scheduler/src/lib.rs:80");
  await link(page, "继续看 worker 怎样返回结果").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Rust 调度与 Python 模型计算的分工");
  await link(page, "我读完了, 返回请求主线").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("执行消息: 调度计划, 缓存地址与 worker 返回值");
  await link(page, "我读完了, 看看 Rust 怎样提交结果").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("提交结果: 更新计算进度与输出历史");
  await link(page, "我读完了, 查看后续阅读大纲").click();
  await expect(page.locator(".outline li")).toHaveCount(12);
  await expect(page.locator(".error")).toHaveCount(0);
  await expect(page.locator(".map-list .read-status").filter({ hasText: "待阅读" })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("light/dark persistence, restart and mobile layout", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "切换到暗色模式" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await snapshot(page, "dark-reading");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await mainRoute(page);
  await page.getByRole("button", { name: "执行下一步", exact: true }).click();
  await snapshot(page, "dark-experiment");
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("分块实验: 32768 个 token 如何分两轮计算");
  await expect(page.locator("[data-value='computed']")).toHaveText("32144");
  await page.setViewportSize({ width: 390, height: 844 });
  await snapshot(page, "mobile-experiment");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "切换到亮色模式" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "重新开始", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("模型推理流程: 从输入文字到输出 token");
  await expect(page.locator("#read-count")).toHaveText("已读 1 篇");
  await expect(page.getByRole("button", { name: "返回", exact: true })).toBeHidden();
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("模型推理流程: 从输入文字到输出 token");
  await expect(page.locator("#read-count")).toHaveText("已读 1 篇");
  await snapshot(page, "mobile-reading");
  const font = await page.evaluate(() => ({
    body: getComputedStyle(document.body).fontFamily,
    loaded: document.fonts.check('500 16px "LXGW WenKai"'),
    code: document.fonts.check('400 14px "Fira Code Nerd Font"'),
  }));
  expect(font.body).toContain("LXGW WenKai");
  expect(font.loaded).toBe(true);
  expect(font.code).toBe(true);
});

test("punctuation, source freshness and corrupt saved state", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.setItem("oh-my-vllm-journey-v1", "null"));
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("模型推理流程: 从输入文字到输出 token");
  const prose = await page.evaluate(() =>
    [...document.querySelectorAll("tw-passagedata")].map((p) => p.textContent).join("\n"));
  expect(prose).not.toMatch(/[，。！？；：、“”‘’（）【】]/u);
  await mainRoute(page);
  const source = await readFile("../crates/scheduler/src/lib.rs");
  const hash = createHash("sha256").update(source).digest("hex");
  const displayedHash = await page.evaluate(() => SugarCube.setup.journeyData.snippets.aligned.sha256);
  expect(displayedHash).toBe(hash);
  await page.getByRole("button", { name: "切换到暗色模式" }).click();
  const colors = await page.evaluate(() => ({
    background: getComputedStyle(document.body).backgroundColor,
    text: getComputedStyle(document.body).color,
  }));
  expect(colors).toEqual({ background: "rgb(22, 29, 35)", text: "rgb(213, 223, 230)" });
  await page.getByRole("button", { name: "重置实验", exact: true }).click();
  await expect(page.locator("[data-value='computed']")).toHaveText("0");
});

test("backward navigation renders monotonic read markers before map macros", async ({ page }) => {
  await page.goto("/");
  await link(page, "阅读地图").click();
  await link(page, "分词: 文字, 词表与 token ID").click();
  await link(page, "我读完了, 接着看位置").click();
  await page.getByRole("button", { name: "返回", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("分词: 文字, 词表与 token ID");
  await page.getByRole("button", { name: "返回", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("阅读地图与教程大纲");
  const row = page.locator(".map-list li").filter({ has: link(page, "分词: 文字, 词表与 token ID") });
  await expect(row.locator(".read-status")).toHaveText("已读");
  await expect(page.locator("#read-count")).toHaveText("已读 2 篇");
});

test("syntax colors, exact source copying and HTTP clipboard fallback", async ({ page, context }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await mainRoute(page);
  const preview = page.locator(".excerpt-preview");
  await expect(preview.locator(".source-line")).toHaveCount(2);
  const data = await page.evaluate(() => SugarCube.setup.journeyData);
  expect(await preview.locator("code").textContent()).toBe(data.alignmentPreview.text);
  // Check all generated excerpts, including generic types and Python indentation.
  const reconstructed = await page.evaluate(() => Object.fromEntries(
    Object.entries(SugarCube.setup.journeyData.snippets).map(([key, source]) => {
      const block = SugarCube.setup.codeBlock(source);
      block.querySelectorAll(".line-number").forEach((number) => number.remove());
      return [key, block.querySelector("code").textContent];
    })));
  for (const [key, source] of Object.entries(data.snippets)) expect(reconstructed[key]).toBe(source.text);
  await preview.getByRole("button", { name: "复制 Rust 代码", exact: true }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(data.alignmentPreview.text);
  await page.getByText("查看 aligned_prefill 源码", { exact: true }).click();
  const source = page.locator("details .code-block");
  await expect(source.locator(".line-number").first()).toHaveText(String(data.snippets.aligned.line));
  await source.getByRole("button", { name: "复制 Rust 代码", exact: true }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(data.snippets.aligned.text);
  await page.evaluate(async () => {
    const clipboard = navigator.clipboard;
    await clipboard.writeText("before HTTP fallback");
    window.readClipboard = () => clipboard.readText();
    Object.defineProperty(navigator, "clipboard", { value: undefined, configurable: true });
  });
  await source.getByRole("button", { name: "复制 Rust 代码", exact: true }).click();
  await expect(source.getByRole("button", { name: "复制 Rust 代码", exact: true })).toHaveText("已复制");
  expect(await page.evaluate(() => window.readClipboard())).toBe(data.snippets.aligned.text);
  await expect(source.getByRole("button", { name: "复制 Rust 代码", exact: true })).toBeFocused();
  await expect(page.locator(".copy-buffer")).toHaveCount(0);
  const keyword = preview.locator(".syntax-token").filter({ hasText: /^let$/ }).first();
  await expect(keyword).toHaveCSS("color", "rgb(189, 41, 59)");
  await snapshot(page, "highlight-rust-light", preview);
  await page.getByRole("button", { name: "切换到暗色模式" }).click();
  await expect(keyword).toHaveCSS("color", "rgb(249, 117, 131)");
  await snapshot(page, "highlight-rust-dark", preview);
  await link(page, "继续看 worker 怎样返回结果").click();
  await link(page, "我读完了, 返回请求主线").click();
  await page.getByText("查看真实 execute_model 的入口", { exact: true }).click();
  const python = page.locator("details").filter({ hasText: "查看真实 execute_model 的入口" });
  await expect(python.locator(".code-language")).toHaveText("Python");
  await expect(python.locator(".syntax-token").filter({ hasText: /^def$/ })).toHaveCSS("color", "rgb(249, 117, 131)");
  await python.getByRole("button", { name: "复制 Python 代码", exact: true }).click();
  expect(await page.evaluate(() => window.readClipboard())).toBe(data.snippets.worker.text);
  for (const theme of ["dark", "light"]) {
    if (theme === "light") await page.getByRole("button", { name: "切换到亮色模式" }).click();
    const contrast = await page.locator(".code-block").evaluateAll((blocks) => {
      const luminance = (color) => color.match(/\d+/g).slice(0, 3).map(Number)
        .map((value) => value / 255).map((value) => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4)
        .reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
      return blocks.flatMap((block) => {
        const background = luminance(getComputedStyle(block).backgroundColor);
        return [...block.querySelectorAll(".syntax-token, .code-language, .code-copy, .line-number")].filter((token) => token.textContent.trim()).map((token) => {
          const toolbar = token.closest(".code-toolbar");
          const surface = toolbar ? luminance(getComputedStyle(toolbar).backgroundColor) : background;
          const foreground = luminance(getComputedStyle(token).color);
          return { ratio: (Math.max(surface, foreground) + .05) / (Math.min(surface, foreground) + .05),
            token: token.className, color: getComputedStyle(token).color,
            background: getComputedStyle(toolbar || block).backgroundColor };
        });
      });
    });
    expect(contrast.length).toBeGreaterThan(30);
    expect(Math.min(...contrast.map((item) => item.ratio)), JSON.stringify(contrast.filter((item) => item.ratio < 4.5))).toBeGreaterThanOrEqual(4.5);
    await snapshot(page, "highlight-python-" + theme, python.locator(".code-block"));
  }
  await page.setViewportSize({ width: 390, height: 844 });
  const pre = python.locator("pre");
  await expect(pre).toBeVisible();
  const size = await pre.evaluate((node) => ({ content: node.scrollWidth, viewport: node.clientWidth }));
  expect(size.content).toBeGreaterThan(size.viewport);
  await pre.focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => pre.evaluate((node) => node.scrollLeft)).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await pre.evaluate((node) => { node.scrollLeft = 0; });
  await snapshot(page, "highlight-python-mobile", python.locator(".code-block"));
  expect(errors).toEqual([]);
});

test("document sizing, clear chapter titles and every displayed field explained", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator("body")).toHaveCSS("font-size", "17px");
  await expect(page.getByRole("heading", { level: 1 })).toHaveCSS("font-size", "28px");
  await link(page, "跟着请求往前走").click();
  await page.getByText("查看 token_ids 的源码与说明", { exact: true }).click();
  await expect(page.locator("pre")).toHaveCSS("font-size", "14px");
  await expect(page.locator("pre")).toContainText("pub token_ids");
  await expect(page.locator("pre")).not.toContainText("pub struct Request");
  await link(page, "我读完了, 接着看位置").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("token ID, 序列位置与计算进度");
  await page.getByText("查看这两个计数字段", { exact: true }).click();
  const counters = page.locator("details").filter({ hasText: "查看这两个计数字段" });
  await expect(counters.locator("pre")).toContainText("pub num_in_flight_tokens");
  await expect(counters.locator("pre")).not.toContainText("pub struct Request");
  await link(page, "我读完了, 看看模型怎样接着写").click();
  await link(page, "我读完了, 继续走选中的路线").click();
  const requestFields = page.getByRole("table", { name: "Request 的 9 个字段", exact: true });
  await expect(requestFields).toBeVisible();
  await expect(requestFields.locator("tbody tr")).toHaveCount(9);
  await expect(requestFields).toContainText("提示词 3 个, 上限 4 个时, 历史最多包含 7 个 token");
  await snapshot(page, "request-fields-light");
  await page.getByRole("button", { name: "切换到暗色模式" }).click();
  await snapshot(page, "request-fields-dark");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator("body")).toHaveCSS("font-size", "16px");
  await expect(page.getByRole("heading", { level: 1 })).toHaveCSS("font-size", "24px");
  await expect(requestFields.getByText("num_in_flight_tokens", { exact: true })).toBeVisible();
  const mobileLabels = await requestFields.locator("tbody tr").first().locator("td").evaluateAll((cells) =>
    cells.slice(1).map((cell) => getComputedStyle(cell, "::before").content));
  expect(mobileLabels[0]).toContain("含义");
  expect(mobileLabels[1]).toContain("用途与例子");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await snapshot(page, "request-fields-mobile");
  // Complete prerequisites once, then check every rendered article from its map link.
  await link(page, "我读完了, 让请求进入队列").click();
  await link(page, "我读完了, 看看缓存的分块单位").click();
  await link(page, "我读完了, 亲手执行两轮调度").click();
  await link(page, "继续看 worker 怎样返回结果").click();
  await link(page, "我读完了, 返回请求主线").click();
  for (const [title, count] of [["ScheduledRequest 的 6 个字段", 6], ["SchedulerOutput 的 5 个字段", 5], ["WorkerOutput 与 RequestOutput 的字段", 5]]) {
    const table = page.getByRole("table", { name: title, exact: true });
    await expect(table).toBeVisible();
    await expect(table.locator("tbody tr")).toHaveCount(count);
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await link(page, "我读完了, 看看 Rust 怎样提交结果").click();
  await link(page, "我读完了, 查看后续阅读大纲").click();
  const lessons = await page.evaluate(() => SugarCube.setup.lessons);
  for (const { title } of Object.values(lessons)) {
    await link(page, title).click();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(title);
    for (const summary of await page.locator(".passage:not(.passage-out) summary").all()) await summary.click();
    await expect(page.locator(".error")).toHaveCount(0);
    const incomplete = await page.locator(".passage:not(.passage-out) details").evaluateAll((details) => details.flatMap((detail) => {
      const source = detail.querySelector("code").cloneNode(true);
      source.querySelectorAll(".line-number").forEach((number) => number.remove());
      const fields = [...source.textContent.matchAll(/^\s*pub\s+(\w+)\s*:/gm)].map((match) => match[1]);
      if (source.textContent.includes("pub struct") && fields.length === 0) return ["No record fields inspected"];
      const guide = detail.querySelector(".field-guide") || detail.previousElementSibling;
      const rows = [...guide.querySelectorAll("tbody tr")];
      return fields.filter((field) => !rows.some((row) => row.cells[0].textContent === field && row.cells[1].textContent.length > 5 && row.cells[2].textContent.length > 5));
    }));
    expect(incomplete, title).toEqual([]);
    if (await page.locator(".passage:not(.passage-out) .primary").count()) {
      const ratio = await page.locator(".passage:not(.passage-out) .primary").evaluate((button) => {
        const luminance = (color) => color.match(/\d+/g).slice(0, 3).map(Number)
          .map((value) => value / 255).map((value) => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4)
          .reduce((sum, value, index) => sum + value * [.2126, .7152, .0722][index], 0);
        const style = getComputedStyle(button);
        const values = [luminance(style.color), luminance(style.backgroundColor)];
        return (Math.max(...values) + .05) / (Math.min(...values) + .05);
      });
      expect(ratio).toBeGreaterThanOrEqual(4.5);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth), title).toBeLessThanOrEqual(390);
    await link(page, "阅读地图").click();
  }
  const notes = await page.evaluate(() => JSON.stringify(SugarCube.setup.journeyData.sourceNotes));
  expect(notes).not.toMatch(/[，。！？；：、“”‘’（）【】]/u);
  expect(errors).toEqual([]);
});
