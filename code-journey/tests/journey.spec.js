import { test, expect } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";

import { createHash } from "node:crypto";

const evidence = process.env.JOURNEY_QA_DIR || "/tmp/oh-my-vllm-journey-qa";
const link = (page, name) => page.getByRole("link", { name, exact: true });
async function snapshot(page, name, codeBlock) {
  await mkdir(evidence, { recursive: true });
  await page.locator("#passages h1").waitFor();
  await page.evaluate(async () => {
    await document.fonts.load('500 28px "LXGW WenKai"');
    await document.fonts.load('400 18px "Fira Code Nerd Font"');
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
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("文字怎样变成 token?");
  await expect(page.getByRole("heading", { level: 1 })).toBeFocused();
  for (const text of [
    "我读完了, 接着看位置", "我读完了, 看看模型怎样接着写",
    "我读完了, 继续走选中的路线", "我读完了, 让请求进入队列",
    "我读完了, 看看缓存的分块单位", "我读完了, 亲手执行两轮调度",
  ]) await link(page, text).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("预算为什么留下一截?");
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
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("谁决定, 谁计算?");
  await link(page, "我读完了, 返回请求主线").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("worker 怎样返回结果?");
  await link(page, "我读完了, 看看 Rust 怎样提交结果").click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("第一个 token 回到了哪里?");
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
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("预算为什么留下一截?");
  await expect(page.locator("[data-value='computed']")).toHaveText("32144");
  await page.setViewportSize({ width: 390, height: 844 });
  await snapshot(page, "mobile-experiment");
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "切换到亮色模式" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "重新开始", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("从一条请求开始");
  await expect(page.locator("#read-count")).toHaveText("已读 1 篇");
  await expect(page.getByRole("button", { name: "返回", exact: true })).toBeHidden();
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("从一条请求开始");
  await expect(page.locator("#read-count")).toHaveText("已读 1 篇");
  await snapshot(page, "mobile-reading");
  const font = await page.evaluate(() => ({
    body: getComputedStyle(document.body).fontFamily,
    loaded: document.fonts.check('500 24px "LXGW WenKai"'),
    code: document.fonts.check('400 18px "Fira Code Nerd Font"'),
  }));
  expect(font.body).toContain("LXGW WenKai");
  expect(font.loaded).toBe(true);
  expect(font.code).toBe(true);
});

test("punctuation, source freshness and corrupt saved state", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.setItem("oh-my-vllm-journey-v1", "null"));
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("从一条请求开始");
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
  await link(page, "文字怎样变成 token?").click();
  await link(page, "我读完了, 接着看位置").click();
  await page.getByRole("button", { name: "返回", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("文字怎样变成 token?");
  await page.getByRole("button", { name: "返回", exact: true }).click();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("阅读地图");
  const row = page.locator(".map-list li").filter({ has: link(page, "文字怎样变成 token?") });
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
