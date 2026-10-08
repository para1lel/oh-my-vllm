import { test, expect } from '@playwright/test';
import { mkdir, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

const evidence = process.env.JOURNEY_QA_DIR || '/tmp/oh-my-vllm-journey-qa';
const route = (page, name) => page.getByRole('link', { name, exact: true });
const heading = (page) => page.getByRole('heading', { level: 1 });
async function snapshot(page, name, region) {
  await mkdir(evidence, { recursive: true });
  await page.evaluate(async () => {
    await document.fonts.load('500 17px "LXGW WenKai"');
    await document.fonts.load('400 14px "Fira Code Nerd Font"');
    await document.fonts.ready;
  });
  if (region) await region.screenshot({ path: evidence + '/' + name + '.png', animations: 'disabled' });
  else {
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: evidence + '/' + name + '.png', animations: 'disabled' });
  }
}
async function openChapter(page, target) {
  await page.goto('/');
  await page.locator('#journey-map').click();
  const chapters = await page.evaluate(() => SugarCube.setup.lessons);
  await route(page, chapters[target].title).click();
  while (await page.evaluate(() => SugarCube.State.passage) !== target) {
    await page.locator('.chapter-choices a').first().click();
  }
  return chapters;
}
function monitor(page) {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (['error', 'warning'].includes(message.type())) errors.push(message.text()); });
  return errors;
}

test('Twine prerequisite routes retain interest and complete all twelve chapters', async ({ page }) => {
  const errors = monitor(page);
  await page.goto('/');
  await expect(page).toHaveTitle('oh-my-vllm 代码之旅');
  await expect(page.locator('tw-storydata')).toHaveAttribute('format', 'SugarCube');
  await expect(page.locator('tw-storydata')).toHaveAttribute('format-version', '2.37.3');
  await snapshot(page, 'twelve-chapters-home-light');
  await route(page, 'Qwen 模型计算').click();
  await expect(heading(page)).toContainText('第 1 章. 输入与生成');
  await expect(heading(page)).toBeFocused();
  expect(await page.evaluate(() => SugarCube.State.variables.interest)).toBe('gpu');
  const chapters = await page.evaluate(() => SugarCube.setup.lessons);
  for (const [id, chapter] of Object.entries(chapters)) {
    await expect(heading(page)).toHaveText('第 ' + chapter.number + ' 章. ' + chapter.title);
    await expect(page.locator('.source-coverage li')).not.toHaveCount(0);
    await expect(page.locator('.error, .katex-error')).toHaveCount(0);
    if (await page.locator('article code').count()) {
      await expect(page.locator('article code').first()).toHaveCSS('font-family', '"Fira Code Nerd Font", "LXGW WenKai", monospace');
    }
    const links = await page.locator('article a, .reading-tools a').evaluateAll((links) => links.map((link) => ({
      list: link.closest('ul')?.className, condition: link.closest('li')?.textContent,
    })));
    expect(links.length).toBeGreaterThan(2);
    for (const item of links) {
      expect(item.list).toContain('navigation-list');
      expect(item.condition).toMatch(/^如果/);
    }
    await page.locator('.chapter-choices a').first().click();
  }
  await expect(heading(page)).toContainText('12 章阅读地图');
  await expect(page.locator('.map-list li')).toHaveCount(12);
  await expect(page.locator('.map-list .read-status').filter({ hasText: '已读' })).toHaveCount(12);
  await expect(page.locator('#read-count')).toHaveText('已读 12 / 12 章');
  await snapshot(page, 'twelve-chapters-map-light');
  await page.reload();
  // Last chapter is restored; a fresh map visit must retain every marker.
  await page.locator('#journey-map').click();
  await expect(page.locator('.map-list .read-status').filter({ hasText: '已读' })).toHaveCount(12);
  expect(errors).toEqual([]);
});

test('actual scheduler boundary traces, refresh and theme persistence', async ({ page }) => {
  const errors = monitor(page);
  await openChapter(page, 'Budget');
  const value = (key) => page.locator('[data-value="' + key + '"]');
  await expect(value('computed')).toHaveText('0');
  await page.getByRole('button', { name: '执行下一步', exact: true }).click();
  await expect(value('computed')).toHaveText('32144');
  await expect(value('outputs')).toHaveText('0');
  await snapshot(page, 'twelve-chapters-budget-light', page.locator('.lab'));
  await page.getByRole('button', { name: '切换到暗色模式' }).click();
  await page.reload();
  await expect(heading(page)).toContainText('第 4 章');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(value('computed')).toHaveText('32144');
  await page.getByRole('button', { name: '执行下一步', exact: true }).click();
  await expect(value('count')).toHaveText('624');
  await expect(value('computed')).toHaveText('32768');
  await expect(value('outputs')).toHaveText('1');
  await expect(page.getByRole('button', { name: '已得到首 token' })).toBeDisabled();
  for (const [length, counts] of [[784, [784]], [785, [784, 1]], [1568, [1568]], [1569, [1568, 1]]]) {
    await page.getByRole('button', { name: String(length), exact: true }).click();
    for (const count of counts) {
      await page.getByRole('button', { name: '执行下一步', exact: true }).click();
      await expect(value('count')).toHaveText(String(count));
    }
    await expect(value('computed')).toHaveText(String(length));
    await expect(value('outputs')).toHaveText('1');
  }
  await page.getByRole('button', { name: '重置实验' }).click();
  await expect(value('computed')).toHaveText('0');
  await snapshot(page, 'twelve-chapters-budget-dark', page.locator('.lab'));
  expect(errors).toEqual([]);
});

test('normalized source display, dual-theme contrast and both clipboard paths', async ({ page, context }) => {
  const errors = monitor(page);
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await openChapter(page, 'Budget');
  const data = await page.evaluate(() => SugarCube.setup.journeyData);
  const reconstructed = await page.evaluate(() => Object.fromEntries(Object.entries(SugarCube.setup.journeyData.snippets).map(([key, source]) => {
    const block = SugarCube.setup.codeBlock(source); block.querySelectorAll('.line-number').forEach((line) => line.remove());
    return [key, block.querySelector('code').textContent];
  })));
  for (const [key, source] of Object.entries(data.snippets)) {
    expect(reconstructed[key]).toBe(source.text);
    const file = await readFile('../' + source.path, 'utf8');
    expect(source.sha256).toBe(createHash('sha256').update(file).digest('hex'));
    expect(file.split('\n').slice(source.line - 1, source.line - 1 + source.originalText.split('\n').length).join('\n')).toBe(source.originalText);
    expect(source.text.split('\n').some((line) => line.trim() && !/^\s/.test(line)), key).toBe(true);
  }
  await page.getByText('展开 aligned_prefill 的完整源码与说明', { exact: true }).click();
  const block = page.locator('details').filter({ hasText: '展开 aligned_prefill' }).locator('.code-block');
  await block.getByRole('button', { name: '复制 Rust 代码' }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(data.snippets.aligned.text);
  await page.evaluate(() => {
    const clipboard = navigator.clipboard; window.readClipboard = () => clipboard.readText();
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
  });
  await block.getByRole('button', { name: '复制 Rust 代码' }).click();
  expect(await page.evaluate(() => window.readClipboard())).toBe(data.snippets.aligned.text);
  await expect(block.getByRole('button', { name: '复制 Rust 代码' })).toBeFocused();
  await expect(page.locator('.copy-buffer')).toHaveCount(0);
  for (const theme of ['light', 'dark']) {
    if (theme === 'dark') await page.getByRole('button', { name: '切换到暗色模式' }).click();
    const contrast = await block.evaluate((block) => {
      const luminance = (color) => color.match(/\d+/g).slice(0, 3).map(Number).map((x) => x / 255)
        .map((x) => x <= .04045 ? x / 12.92 : ((x + .055) / 1.055) ** 2.4)
        .reduce((sum, x, i) => sum + x * [.2126, .7152, .0722][i], 0);
      return [...block.querySelectorAll('.syntax-token, .code-language, .code-copy, .line-number')].filter((t) => t.textContent.trim()).map((t) => {
        const bg = luminance(getComputedStyle(t.closest('.code-toolbar') || block).backgroundColor);
        const fg = luminance(getComputedStyle(t).color); return (Math.max(bg, fg) + .05) / (Math.min(bg, fg) + .05);
      });
    });
    expect(Math.min(...contrast)).toBeGreaterThanOrEqual(4.5);
    await snapshot(page, 'twelve-chapters-dedented-rust-' + theme, block);
  }
  await openChapter(page, 'Ownership');
  await page.getByText('展开 execute_model 的入口和状态说明', { exact: true }).click();
  const python = page.locator('details').filter({ hasText: '展开 execute_model' }).locator('.code-block');
  await expect(python.getByRole('button', { name: '复制 Python 代码' })).toHaveCSS('font-family', '"LXGW WenKai", serif');
  await expect(python.locator('.code-language')).toHaveCSS('font-style', 'normal');
  await expect(page.locator('article > p code').filter({ hasText: 'num_gpu_blocks // 3' })).toHaveText('num_gpu_blocks // 3');
  await python.getByRole('button', { name: '复制 Python 代码' }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(data.snippets.worker.text);
  await snapshot(page, 'twelve-chapters-dedented-python-dark', python);
  await page.setViewportSize({ width: 390, height: 844 });
  await python.locator('pre').focus();
  await page.keyboard.press('ArrowRight');
  await expect.poll(() => python.locator('pre').evaluate((node) => node.scrollLeft)).toBeGreaterThan(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  expect(errors).toEqual([]);
});

test('every chapter, displayed field and formula works at mobile document sizes', async ({ page }) => {
  test.setTimeout(90000);
  const errors = monitor(page);
  const chapters = await openChapter(page, 'Validation');
  await expect(page.locator('body')).toHaveCSS('font-size', '17px');
  await expect(heading(page)).toHaveCSS('font-size', '28px');
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.locator('body')).toHaveCSS('font-size', '16px');
  await expect(heading(page)).toHaveCSS('font-size', '24px');
  await page.locator('.chapter-choices a').last().click();
  for (const [id, chapter] of Object.entries(chapters)) {
    await route(page, chapter.title).click();
    await expect(heading(page)).toHaveText('第 ' + chapter.number + ' 章. ' + chapter.title);
    for (const summary of await page.locator('article summary').all()) await summary.click();
    await expect(page.locator('.error, .katex-error')).toHaveCount(0);
    if (await page.locator('article pre').count()) await expect(page.locator('article pre').first()).toHaveCSS('font-size', '13px');
    const missing = await page.locator('article details').evaluateAll((details) => details.flatMap((detail) => {
      const source = detail.querySelector('pre code').cloneNode(true); source.querySelectorAll('.line-number').forEach((line) => line.remove());
      const text = source.textContent;
      const fields = [...text.matchAll(/^class /m.test(text) ? /^    (\w+): /gm : /^\s*pub\s+(\w+)\s*:/gm)].map((match) => match[1]);
      const guide = detail.querySelector('.field-guide') || detail.previousElementSibling;
      return fields.filter((field) => ![...guide.querySelectorAll('tbody tr')].some((row) => row.cells[0].textContent === field && row.cells[1].textContent && row.cells[2].textContent));
    }));
    expect(missing, id).toEqual([]);
    await expect(page.locator('.syntax-guide')).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth), id).toBeLessThanOrEqual(390);
    if (id === 'Requests') {
      await expect(page.getByRole('table', { name: 'Request 的 9 个字段', exact: true }).locator('tbody tr')).toHaveCount(9);
      await snapshot(page, 'twelve-chapters-fields-mobile', page.locator('.field-guide').first());
    }
    if (id === 'Model') await snapshot(page, 'twelve-chapters-model-mobile');
    await page.locator('#journey-map').click();
  }
  const prose = await page.evaluate(() => [...document.querySelectorAll('tw-passagedata')].map((p) => p.textContent).join('\n') + JSON.stringify(SugarCube.setup.journeyData.sourceNotes));
  expect(prose).not.toMatch(/[，。！？；：、“”‘’（）【】]/u);
  expect(prose).not.toMatch(/pub 表示|struct 定义|fn 定义|def 定义|缩进表示|收集位置与命名参数|u64 是 64 位/);
  const fonts = await page.evaluate(() => ({ prose: document.fonts.check('500 16px "LXGW WenKai"'), code: document.fonts.check('400 13px "Fira Code Nerd Font"') }));
  expect(fonts).toEqual({ prose: true, code: true });
  expect(errors).toEqual([]);
});

test('field documentation is rendered and copied with no language primer', async ({ page, context }) => {
  const errors = monitor(page);
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await openChapter(page, 'Basics');
  const expected = '/// Prompt tokens plus all accepted output tokens so far.\n///\n/// Does NOT include unverified MTP draft tokens; those live in\n/// `draft_token_ids` and are concatenated by `num_tokens_with_spec()`.\npub token_ids: Vec<u32>,';
  await page.getByText('展开 token_ids 的源码与说明', { exact: true }).click();
  const block = page.locator('details').filter({ hasText: '展开 token_ids' }).locator('.code-block');
  const source = await page.evaluate(() => SugarCube.setup.journeyData.snippets.tokenHistory);
  expect(source.text).toBe(expected);
  expect(source.line).toBe(27);
  const shown = await block.locator('code').evaluate((node) => {
    const copy = node.cloneNode(true); copy.querySelectorAll('.line-number').forEach((line) => line.remove()); return copy.textContent;
  });
  expect(shown).toBe(expected);
  await block.getByRole('button', { name: '复制 Rust 代码' }).click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(expected);
  await expect(page.locator('.syntax-guide')).toHaveCount(0);
  await snapshot(page, 'documented-token-history-light', block);
  await page.getByRole('button', { name: '切换到暗色模式' }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await snapshot(page, 'documented-token-history-mobile-dark', block);
  await expect(page.locator('.error, .katex-error')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('complete source coverage links and highlighted full-file line anchors', async ({ page, request }) => {
  test.setTimeout(90000);
  const errors = monitor(page);
  await openChapter(page, 'Model');
  const records = await page.evaluate(() => SugarCube.setup.journeyData.coverage);
  for (const record of records) {
    const source = await readFile('../' + record.path, 'utf8');
    expect(record.sha256).toBe(createHash('sha256').update(source).digest('hex'));
    const response = await request.get('/' + record.url);
    expect(response.ok(), record.path).toBe(true);
    expect(await response.text()).toContain('id="L' + record.line + '"');
  }
  for (const path of ['crates/scheduler/src/lib.rs', 'python/oh_my_vllm/models/qwen.py', 'python/oh_my_vllm/kernels/cuda_backend/kernels.cu']) {
    const record = records.find((item) => item.path === path);
    await page.goto('/' + record.url);
    await expect(heading(page)).toHaveText('完整源码');
    await expect(page.locator('#L' + record.line)).toBeVisible();
    expect(await page.locator('pre code').evaluate((node) => {
      const copy = node.cloneNode(true); copy.querySelectorAll('.line-number').forEach((line) => line.remove()); return copy.textContent;
    })).toBe(await readFile('../' + path, 'utf8'));
    if (path.endsWith('qwen.py')) {
      await snapshot(page, 'twelve-chapters-full-source-light');
      await page.getByRole('button', { name: '切换亮色 / 暗色' }).click();
      await snapshot(page, 'twelve-chapters-full-source-dark');
      await page.setViewportSize({ width: 390, height: 844 });
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
      await page.setViewportSize({ width: 1536, height: 1024 });
      await page.locator('.navigation-list a').first().click();
      await expect(heading(page)).toContainText('第 8 章');
    }
  }
  expect(errors).toEqual([]);
});

test('map history stays monotonic; old short passages migrate to visited', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => {
    localStorage.removeItem('oh-my-vllm-journey-v2');
    localStorage.setItem('oh-my-vllm-journey-v1', JSON.stringify({ read: ['Start', 'Token', 'Position'], visited: ['Token', 'Position', 'Experiment'], last: 'Experiment', goal: 'Worker', interest: 'gpu', expLength: 32768, expIndex: 0 }));
  });
  await page.reload();
  await expect(heading(page)).toContainText('第 1 章');
  await expect(page.locator('#read-count')).toHaveText('已读 0 / 12 章');
  await page.locator('#journey-map').click();
  await expect(page.locator('.map-list .read-status').filter({ hasText: '阅读中' })).toHaveCount(2);
  await page.locator('.map-list a').first().click();
  await page.locator('.chapter-choices a').first().click();
  await page.getByRole('button', { name: '返回', exact: true }).click();
  await page.getByRole('button', { name: '返回', exact: true }).click();
  await expect(heading(page)).toContainText('12 章阅读地图');
  await expect(page.locator('.map-list li').first().locator('.read-status')).toHaveText('已读');
  await expect(page.locator('#read-count')).toHaveText('已读 1 / 12 章');
  await page.getByRole('button', { name: '重新开始', exact: true }).click();
  await expect(heading(page)).toHaveText('模型推理流程: 从输入文字到输出 token');
  await expect(page.locator('#read-count')).toHaveText('已读 0 / 12 章');
  await expect(page.getByRole('button', { name: '返回', exact: true })).toBeHidden();
  await page.evaluate(() => localStorage.setItem('oh-my-vllm-journey-v2', 'null'));
  await page.reload();
  await expect(heading(page)).toHaveText('模型推理流程: 从输入文字到输出 token');
});

test('review and map choices preserve an unfinished chapter', async ({ page }) => {
  const chapters = await openChapter(page, 'Requests');
  await page.locator('.chapter-choices li').filter({ hasText: '如果想复习' }).getByRole('link').click();
  await expect(heading(page)).toContainText('第 2 章');
  await page.locator('#journey-map').click();
  await expect(page.locator('.map-list li').filter({ has: route(page, chapters.Requests.title) }).locator('.read-status')).toHaveText('阅读中');
  await route(page, chapters.Requests.title).click();
  await page.locator('.chapter-choices li').filter({ hasText: '如果想选择其他主题' }).getByRole('link').click();
  await expect(page.locator('.map-list li').filter({ has: route(page, chapters.Requests.title) }).locator('.read-status')).toHaveText('阅读中');
  await route(page, chapters.Requests.title).click();
  await page.locator('.chapter-choices li').filter({ hasText: '如果已理解本章' }).first().getByRole('link').click();
  await page.locator('#journey-map').click();
  await expect(page.locator('.map-list li').filter({ has: route(page, chapters.Requests.title) }).locator('.read-status')).toHaveText('已读');
});
