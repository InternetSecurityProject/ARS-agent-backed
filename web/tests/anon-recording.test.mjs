import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// 통화가 끝난 뒤의 '통화 기록' 묶음(통화 기록 받기·서버 기록 지우기)은 익명 모드(시연용)에서 표시하지 않음.
// 통화 기록 남기기를 켠 끝난 통화를 로컬 상태로 렌더 (서버 API 는 부르지 않음).
const fixture = `
window.renderEnded = (anon) => {
  Prefs.set("anonymous", anon ? "1" : null);
  const server = { ...serverState0(), hasSession: true, recordRequested: true, recording: null,
    snapshot: mkSnap("ended", "통화가 끝났어요.") };
  const root = document.getElementById("root");
  render(null, root);
  render(html\`<div class="app"><\${LiveSessionExtras} server=\${server} voice=\${null} audio=\${null} compactInput=\${false}/></div>\`, root);
};`;

let browser, server, origin;
before(async () => {
  let html = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const entry = 'render(html`<${Root}/>`, document.getElementById("root"));';
  assert(html.includes(entry));
  html = html.replace(entry, fixture);
  server = createServer((req, res) => { res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(html); });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
});
after(async () => { await browser?.close(); server?.close(); });

async function shown(anon) {
  const page = await browser.newPage();
  await page.goto(origin + '/app');
  await page.waitForFunction(() => typeof window.renderEnded === 'function');
  await page.evaluate((a) => window.renderEnded(a), anon);
  const text = await page.locator('#root').innerText();
  await page.close();
  return text;
}

test('익명 모드가 꺼져 있으면 통화 기록 받기·서버 기록 지우기가 보인다', async () => {
  const text = await shown(false);
  assert.match(text, /통화 기록 받기/);
  assert.match(text, /서버 기록 지우기/);
});

test('익명 모드(시연용)에서는 통화 기록 묶음이 보이지 않는다', async () => {
  const text = await shown(true);
  assert.doesNotMatch(text, /통화 기록 받기/);
  assert.doesNotMatch(text, /서버 기록 지우기/);
  assert.doesNotMatch(text, /recordings 폴더/);
});
