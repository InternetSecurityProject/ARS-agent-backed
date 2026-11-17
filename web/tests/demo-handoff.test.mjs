import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

// Use the real demo model (including its timers), never the live-call controller.
const fixture = `
window.demoSet = patch => phoneModel.store.update(s => ({ ...appState0(), screen: "Call", number: "15889999", agent: true, phase: "Listening", goal: "카드 발급", ...patch }));
window.demoRead = () => phoneModel.store.value;
window.demoDispatch = action => phoneModel.dispatch(action);
window.demoSet({});
function DemoFixture() {
  const s = useStore(phoneModel.store);
  const [large, setLarge] = useState(false);
  return html\`<\${TextSizeContext.Provider} value=\${large}><div class=\${"app" + (large ? " large-text" : "")}>
    <header class="app-toolbar"><\${TextSizeControl} largeText=\${large} onToggle=\${() => setLarge(v => !v)}/></header>
    <div class="app-viewport"><\${ArsApp} state=\${s} onAction=\${phoneModel.dispatch}/></div></div><//>\`;
}
render(html\`<\${DemoFixture}/>\`, document.getElementById("root"));`;
let browser, server, origin;
before(async () => {
  let html = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  html = html.replace('render(html`<${Root}/>`, document.getElementById("root"));', fixture);
  const fonts = new Map();
  for (const [path, file, family, weight] of [
    ['/noto.ttf', process.env.WEB_TEST_NOTO_FONT, 'Noto Sans KR', '100 900'],
    ['/gowun.ttf', process.env.WEB_TEST_GOWUN_FONT, 'Gowun Batang', '400'],
  ]) {
    if (!file) continue;
    fonts.set(path, await readFile(file));
    html = html.replace('</head>', `<style>@font-face{font-family:'${family}';src:url('${path}');font-weight:${weight};font-display:block}</style></head>`);
  }
  server = createServer((req, res) => {
    if (req.method === 'GET' && fonts.has(req.url)) { res.writeHead(200, { 'Content-Type': 'font/ttf' }); return res.end(fonts.get(req.url)); }
    if (req.method === 'GET' && req.url === '/app') { res.writeHead(200, { 'Content-Type': 'text/html;charset=utf-8' }); return res.end(html); }
    res.writeHead(403); res.end();
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
});
after(async () => { await browser?.close(); await new Promise(r => server?.close(r)); });
async function open(width = 390, height = 844) {
  const context = await browser.newContext({ viewport: { width, height } });
  const page = await context.newPage(), calls = [], errors = [];
  await context.route('**/*', route => {
    const req = route.request();
    if (req.method() !== 'GET' || req.url().includes('/v1/')) calls.push(req.url());
    return req.url().startsWith(origin) && !req.url().includes('/v1/') ? route.continue() : route.abort();
  });
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(origin + '/app'); await page.evaluate(() => document.fonts.ready);
  return { page, context, calls, errors };
}
const set = (page, patch) => page.evaluate(p => window.demoSet(p), patch);
const read = page => page.evaluate(() => window.demoRead());

for (const [width, height, large] of [[320, 568, true], [390, 844, false], [390, 844, true]]) {
  test(`상담원 버튼은 네 번째 60px 버튼으로 겹치지 않고 입력은 큰글씨를 유지한다 (${width}, ${large})`, async () => {
    const { page, context, calls, errors } = await open(width, height);
    try {
      if (large) await page.getByRole('button', { name: '큰글씨 보기', exact: true }).click();
      await page.getByRole('button', { name: '상담원 연결', exact: true }).waitFor({ timeout: 2000 });
      const buttons = page.locator('.agent-controls > button');
      assert.equal(await buttons.count(), 4);
      const boxes = await buttons.locator('.round').evaluateAll(els => els.map(el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, width: r.width, right: r.right }; }));
      for (let i = 0; i < boxes.length; i++) {
        assert.equal(boxes[i].width, 60);
        assert(boxes[i].x >= 0 && boxes[i].right <= width);
        if (i) { assert(boxes[i].x >= boxes[i - 1].right); assert.equal(boxes[i].y, boxes[0].y); }
      }
      const endBox = await page.getByRole('button', { name: '종료', exact: true }).boundingBox();
      const controlsBox = await page.locator('.agent-controls').boundingBox();
      assert(endBox.y >= controlsBox.y + controlsBox.height - 1, '종료 버튼과 네 라벨은 겹치면 안 된다');
      await page.getByRole('button', { name: '상담원 연결', exact: true }).click();
      await page.getByText('상담원 연결을 위해선 원하는 목적을 말씀해주세요', { exact: true }).waitFor();
      const input = page.getByRole('textbox');
      assert.equal(await input.inputValue(), '카드 발급');
      assert.equal(await page.locator('.app.large-text').count(), large ? 1 : 0);
      await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/handoff-purpose-${width}-${large}.png`, import.meta.url)) });
      await input.fill('   ');
      assert(await page.getByRole('button', { name: '상담원 연결 요청', exact: true }).isDisabled());
      await input.fill('카드 발급 문의');
      await page.waitForTimeout(3300);
      assert.equal((await read(page)).handoffRequest.draft, '카드 발급 문의', '타이머가 입력 중 화면을 넘기면 안 된다');
      await page.getByRole('button', { name: '상담원 연결 요청', exact: true }).click();
      await page.getByText('상담원 연결 메뉴를 찾고 있어요', { exact: true }).waitFor();
      assert.equal((await read(page)).handoffGoal, '카드 발급 문의');
      assert.equal((await read(page)).goal, '카드 발급', '원래 업무 목적은 보존한다');
      assert.equal(await page.getByText('수원', { exact: true }).count(), 0);
      await page.getByText('상담원 연결을 기다리고 있어요', { exact: true }).waitFor();
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('입력 취소·뒤로가기는 기존 단계를 복원하고 종료는 임시 목적을 폐기한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { phase: 'Choice', selectedKey: '3' });
    await page.getByRole('button', { name: '상담원 연결', exact: true }).click();
    await page.getByRole('textbox').fill('계좌 문의');
    await page.getByRole('button', { name: '돌아가기', exact: true }).click();
    await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor();
    assert.equal((await read(page)).handoffRequest, null);
    assert.equal((await read(page)).selectedKey, '3');
    await page.getByRole('button', { name: '상담원 연결', exact: true }).click();
    await page.evaluate(() => window.demoDispatch({ type: 'Back' }));
    assert.equal((await read(page)).phase, 'Choice');
    assert.equal((await read(page)).handoffRequest, null);
    await page.getByRole('button', { name: '상담원 연결', exact: true }).click();
    await page.getByRole('button', { name: '종료', exact: true }).click();
    await page.getByRole('button', { name: '키패드로 돌아가기', exact: true }).waitFor();
    assert.equal((await read(page)).handoffRequest, null);
    assert.equal((await read(page)).handoffGoal, '');
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('연결 중·개인정보 입력·상담원 대기는 상담원 목적을 다시 요청하지 않는다', async () => {
  const { page, context } = await open();
  try {
    for (const phase of ['Dialing', 'WaitingAgent']) {
      await set(page, { phase });
      assert(await page.getByRole('button', { name: '상담원 연결', exact: true }).isDisabled());
    }
    for (const phase of ['Secure', 'Sending']) {
      await set(page, { phase });
      assert.equal(await page.getByRole('button', { name: '상담원 연결', exact: true }).count(), 0);
      await page.evaluate(() => window.demoDispatch({ type: 'RequestHandoff' }));
      assert.equal((await read(page)).handoffRequest, null);
    }
  } finally { await context.close(); }
});
