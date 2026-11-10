import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

// Render the real call components with local data only. No test bridge is shipped in the app.
const fixture = `
const fixtureStore = new Store({ demo: false, phase: "Listening", status: "listening", request: null, connected: true, large: false, unavailable: false, retry: 0, answerDraft: "", secureCount: 0 });
window.fixtureCommands = [];
window.fixtureSet = patch => fixtureStore.update(s => ({ ...s, ...patch }));
function Fixture() {
  const s = useStore(fixtureStore);
  const dispatch = a => { window.fixtureCommands.push(a); if (s.demo) { const next = reduce({ ...appState0(), screen: "Call", agent: true, number: "15889999", ...s }, a); window.fixtureSet(next); } };
  const controls = { ...liveControls0(), elapsedSeconds: 18, endEnabled: true, progressConnected: s.connected, stateUnavailable: s.unavailable };
  return html\`<\${TextSizeContext.Provider} value=\${s.large}><div class=\${"app" + (s.large ? " large-text" : "")}>
    <header class="app-toolbar"><\${TextSizeControl} largeText=\${s.large} onToggle=\${() => window.fixtureSet({ large: !s.large })}/></header>
    <div class="app-viewport">\${s.demo ? html\`<div class="screen grad"><\${CallScreen} state=\${{ ...appState0(), screen: "Call", agent: true, number: "15889999", ...s }} dispatch=\${dispatch}/></div>\`
    : html\`<\${BServerCallScreen} number="15889999" snapshot=\${mkSnap(s.status, "처리 중", { pendingRequest: s.request })}
      onEnd=\${() => dispatch({ type: "End" })} onBack=\${() => {}} controls=\${controls}
      choiceInput=\${{ canSend: s.connected, retryVersion: s.retry, onSelect: dispatch }}
      answerInput=\${{ canSend: s.connected, onAnswer: dispatch, canUseMic: false, onMic() {} }}
      secureInput=\${{ canSend: s.connected, retryVersion: s.retry, onSubmit: dispatch }}/>\`}</div></div><//>\`;
}
render(html\`<\${Fixture}/>\`, document.getElementById("root"));`;

let browser, server, origin;
before(async () => {
  let html = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const entry = 'render(html`<${Root}/>`, document.getElementById("root"));';
  assert(html.includes(entry));
  html = html.replace(entry, fixture);
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
async function open(viewport = { width: 390, height: 844 }, reducedMotion = 'no-preference') {
  const context = await browser.newContext({ viewport, reducedMotion });
  const page = await context.newPage(), calls = [], errors = [];
  await context.route('**/*', route => {
    if (route.request().method() !== 'GET') calls.push(route.request().url());
    return route.request().url().startsWith(origin) && !route.request().url().includes('/v1/') ? route.continue() : route.abort();
  });
  await page.addInitScript(() => {
    window.responsePulses = 0;
    const animate = Element.prototype.animate;
    Element.prototype.animate = function (...args) { window.responsePulses++; return animate.apply(this, args); };
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin + '/app'); await page.evaluate(() => document.fonts.ready);
  await page.locator('.call-number').waitFor({ timeout: 3000 }).catch(async error => { throw new Error(JSON.stringify({ errors, body: await page.locator('body').innerText() }), { cause: error }); });
  return { page, context, calls, errors };
}
const set = (page, value) => page.evaluate(patch => window.fixtureSet(patch), value);
const request = (overrides = {}) => ({ kind: 'clarification', requestId: 'q1', message: '신용·체크카드와 현금카드 중 어떤 카드를 원하세요?', reason: 'ambiguous', expiresAt: new Date(Date.now() + 60000).toISOString(), options: [{ key: '1', label: '신용·체크카드' }, { key: '2', label: '현금카드' }], ...overrides });
const highlighted = page => page.locator('.card.needs-response');

test('진행 중에는 기다림을, 응답 가능할 때는 서버 질문을 한 번만 제목으로 표시한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    assert.equal(await page.getByText('기다려 주세요', { exact: true }).count(), 1);
    await page.locator('.t-title').filter({ hasText: '원하는 메뉴를 찾고 있어요' }).waitFor();
    assert.equal(await highlighted(page).count(), 0);
    const q = request(); await set(page, { status: 'awaiting_user', request: q });
    await highlighted(page).waitFor();
    assert.equal(await page.getByText(q.message, { exact: true }).count(), 1);
    assert.equal(await page.locator('.t-title').filter({ hasText: q.message }).count(), 1);
    assert.equal(await page.getByText('입력이 필요해요', { exact: true }).count(), 0);
    await page.getByRole('button', { name: '신용·체크카드 1번', exact: true }).click();
    await page.getByText('선택을 전달했어요. 응답을 기다려 주세요.', { exact: true }).waitFor();
    assert.equal(await highlighted(page).count(), 0);
    assert.deepEqual(await page.evaluate(() => window.fixtureCommands), [{ requestId: 'q1', key: '1' }]);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('같은 요청 갱신·큰글씨 전환은 강조와 입력을 다시 시작하지 않고 새 요청만 강조한다', async () => {
  const { page, context } = await open();
  try {
    const q = request(); await set(page, { status: 'awaiting_user', request: q }); await highlighted(page).waitFor();
    await page.getByRole('textbox').fill('현금카드');
    await page.waitForFunction(() => window.responsePulses === 1);
    await set(page, { large: true, request: { ...q, expiresAt: new Date(Date.now() + 90000).toISOString() } });
    assert.equal(await page.getByRole('textbox').inputValue(), '현금카드');
    assert.equal(await page.evaluate(() => window.responsePulses), 1);
    await set(page, { request: request({ requestId: 'q2', message: '어떤 업무인가요?' }) });
    await page.waitForFunction(() => window.responsePulses === 2);
    assert.equal(await page.getByRole('textbox').inputValue(), '');
    await set(page, { connected: false });
    assert.equal(await highlighted(page).count(), 0);
    await set(page, { unavailable: true });
    await page.getByText('통화 상태 확인 불가', { exact: true }).waitFor();
    assert.equal(await highlighted(page).count(), 0);
  } finally { await context.close(); }
});

test('중요한 선택은 질문 원문과 버튼만 표시하고 답변 입력을 만들지 않는다', async () => {
  const { page, context } = await open();
  try {
    const q = request({ reason: 'critical_action', message: '입력한 정보가 맞나요?' });
    await set(page, { status: 'awaiting_user', request: q }); await highlighted(page).waitFor();
    assert.equal(await page.getByText(q.message, { exact: true }).count(), 1);
    assert.equal(await page.getByRole('textbox').count(), 0);
    await page.getByText('중요한 선택이에요. 버튼으로 선택해 주세요.', { exact: true }).waitFor();
    await set(page, { status: 'listening' });
    assert.equal(await highlighted(page).count(), 0);
    await set(page, { status: 'ended' });
    assert.equal(await highlighted(page).count(), 0);
  } finally { await context.close(); }
});

test('선택지 없는 질문도 제목을 한 번만 표시하고 답변 전송 후 연결 대기 중에는 강조하지 않는다', async () => {
  const { page, context } = await open();
  try {
    const q = request({ options: [], message: '어느 지역의 날씨가 궁금하세요?' });
    await set(page, { status: 'awaiting_user', request: q }); await highlighted(page).waitFor();
    assert.equal(await page.getByText(q.message, { exact: true }).count(), 1);
    await page.getByText('답변해 주세요', { exact: true }).waitFor();
    await page.getByRole('textbox').fill('수원');
    await page.getByRole('button', { name: '답변 보내기', exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.fixtureCommands), [{ requestId: 'q1', text: '수원' }]);
    await set(page, { connected: false });
    assert.equal(await highlighted(page).count(), 0);
  } finally { await context.close(); }
});

test('기한이 지나면 질문 강조와 선택 가능 상태를 함께 해제한다', async () => {
  const { page, context } = await open();
  try {
    await set(page, { status: 'awaiting_user', request: request({ expiresAt: new Date(Date.now() + 1800).toISOString() }) });
    await highlighted(page).waitFor();
    await page.getByRole('textbox').fill('수원');
    await page.getByText('선택 시간이 지났어요. 새 안내를 기다려 주세요.', { exact: true }).waitFor();
    assert.equal(await highlighted(page).count(), 0);
    assert.equal(await page.getByRole('textbox').inputValue(), '');
    assert.equal(await page.getByRole('button', { name: '현금카드 2번', exact: true }).isEnabled(), false);
  } finally { await context.close(); }
});

test('개인정보 키패드의 항목 제목·카드 밖 배치·입력값은 큰글씨와 기한 갱신 후에도 유지한다', async () => {
  const { page, context, calls } = await open();
  try {
    const q = request({ kind: 'secure_input', label: '주민등록번호 앞 6자리', message: '앞 6자리와 샵을 입력해 주세요.', terminator: '#', options: [] });
    await set(page, { status: 'awaiting_user', request: q });
    await page.getByText('입력해 주세요', { exact: true }).waitFor();
    await page.locator('.t-title').filter({ hasText: q.label }).waitFor();
    assert.equal(await page.locator('.card').count(), 0);
    for (const digit of '123') await page.getByRole('button', { name: '키 ' + digit, exact: true }).click();
    await set(page, { large: true, request: { ...q, expiresAt: new Date(Date.now() + 90000).toISOString() } });
    assert.equal(await page.getByRole('img', { name: '개인정보 3자리 입력됨', exact: true }).count(), 1);
    await page.getByRole('button', { name: '입력 완료', exact: true }).click();
    await page.getByText('입력을 전달했어요. 응답을 기다려 주세요.', { exact: true }).waitFor();
    assert.equal(await page.getByText('입력해 주세요', { exact: true }).count(), 0);
    assert.deepEqual(await page.evaluate(() => window.fixtureCommands), [{ requestId: 'q1', digits: '123' }]);
    assert.deepEqual(calls, []);
  } finally { await context.close(); }
});

test('데모의 질문·직접 입력과 진행 상태도 같은 구분을 적용한다', async () => {
  const { page, context } = await open();
  try {
    await set(page, { demo: true, phase: 'Choice' }); await highlighted(page).waitFor();
    await page.getByRole('button', { name: '현재 날씨 3번', exact: true }).click();
    await page.getByText('기다려 주세요', { exact: true }).waitFor(); assert.equal(await highlighted(page).count(), 0);
    await set(page, { phase: 'Answer', large: true }); await highlighted(page).waitFor();
    await page.getByRole('textbox').fill('수원시'); await set(page, { large: false });
    assert.equal(await page.getByRole('textbox').inputValue(), '수원시');
    await set(page, { phase: 'Secure' }); await page.getByText('입력해 주세요', { exact: true }).waitFor();
    await set(page, { phase: 'Sending' }); assert.equal(await page.getByText('입력해 주세요', { exact: true }).count(), 0);
  } finally { await context.close(); }
});

test('모션 감소 설정에서는 질문 테두리를 유지하되 움직임을 실행하지 않는다', async () => {
  const { page, context } = await open(undefined, 'reduce');
  try {
    await set(page, { status: 'awaiting_user', request: request() }); await highlighted(page).waitFor();
    assert.equal(await page.evaluate(() => window.responsePulses), 0);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 327, height: 688 }, { width: 390, height: 844 }]) {
  test(`긴 서버 질문은 기본·큰글씨에서 스크롤로 읽고 하단 버튼은 유지한다 (${viewport.width}px)`, async () => {
    const { page, context, calls, errors } = await open(viewport);
    try {
      const end = page.getByRole('button', { name: '종료', exact: true }); const before = await end.boundingBox();
      const q = request({ message: '상담원 연결을 원하시나요? 또는 혜택을 확인하고 카드 발급을 신청할 수 있는 모바일 홈페이지 주소를 문자로 받으시겠어요?' });
      await set(page, { status: 'awaiting_user', request: q }); await highlighted(page).waitFor();
      await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
      for (const large of [false, true]) {
        await set(page, { large }); const after = await end.boundingBox();
        assert(Math.abs(before.y - after.y) < 3); assert(after.y + after.height <= viewport.height);
        const title = page.getByText(q.message, { exact: true }); await title.scrollIntoViewIfNeeded();
        assert.equal(await title.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
        assert.equal(await title.evaluate(el => parseFloat(getComputedStyle(el).fontSize)), large ? 25 : 20);
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/response-${viewport.width}-${large ? 'large' : 'default'}.png`, import.meta.url)) });
      }
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}
