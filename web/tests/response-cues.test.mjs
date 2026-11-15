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
    : html\`<\${BServerCallScreen} number="15889999" result=\${s.result ?? null} snapshot=\${mkSnap(s.status, "처리 중", { pendingRequest: s.request, path: s.path ?? [], captions: s.captions ?? [], summary: s.summary ?? null })}
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

test('서버 통화에는 데모 전용 상담원 연결 버튼을 추가하지 않는다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    assert.equal(await page.locator('.agent-controls > button').count(), 3);
    assert.equal(await page.getByRole('button', { name: '상담원 연결', exact: true }).count(), 0);
    await set(page, { status: 'awaiting_user', request: request() });
    await highlighted(page).waitFor();
    assert.equal(await page.getByRole('button', { name: '상담원 연결', exact: true }).count(), 0);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }]) {
  for (const demo of [false, true]) {
    test(`큰글씨 종료 화면은 번호 아래에 종료 상태·시간을 표시하고 확대를 유지한다 (${viewport.width}px, ${demo ? '데모' : '서버'})`, async () => {
      const { page, context, calls, errors } = await open(viewport);
      try {
        await set(page, demo
          ? { demo: true, screen: 'Ended', elapsedSeconds: 39 }
          : { status: 'ended', result: { path: [], summary: null, keyInfo: [], handoffScript: null, endReason: 'user_ended', durationSec: 39 } });
        const detail = page.getByText('통화 종료 · 00:39', { exact: true });
        await detail.waitFor();
        const base = await detail.evaluate(el => parseFloat(getComputedStyle(el).fontSize));
        await set(page, { large: true });
        assert((await detail.evaluate(el => parseFloat(getComputedStyle(el).fontSize))) >= base * 1.2, '종료 상태·시간의 큰글씨 크기는 유지해야 한다');
        assert.equal(await page.locator('.call-number').count(), 1);
        const numberBox = await page.locator('.call-number').boundingBox();
        const detailBox = await detail.boundingBox();
        assert(detailBox.y >= numberBox.y + numberBox.height, '종료 상태·시간은 번호 옆이 아닌 아래에 있어야 한다');
        assert(Math.abs(detailBox.x + detailBox.width / 2 - numberBox.x - numberBox.width / 2) <= 1, '번호와 종료 정보의 가로 중심을 맞춰야 한다');
        const back = await page.getByRole('button', { name: '키패드로 돌아가기', exact: true }).boundingBox();
        assert(back.y >= 0 && back.y + back.height <= viewport.height, '돌아가기 버튼은 화면 안에 있어야 한다');
        await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/ended-${viewport.width}-${demo ? 'demo' : 'server'}-large.png`, import.meta.url)) });
        assert.deepEqual(calls, []); assert.deepEqual(errors, []);
      } finally { await context.close(); }
    });
  }
}

test('진행 중에는 대기 배지를 생략하고 응답 가능할 때만 질문을 강조한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    assert.equal(await page.getByText('기다려 주세요', { exact: true }).count(), 0);
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
    const label = page.locator('.option .t-bodym').first();
    assert.equal(await label.evaluate(el => parseFloat(getComputedStyle(el).fontSize)), 16);
    await set(page, { large: true });
    assert.equal(await label.evaluate(el => parseFloat(getComputedStyle(el).fontSize)), 20);
    await page.getByRole('button', { name: '현재 날씨 3번', exact: true }).click();
    await page.getByText('최근 선택한 메뉴', { exact: true }).waitFor();
    assert.equal(await page.getByText('기다려 주세요', { exact: true }).count(), 0); assert.equal(await highlighted(page).count(), 0);
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

for (const viewport of [{ width: 327, height: 688 }, { width: 390, height: 844 }]) {
  test(`후보는 16px·큰글씨 20px로 읽고 긴 메뉴와 별표·샵도 선택할 수 있다 (${viewport.width}px)`, async () => {
    const { page, context, calls, errors } = await open(viewport);
    try {
      const longLabel = '카드 발급 신청을 할 수 있는 모바일 홈페이지 주소를 문자로 받기';
      await set(page, { status: 'awaiting_user', request: request({ options: [
        { key: '1', label: '신용·체크카드' }, { key: '*', label: longLabel }, { key: '#', label: '다른 서비스' },
      ] }) });
      await highlighted(page).waitFor();
      const end = page.getByRole('button', { name: '종료', exact: true }); const endBefore = await end.boundingBox();
      for (const large of [false, true]) {
        await set(page, { large });
        const short = page.getByRole('button', { name: '신용·체크카드 1번', exact: true });
        const long = page.getByRole('button', { name: longLabel + ' *', exact: true });
        const label = short.locator('.t-bodym').first();
        assert.equal(await label.evaluate(el => parseFloat(getComputedStyle(el).fontSize)), large ? 20 : 16);
        const shortBox = await short.boundingBox(), longBox = await long.boundingBox();
        assert(shortBox.height >= (large ? 70 : 56));
        assert(longBox.height > shortBox.height, '긴 메뉴는 잘리지 않고 버튼 높이가 늘어나야 한다');
        assert.equal(await long.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
        assert(Math.abs((await end.boundingBox()).y - endBefore.y) < 3);
        await page.getByRole('button', { name: '다른 서비스 #', exact: true }).scrollIntoViewIfNeeded();
        await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/choices-${viewport.width}-${large ? 'large' : 'default'}.png`, import.meta.url)) });
      }
      await page.getByRole('button', { name: '다른 서비스 #', exact: true }).click();
      assert.deepEqual(await page.evaluate(() => window.fixtureCommands), [{ requestId: 'q1', key: '#' }]);
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

const samplePath = [
  { step: 1, selected: { key: '5', label: '카드 및 기타 서비스' } },
  { step: 2, selected: { key: '3', label: '카드 문의' } },
  { step: 3, selected: { key: '*', label: '신용·체크카드' } },
];
const longCaption = '카드 발급 문의 서비스입니다. 혜택을 확인하고 카드 발급을 신청할 수 있는 모바일 홈페이지 주소를 문자로 받으시려면 1번, 상담원 연결은 0번을 눌러 주세요. '.repeat(3);
const caption = text => [{ speaker: 'ars', text }];

const menuOptions = [
  { key: '1', label: '신용·체크카드' }, { key: '2', label: '현금카드' },
  { key: '*', label: '다시 듣기' }, { key: '#', label: '다른 서비스' },
];
const menuStep = { step: 3, options: menuOptions, selected: menuOptions[0] };
const menuCaption = { step: 3, speaker: 'ars', text: '신용·체크카드는 1번, 현금카드는 2번, 안내를 다시 들으시려면 별표, 다른 서비스는 샵을 눌러 주세요.' };

for (const viewport of [{ width: 327, height: 688 }, { width: 390, height: 844 }]) {
  test(`같은 안내문 앞부분에서 더보기로 전체를 이어 보고 접기는 전사문 위에 표시한다 (${viewport.width}px)`, async () => {
    const { page, context, calls, errors } = await open(viewport);
    try {
      const text = menuCaption.text + ' 선택하신 서비스에 따라 다음 안내가 이어집니다. 필요한 번호를 확인해 주세요.';
      await set(page, { path: [...samplePath.slice(0, 2), menuStep], captions: [{ ...menuCaption, text }] });
      await page.getByText('최근 선택한 메뉴', { exact: true }).waitFor();
      assert.equal(await page.getByText('받은 메뉴', { exact: true }).count(), 0);
      assert.equal(await page.locator('.menu-brief').count(), 0, '별도 메뉴 요약을 안내문과 나누어 보여주지 않는다');
      const body = page.locator('.ars-caption');
      const expand = page.getByRole('button', { name: '더보기', exact: true });
      await expand.waitFor({ timeout: 1500 });
      assert.equal(await body.textContent(), text);
      assert.equal(await body.isVisible(), true);
      assert.equal(await body.evaluate(el => Math.abs(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight) - 3) < .1), true);
      const recent = await page.locator('.recent-menu').boundingBox();
      const heading = await page.getByText('ARS 안내', { exact: true }).boundingBox();
      assert(heading.y - recent.y - recent.height >= 20, '최근 선택과 안내문 사이 간격을 유지한다');
      const preview = await body.boundingBox(), more = await expand.boundingBox();
      assert(more.y >= preview.y + preview.height, '더보기는 읽던 미리보기 아래에 둔다');
      await expand.focus(); await expand.press('Enter');
      const collapse = page.getByRole('button', { name: '접기', exact: true });
      await collapse.waitFor();
      assert.equal(await collapse.getAttribute('aria-expanded'), 'true');
      assert.equal(await body.textContent(), text, '펼침은 같은 안내문을 사용해야 한다');
      assert.equal(await body.evaluate(el => el.clientHeight >= el.scrollHeight - 1), true);
      assert.equal(await collapse.evaluate(el => el === document.activeElement), true);
      const buttonBox = await collapse.boundingBox(), textBox = await body.boundingBox();
      assert(buttonBox.y + buttonBox.height <= textBox.y);
      const headingBox = await page.getByText('ARS 안내', { exact: true }).boundingBox();
      assert(headingBox.y < buttonBox.y + buttonBox.height && headingBox.y + headingBox.height > buttonBox.y, '접기는 안내 제목과 같은 줄이어야 한다');
      assert(buttonBox.x >= headingBox.x + headingBox.width, '제목과 접기가 겹치지 않아야 한다');
      assert(textBox.y - Math.max(buttonBox.y + buttonBox.height, headingBox.y + headingBox.height) <= 12, '제목 줄과 전사문 사이의 여백을 줄인다');
      await set(page, { large: true, captions: [{ ...menuCaption, text: text + ' 다음 안내입니다.' }] });
      assert.equal(await collapse.getAttribute('aria-expanded'), 'true');
      assert.equal(await body.textContent(), text + ' 다음 안내입니다.');
      await collapse.click();
      assert.equal(await expand.getAttribute('aria-expanded'), 'false');
      assert.equal(await body.isVisible(), true);
      assert.equal(await body.evaluate(el => Math.abs(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight) - 3) < .1), true);
      await page.locator('.card').evaluate(el => { el.scrollTop = 0; });
      await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/caption-more-${viewport.width}-large.png`, import.meta.url)) });
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('짧은 안내는 더보기를 만들지 않고 질문·개인정보·결과 원문을 유지한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { path: [menuStep], captions: [{ ...menuCaption, text: '신용·체크카드 1번' }] });
    await page.getByText('신용·체크카드 1번', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
    await set(page, { status: 'listening_result', captions: [{ ...menuCaption, text: longCaption }] });
    assert.equal(await page.locator('.ars-caption').textContent(), longCaption);
    assert.equal(await page.locator('.ars-caption').evaluate(el => el.clientHeight >= el.scrollHeight - 1), true);
    assert.equal(await page.getByRole('button', { name: /더보기|^접기$/ }).count(), 0);
    const q = request({ reason: 'critical_action' });
    await set(page, { status: 'awaiting_user', request: q });
    await page.getByText(q.message, { exact: true }).waitFor();
    assert.equal(await page.locator('.ars-caption').count(), 0);
    await page.getByRole('button', { name: '신용·체크카드 1번', exact: true }).click();
    assert.deepEqual(await page.evaluate(() => window.fixtureCommands), [{ requestId: 'q1', key: '1' }]);
    await set(page, { request: request({ kind: 'secure_input', label: '생년월일 6자리', message: '생년월일을 입력해 주세요.', terminator: '#', options: [] }) });
    await page.getByText('생년월일 6자리', { exact: true }).waitFor();
    assert.equal(await page.locator('.ars-caption').count(), 0);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('데모도 안내문 더보기·접기를 사용하고 큰글씨와 단계 갱신 시 같은 원문을 유지한다', async () => {
  const { page, context, calls, errors } = await open({ width: 327, height: 688 });
  try {
    await set(page, { demo: true, large: true, phase: 'Listening' });
    const body = page.locator('.ars-caption');
    await page.getByRole('button', { name: '더보기', exact: true }).click({ timeout: 1500 });
    const text = await body.textContent();
    await set(page, { large: false, phase: 'Selecting', selectedKey: '3' });
    assert.equal(await page.getByRole('button', { name: '접기', exact: true }).getAttribute('aria-expanded'), 'true');
    assert.equal(await body.textContent(), text);
    await set(page, { phase: 'Choice' });
    assert.equal(await body.count(), 0);
    assert.equal(await page.getByRole('button', { name: '현재 날씨 3번', exact: true }).count(), 1);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

// A new response must not inherit the reading position of the previous content.
const readingScroll = page => page.locator('.card').evaluate(el => {
  const scroller = el.closest('.scroll');
  return { top: scroller.scrollTop, max: scroller.scrollHeight - scroller.clientHeight };
});
const readToBottom = page => page.locator('.card').evaluate(el => {
  const scroller = el.closest('.scroll'); scroller.scrollTop = scroller.scrollHeight;
});
async function assertQuestionAtTop(page) {
  const bounds = await page.locator('.card').evaluate(el => {
    const scroller = el.closest('.scroll').getBoundingClientRect();
    const title = el.querySelector('.t-title').getBoundingClientRect();
    const cue = el.querySelector('.task-cue').getBoundingClientRect();
    return { top: scroller.top, bottom: scroller.bottom, titleTop: title.top, titleBottom: title.bottom, cueTop: cue.top };
  });
  assert(bounds.cueTop >= bounds.top - 1, '새 응답 안내가 스크롤 영역 위로 가려지면 안 된다');
  assert(bounds.titleTop >= bounds.top - 1 && bounds.titleBottom <= bounds.bottom + 1, '새 질문 전체가 보이는 위치에서 시작해야 한다');
}

for (const viewport of [{ width: 327, height: 688 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
  for (const large of [false, true]) {
    test(`이전 안내를 읽던 중 새 질문이 오면 위로 맞추고 같은 질문 갱신은 읽기·입력을 유지한다 (${viewport.width}px, ${large ? '큰글씨' : '기본'})`, async () => {
      const { page, context, calls, errors } = await open(viewport, large ? 'reduce' : 'no-preference');
      try {
        await set(page, { large, path: samplePath, captions: caption(longCaption) });
        await page.getByText('지나온 메뉴 보기', { exact: true }).click();
        await readToBottom(page);
        assert((await readingScroll(page)).top > 0, '이전 안내를 아래까지 읽는 상황이어야 한다');
        const q = request({ options: Array.from({ length: 6 }, (_, i) => ({ key: String(i), label: '카드 및 금융 서비스 ' + i })) });
        await set(page, { status: 'awaiting_user', request: q });
        await highlighted(page).waitFor();
        await assertQuestionAtTop(page);

        await page.getByRole('textbox').fill('현금카드');
        await readToBottom(page);
        const before = await readingScroll(page);
        assert(before.top > 0);
        await set(page, { request: { ...q, expiresAt: new Date(Date.now() + 90000).toISOString() }, captions: caption('갱신된 안내') });
        assert(Math.abs((await readingScroll(page)).top - before.top) <= 1, '같은 질문의 기한·안내 갱신으로 스크롤을 되돌리면 안 된다');
        assert.equal(await page.getByRole('textbox').inputValue(), '현금카드');
        assert.equal(await page.getByRole('textbox').evaluate(el => el === document.activeElement), true);
        await set(page, { large: !large });
        assert((await readingScroll(page)).top > 0, '글씨 크기 전환을 새 질문처럼 처리하면 안 된다');
        assert.equal(await page.getByRole('textbox').inputValue(), '현금카드');

        await set(page, { request: { ...q, requestId: 'q2', message: '어떤 서비스를 원하세요?' } });
        await page.getByText('어떤 서비스를 원하세요?', { exact: true }).waitFor();
        await assertQuestionAtTop(page);
        assert.equal(await page.getByRole('textbox').inputValue(), '');
        assert.equal(await page.getByRole('textbox').evaluate(el => el === document.activeElement), true, '스크롤 수정은 입력 포커스를 임의로 옮기지 않는다');
        assert.deepEqual(calls, []); assert.deepEqual(errors, []);
        await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/question-scroll-${viewport.width}-${large ? 'large' : 'default'}.png`, import.meta.url)) });
      } finally { await context.close(); }
    });
  }
}

for (const large of [false, true]) {
  test(`데모에서도 새 질문으로 전환하면 읽던 스크롤을 위로 맞춘다 (${large ? '큰글씨' : '기본'})`, async () => {
    const { page, context, calls, errors } = await open({ width: 320, height: 568 });
    try {
      await set(page, { demo: true, large, phase: 'Selecting', selectedKey: '3' });
      await page.getByText('지나온 메뉴 보기', { exact: true }).click();
      await readToBottom(page);
      assert((await readingScroll(page)).top > 0);
      await set(page, { phase: 'Choice' });
      await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor();
      await assertQuestionAtTop(page);
      await readToBottom(page);
      assert((await readingScroll(page)).top > 0);
      await set(page, { phase: 'Answer' });
      await page.getByText('어느 지역의 날씨인가요?', { exact: true }).waitFor();
      await assertQuestionAtTop(page);
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('카드의 파란 소제목은 서버·데모·큰글씨 모두 일반 글자 대비 4.5:1 이상이다', async () => {
  const { page, context, calls, errors } = await open();
  const luminance = color => {
    const values = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => v / 255).map(v => v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4);
    return values[0] * .2126 + values[1] * .7152 + values[2] * .0722;
  };
  try {
    for (const large of [false, true]) {
      for (const state of [{ demo: false }, { demo: true, phase: 'Listening' }, { demo: true, phase: 'Selecting', selectedKey: '3' }, { demo: true, phase: 'ListeningResult' }]) {
        await set(page, { ...state, large, path: samplePath, captions: caption('현재 안내') });
        await page.locator('.card .t-label.blue').first().waitFor();
        const colors = await page.locator('.card .t-label.blue').evaluateAll(els => els.map(el => ({ text: el.textContent, fg: getComputedStyle(el).color, bg: getComputedStyle(el.closest('.card')).backgroundColor })));
        for (const { text, fg, bg } of colors) {
          const [lighter, darker] = [luminance(fg), luminance(bg)].sort((a, b) => b - a);
          assert((lighter + .05) / (darker + .05) >= 4.5, text + ' 글자 대비가 부족하다');
        }
      }
    }
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('최근 선택만 펼쳐두고 전체 경로는 순서대로 접어 표시하며 한 단계는 중복하지 않는다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { path: [samplePath[2], samplePath[0], samplePath[1]] });
    await page.getByText('최근 선택한 메뉴', { exact: true }).waitFor({ timeout: 1500 });
    for (const large of [false, true]) {
      await set(page, { large });
      const toggleBox = await page.locator('summary').boundingBox();
      assert.equal(await page.getByText('기다려 주세요', { exact: true }).count(), 0);
      const titleBox = await page.getByText('원하는 메뉴를 찾고 있어요', { exact: true }).boundingBox();
      assert(toggleBox.y + toggleBox.height <= titleBox.y, '경로 토글은 현재 진행 제목보다 위에 있어야 한다');
    }
    assert.equal(await page.getByText('신용·체크카드 (*번)', { exact: true }).count(), 0);
    assert.equal(await page.locator('.recent-menu').innerText(), '신용·체크카드 (*)');
    const history = page.locator('.path-history');
    assert.equal(await history.isVisible(), false);
    await page.getByText('지나온 메뉴 보기', { exact: true }).click();
    assert.deepEqual(await history.locator('li').allTextContents(), ['카드 및 기타 서비스 (5번)', '카드 문의 (3번)', '신용·체크카드 (*)']);
    await set(page, { large: true, path: [...samplePath, { step: 4, selected: { key: '#', label: '다른 서비스' } }] });
    assert.equal(await history.isVisible(), true);
    assert.equal(await history.locator('li').count(), 4);
    await set(page, { path: [samplePath[0]] });
    assert.equal(await page.getByText('카드 및 기타 서비스 (5번)', { exact: true }).count(), 1);
    assert.equal(await page.getByText(/지나온 메뉴 보기|지나온 메뉴 접기/).count(), 0);
    await set(page, { path: [] });
    assert.equal(await page.getByText('최근 선택한 메뉴', { exact: true }).count(), 0);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 327, height: 688 }, { width: 390, height: 844 }]) {
  test(`긴 현재 안내는 3줄로 접고 펼침을 안내 갱신·큰글씨·질문 전환 후에도 유지한다 (${viewport.width}px)`, async () => {
    const { page, context, calls, errors } = await open(viewport);
    try {
      await set(page, { path: samplePath, captions: [{ speaker: 'ars', text: '이전 안내입니다.' }, ...caption(longCaption)] });
      const expand = page.getByRole('button', { name: '더보기', exact: true });
      await expand.waitFor({ timeout: 1500 });
      const body = page.locator('.ars-caption');
      assert.equal(await page.getByText('이전 안내입니다.', { exact: true }).count(), 0);
      assert.equal(await expand.getAttribute('aria-expanded'), 'false');
      assert.equal(await body.evaluate(el => Math.abs(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight) - 3) < .1), true);
      await expand.click();
      const collapse = page.getByRole('button', { name: '접기', exact: true });
      await collapse.waitFor(); assert.equal(await collapse.getAttribute('aria-expanded'), 'true');
      const collapseBox = await collapse.boundingBox();
      const transcriptBox = await body.boundingBox();
      assert(collapseBox.y + collapseBox.height <= transcriptBox.y, '메뉴 정리가 없는 전사문도 접기 버튼은 위에 둔다');
      assert.equal(await body.evaluate(el => el.getBoundingClientRect().height > parseFloat(getComputedStyle(el).lineHeight) * 3), true);
      await set(page, { large: true, captions: caption(longCaption + '새 안내입니다.') });
      assert.equal(await collapse.getAttribute('aria-expanded'), 'true');
      await page.getByText('지나온 메뉴 보기', { exact: true }).click();
      await set(page, { status: 'awaiting_user', request: request() });
      await page.getByRole('textbox').fill('현금카드');
      await set(page, { large: false });
      assert.equal(await page.getByRole('textbox').inputValue(), '현금카드');
      assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
      await set(page, { status: 'listening', request: null });
      await page.getByRole('button', { name: '접기', exact: true }).waitFor();
      assert.equal(await page.locator('.path-history').isVisible(), true);
      await page.getByText('지나온 메뉴 접기', { exact: true }).click();
      await page.getByRole('button', { name: '접기', exact: true }).click();
      await set(page, { large: true });
      await page.getByRole('button', { name: '더보기', exact: true }).waitFor();
      assert.equal(await body.evaluate(el => Math.abs(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight) - 3) < .1), true);
      await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
      await page.locator('.card').evaluate(el => { el.scrollTop = 0; });
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/progress-folded-${viewport.width}.png`, import.meta.url)) });
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('짧은 안내·결과 단계·결과 요약·개인정보 입력 안내는 접지 않는다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { captions: caption('현재 날씨 안내입니다.') });
    await page.getByText('현재 날씨 안내입니다.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
    await set(page, { status: 'listening_result', captions: caption(longCaption) });
    await page.getByText(longCaption.trim(), { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: /더보기|접기/ }).count(), 0);
    assert.equal(await page.locator('.ars-caption').evaluate(el => el.clientHeight >= el.scrollHeight - 1), true);
    await set(page, { status: 'awaiting_user', request: request({ kind: 'secure_input', label: '주민등록번호 앞 6자리', message: longCaption, terminator: '#', options: [] }) });
    await page.getByText('주민등록번호 앞 6자리', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
    await set(page, { status: 'ended', request: null, result: { path: samplePath, summary: '문자 발송 예정 안내까지 진행했어요.', keyInfo: [{ label: '처리', value: '문자 발송 예정' }], handoffScript: null, endReason: 'message_sent', durationSec: 180 } });
    await page.getByText('문자 발송 예정 안내까지 진행했어요.', { exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('접기 조작은 키보드로 가능하며 새 통화 화면에서는 기본 접힘으로 시작한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { path: samplePath, captions: caption(longCaption) });
    const summary = page.locator('summary');
    await summary.focus(); await summary.press('Enter');
    await page.getByText('지나온 메뉴 접기', { exact: true }).waitFor();
    assert.equal(await page.locator('.path-history').isVisible(), true);
    await summary.press('Space');
    await page.getByText('지나온 메뉴 보기', { exact: true }).waitFor();
    assert.equal(await page.locator('.path-history').isVisible(), false);
    const expand = page.getByRole('button', { name: '더보기', exact: true });
    await expand.focus(); await expand.press('Enter');
    await page.getByRole('button', { name: '접기', exact: true }).waitFor();
    await page.setViewportSize({ width: 320, height: 568 });
    await set(page, { large: true });
    assert.equal(await page.getByRole('button', { name: '접기', exact: true }).getAttribute('aria-expanded'), 'true');
    await page.reload(); await page.locator('.call-number').waitFor();
    await set(page, { path: samplePath, captions: caption(longCaption) });
    await page.getByRole('button', { name: '더보기', exact: true }).waitFor();
    assert.equal(await page.locator('.path-history').isVisible(), false);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('데모의 선택 경로도 최근 선택과 접힌 전체 경로로 표시하고 큰글씨 전환 시 유지한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await set(page, { demo: true, phase: 'Selecting', selectedKey: '3' });
    await page.getByText('최근 선택한 메뉴', { exact: true }).waitFor();
    assert.equal(await page.locator('.recent-menu').innerText(), '현재 날씨 (3번)');
    assert.equal(await page.locator('.path-history').isVisible(), false);
    await page.getByText('지나온 메뉴 보기', { exact: true }).click();
    await set(page, { large: true });
    assert.deepEqual(await page.locator('.path-history li').allTextContents(), ['경기 지역', '수원', '현재 날씨 (3번)']);
    assert.equal(await page.locator('.path-history').isVisible(), true);
    await set(page, { phase: 'ListeningResult' });
    assert.equal(await page.getByRole('button', { name: '더보기', exact: true }).count(), 0);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});
