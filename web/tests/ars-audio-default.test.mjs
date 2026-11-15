import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// Exercise the shipped binding/player with local PCM frames. No session API or real call.
const fixture = `
const audioFixture = new Store({ generation: 0, state: serverState0() });
window.audioSession = id => {
  const value = { sessionId: id, sessionToken: "test-token", audioUrl: "wss://ars-ui.test/v1/sessions/" + id + "/audio" };
  sessions.current = value; sessions.observer?.(value);
  audioFixture.set({ generation: audioFixture.value.generation + 1,
    state: { ...serverState0(), hasSession: true, streamConnected: true, snapshot: mkSnap("dialing", "연결 중") } });
};
window.audioState = (status, extra = {}) => audioFixture.update(f => ({ ...f,
  state: { ...f.state, ...extra, snapshot: { ...f.state.snapshot, status } } }));
window.audioVisibility = value => {
  Object.defineProperty(document, "visibilityState", { configurable: true, value });
  document.dispatchEvent(new Event("visibilitychange"));
};
window.demoAudio = () => {
  let state = reduce({ ...appState0(), number: "131" }, { type: "StartAgent" });
  state = reduce(state, { type: "GoalChanged", text: "날씨 안내" });
  state = reduce(state, { type: "CommitGoal" });
  state = reduce(state, { type: "DemoAdvance" });
  const connected = state.audioOn;
  state = reduce(state, { type: "ToggleAudio" });
  state = reduce(state, { type: "DemoAdvance" });
  return { connected, afterMuteAndAdvance: state.audioOn, ended: reduce(state, { type: "End" }).audioOn };
};
function AudioFixture() {
  const f = useStore(audioFixture);
  return html\`<button onClick=\${() => window.audioSession("ses_one")}>새 테스트 통화</button>
    <\${MainLiveBindings} entry=\${f.generation ? { generation: f.generation } : null} state=\${f.state}>
    \${(voice) => html\`<button disabled=\${!voice} onClick=\${() => voice?.toggleArs()}>\${voice?.arsEnabled ? "ARS 소리 끄기" : "ARS 소리 켜기"}</button>
      <output data-testid="playing">\${String(voice?.arsPlaying === true)}</output>
      <output data-testid="audio-error">\${voice?.arsError ?? ""}</output>\`}<//>\`;
}
if (location.hash === "#confirm") {
  Prefs.appKey = "test-only-key";
  render(html\`<\${LiveConfirmDialog} target=\${{ number: "131", goal: "날씨 안내" }} dismiss=\${() => {}}
    confirm=\${() => { window.confirmedAudioContext = sharedCtx; }}/>\`, document.getElementById("root"));
} else render(html\`<\${AudioFixture}/>\`, document.getElementById("root"));`;

let browser, source;
before(async () => {
  source = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const entry = 'render(html`<${Root}/>`, document.getElementById("root"));';
  assert(source.includes(entry));
  source = source.replace(entry, fixture);
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
});
after(async () => browser?.close());

async function open(confirmOnly = false) {
  const context = await browser.newContext();
  const sockets = [], calls = [], errors = [];
  await context.route('**/*', route => {
    if (route.request().url() === 'https://ars-ui.test/app') return route.fulfill({ contentType: 'text/html', body: source });
    if (confirmOnly && route.request().url() === 'https://ars-ui.test/v1/health') return route.fulfill({ status: 503, body: '' });
    if (route.request().url().includes('/v1/') || route.request().method() !== 'GET') calls.push(route.request().url());
    return route.abort();
  });
  // Intercept WebSocket transport only: URL validation, PCM player and binding are real.
  await context.routeWebSocket('**/*', ws => {
    assert.match(ws.url(), /^wss:\/\/ars-ui\.test\/v1\/sessions\/ses_(one|two)\/audio\?token=test-token$/);
    sockets.push(ws);
    ws.send(Buffer.alloc(160, 0xff));
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('https://ars-ui.test/app' + (confirmOnly ? '#confirm' : ''));
  if (!confirmOnly) await page.getByRole('button', { name: '새 테스트 통화', exact: true }).click();
  return { context, page, sockets, calls, errors };
}
const update = (page, status, extra = {}) => page.evaluate(({ status, extra }) => audioState(status, extra), { status, extra });
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const on = page => page.getByRole('button', { name: 'ARS 소리 끄기', exact: true });
const off = page => page.getByRole('button', { name: 'ARS 소리 켜기', exact: true });

test('키보드로 발신을 승인할 때 사용자 조작 안에서 오디오를 활성화한다', async () => {
  const { page, context, calls, errors } = await open(true);
  try {
    const confirm = page.getByRole('button', { name: '실제 발신', exact: true });
    await confirm.focus(); await page.keyboard.press('Enter');
    await page.waitForFunction(() => window.confirmedAudioContext?.state === 'running', null, { timeout: 2000 });
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('연결되면 ARS PCM을 자동 재생하고 사용자가 끈 선택은 상태 갱신에도 유지한다', async () => {
  const { page, context, sockets, calls, errors } = await open();
  try {
    assert.equal(sockets.length, 0, '발신 중에는 자동 재생하지 않는다');
    await update(page, 'listening');
    await on(page).waitFor({ timeout: 2500 });
    await page.waitForFunction(() => document.querySelector('[data-testid="playing"]').textContent === 'true');
    assert.equal(sockets.length, 1);
    await update(page, 'selecting'); await settle(page);
    assert.equal(sockets.length, 1, '단계 변경마다 다시 연결하지 않는다');
    await on(page).click();
    await update(page, 'listening_result'); await settle(page);
    await off(page).waitFor(); assert.equal(sockets.length, 1, '사용자가 끈 뒤 자동으로 켜지 않는다');
    await off(page).click(); await on(page).waitFor();
    assert.equal(sockets.length, 2, '원할 때 직접 다시 켤 수 있다');
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('새 통화는 다시 기본 재생하며 종료·상담원 연결 시 재생을 멈춘다', async () => {
  const { page, context, sockets, calls, errors } = await open();
  try {
    await update(page, 'listening'); await on(page).waitFor({ timeout: 2500 });
    await on(page).click();
    await page.evaluate(() => audioSession('ses_two'));
    await off(page).waitFor();
    await update(page, 'awaiting_user'); await on(page).waitFor();
    assert.equal(sockets.length, 2);
    await update(page, 'connected_to_agent'); await off(page).waitFor();
    await update(page, 'ended'); await settle(page);
    assert.equal(sockets.length, 2);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('재생 오류가 나도 자동 재시도하지 않고 사용자가 다시 켤 수 있다', async () => {
  const { page, context, sockets, calls, errors } = await open();
  try {
    await update(page, 'listening'); await on(page).waitFor({ timeout: 2500 });
    sockets[0].close({ code: 1011, reason: 'test disconnect' });
    await off(page).waitFor();
    await page.waitForFunction(() => document.querySelector('[data-testid="audio-error"]').textContent.length > 0);
    await update(page, 'selecting'); await settle(page);
    assert.equal(sockets.length, 1);
    await off(page).click(); await on(page).waitFor(); assert.equal(sockets.length, 2);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('백그라운드·상태 확인 불가 시 소리를 멈추고 화면 복귀만으로 재생하지 않는다', async () => {
  const { page, context, sockets, calls, errors } = await open();
  try {
    await update(page, 'listening'); await on(page).waitFor({ timeout: 2500 });
    await page.evaluate(() => audioVisibility('hidden')); await off(page).waitFor();
    await update(page, 'selecting'); await settle(page); assert.equal(sockets.length, 1);
    await page.evaluate(() => audioVisibility('visible')); await settle(page);
    await off(page).waitFor(); assert.equal(sockets.length, 1);
    await off(page).click(); await on(page).waitFor();
    await update(page, 'listening', { sessionUnavailable: true }); await off(page).waitFor();
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('데모도 연결 후 기본 켜짐이며 수동 끄기와 종료 상태를 유지한다', async () => {
  const { page, context } = await open();
  try {
    assert.deepEqual(await page.evaluate(() => demoAudio()), { connected: true, afterMuteAndAdvance: false, ended: false });
  } finally { await context.close(); }
});

test('발신 중 화면을 떠난 경우 뒤늦은 연결로 자동 재생하지 않는다', async () => {
  const { page, context, sockets, calls, errors } = await open();
  try {
    await page.evaluate(() => audioVisibility('hidden'));
    // A session credential can also arrive while the page is already hidden.
    await page.evaluate(() => audioSession('ses_two'));
    await update(page, 'listening'); await settle(page);
    assert.equal(sockets.length, 0);
    await page.evaluate(() => audioVisibility('visible')); await settle(page);
    await off(page).waitFor({ timeout: 1500 }); assert.equal(sockets.length, 0);
    await off(page).click(); await on(page).waitFor(); assert.equal(sockets.length, 1);
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

for (const outcome of ['manual-off', 'stream-error']) {
  test(`발신 중 직접 재생한 뒤 ${outcome}이면 연결되어도 자동으로 켜지 않는다`, async () => {
    const { page, context, sockets, calls, errors } = await open();
    try {
      await off(page).click(); await on(page).waitFor();
      await page.waitForFunction(() => document.querySelector('[data-testid="playing"]').textContent === 'true');
      if (outcome === 'manual-off') await on(page).click();
      else sockets[0].close({ code: 1011, reason: 'test disconnect before connection' });
      await off(page).waitFor();
      await update(page, 'listening'); await settle(page);
      await off(page).waitFor({ timeout: 1500 }); assert.equal(sockets.length, 1);
      await off(page).click(); await on(page).waitFor();
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}
