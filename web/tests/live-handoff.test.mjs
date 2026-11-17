import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { chromium } from 'playwright';

// 실제 통화 화면의 상담원 연결 버튼(통화 중 request_handoff)과 업무 입력(handoff_goal)을 로컬 상태로 렌더
// 서버 API 호출 없음. 화면 쪽 호출과 컨트롤러가 만드는 요청 본문만 기록
const fixture = `
const fx = new Store({ status: "listening", request: null, available: true, canSend: true });
window.fxSet = (patch) => fx.update((s) => ({ ...s, ...patch }));
window.fxCalls = [];
function Fixture() {
  const s = useStore(fx);
  const controls = { ...liveControls0(), elapsedSeconds: 18, endEnabled: true, progressConnected: true, arsAvailable: true,
    handoffAvailable: s.available, onHandoff: () => window.fxCalls.push({ type: "handoff" }) };
  return html\`<div class="app"><div class="app-viewport"><\${BServerCallScreen} number="15889999"
    snapshot=\${mkSnap(s.status, "처리 중", { pendingRequest: s.request })} onEnd=\${() => {}} onBack=\${() => {}} controls=\${controls}
    choiceInput=\${{ canSend: s.canSend, retryVersion: 0, onSelect: (a) => window.fxCalls.push({ type: "select", ...a }) }}
    answerInput=\${{ canSend: s.canSend, onAnswer: (a) => window.fxCalls.push({ type: "answer", ...a }), onMic() {}, canUseMic: false,
      onCancelHandoff: (requestId) => window.fxCalls.push({ type: "cancel_handoff", requestId }) }}
    secureInput=\${{ canSend: s.canSend, retryVersion: 0, onSubmit: () => {} }}/></div></div>\`;
}
// 컨트롤러 시험용: 가짜 게이트웨이로 LiveCallOwner 를 만들어 보낸 요청 본문을 기록한다
window.makeOwner = (status, request = null, fail = null) => {
  const sent = [];
  const fake = { act: async (session, action) => { sent.push(action); if (fail) throw new ApiHttpError(409, fail);
    return { accepted: true, keySent: null, status: null, goalSummary: null }; }, close() {} };
  const o = new LiveCallOwner(() => fake);
  o.entry.set({ number: "15889999", generation: 1, goal: "통장 잔액", agent: false, confirmation: null, elapsedSeconds: null });
  o.server.api = fake; o.server.session = { sessionId: "ses_t", sessionToken: "tok" };
  o.server.store.set({ ...serverState0(), hasSession: true, streamConnected: true, snapshot: mkSnap(status, "처리 중", { pendingRequest: request }) });
  return { o, sent };
};
// 실제 API 클라이언트 경로: HttpServerTestGateway → ArsApiClient(행동 검사·주소 구성) → fetch. fetch 만 가짜로 바꿔 보낸 본문을 기록
window.makeRealOwner = (status, request = null) => {
  const sent = [];
  window.fetchReply = null;
  window.fetch = async (url, init) => {
    sent.push({ url: String(url), body: JSON.parse(init.body) });
    const r = window.fetchReply ? window.fetchReply(JSON.parse(init.body)) : null;
    if (r === "network") throw new TypeError("network");
    if (r) return new Response(JSON.stringify(r.body), { status: r.status, headers: { "Content-Type": "application/json" } });
    return new Response(JSON.stringify({ accepted: true }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
  const o = new LiveCallOwner(() => null);
  o.entry.set({ number: "15889999", generation: 1, goal: "통장 잔액", agent: false, confirmation: null, elapsedSeconds: null });
  o.server.api = new HttpServerTestGateway("test-key", location.origin + "/v1", true);
  o.server.session = { sessionId: "ses_t", sessionToken: "tok" };
  o.server.store.set({ ...serverState0(), hasSession: true, streamConnected: true, snapshot: mkSnap(status, "처리 중", { pendingRequest: request }) });
  return { o, sent };
};
render(html\`<\${Fixture}/>\`, document.getElementById("root"));`;

let browser, server, origin;
before(async () => {
  let html = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const entry = 'render(html`<${Root}/>`, document.getElementById("root"));';
  assert(html.includes(entry));
  html = html.replace(entry, fixture);
  server = createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/app') { res.writeHead(200, { 'Content-Type': 'text/html;charset=utf-8' }); return res.end(html); }
    res.writeHead(403); res.end();
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
});
after(async () => { await browser?.close(); await new Promise(r => server?.close(r)); });

async function open(viewport = { width: 390, height: 844 }) {
  const context = await browser.newContext({ viewport });
  const page = await context.newPage(), errors = [], calls = [];
  await context.route('**/*', route => {
    if (route.request().method() !== 'GET') calls.push(route.request().url());
    return route.request().url().startsWith(origin) && !route.request().url().includes('/v1/') ? route.continue() : route.abort();
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin + '/app');
  await page.locator('.call-number').waitFor({ timeout: 3000 });
  return { page, context, errors, calls };
}
const set = (page, patch) => page.evaluate(p => window.fxSet(p), patch);
const fxCalls = (page) => page.evaluate(() => window.fxCalls);
const handoffButton = (page) => page.getByRole('button', { name: '상담원 연결', exact: true });
const handoffRequest = (overrides = {}) => ({ kind: 'handoff_goal', requestId: 'req_7', message: '상담원에게 연결해 드릴게요. 어떤 업무로 상담받으실지 말씀해 주세요.',
  reason: null, expiresAt: new Date(Date.now() + 60000).toISOString(), options: [], label: null, arsPrompt: null, terminator: null, ...overrides });

test('실제 통화 화면에 상담원 연결 버튼이 있고 누르면 통화 중 상담원 연결을 요청한다', async () => {
  const { page, context, errors, calls } = await open();
  try {
    const button = handoffButton(page);
    await button.waitFor();
    assert.equal(await button.isEnabled(), true);
    await button.click();
    assert.deepEqual(await fxCalls(page), [{ type: 'handoff' }]);
    await set(page, { available: false });
    assert.equal(await button.isEnabled(), false, '요청할 수 없는 상태면 누를 수 없다');
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('업무 입력 요청(handoff_goal)이 오면 서버 문장을 제목으로 보이고 입력한 업무를 보내거나 돌아갈 수 있다', async () => {
  const { page, context, errors, calls } = await open();
  try {
    const q = handoffRequest();
    await set(page, { status: 'awaiting_user', request: q, available: false });
    await page.locator('.t-title').filter({ hasText: q.message }).waitFor();
    await page.locator('.card.needs-response').waitFor();
    const submit = page.getByTestId('handoff-submit');
    assert.equal(await submit.isEnabled(), false, '업무를 넣기 전에는 보낼 수 없다');
    await page.getByRole('textbox').fill('  카드 발급 문의 ');
    await submit.click();
    assert.deepEqual(await fxCalls(page), [{ type: 'answer', requestId: 'req_7', text: '카드 발급 문의' }]);
    await page.getByRole('button', { name: '돌아가기', exact: true }).click();
    assert.deepEqual((await fxCalls(page)).at(-1), { type: 'cancel_handoff', requestId: 'req_7' });
    assert.equal(await handoffButton(page).isEnabled(), false, '업무 입력 중에는 다시 요청하지 않는다');
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('보낼 수 없거나 기한이 지난 업무 입력 요청은 입력·전송·돌아가기를 막는다', async () => {
  const { page, context, errors } = await open();
  try {
    await set(page, { status: 'awaiting_user', request: handoffRequest(), canSend: false });
    await page.getByTestId('handoff-submit').waitFor();
    assert.equal(await page.getByRole('textbox').isEnabled(), false);
    assert.equal(await page.getByRole('button', { name: '돌아가기', exact: true }).isEnabled(), false);
    await set(page, { canSend: true, request: handoffRequest({ requestId: 'req_8', expiresAt: new Date(Date.now() - 1000).toISOString() }) });
    await page.waitForTimeout(50);
    assert.equal(await page.getByRole('textbox').isEnabled(), false, '기한이 지나면 입력할 수 없다');
    assert.equal(await page.locator('.card.needs-response').count(), 0);
    assert.deepEqual(await fxCalls(page), []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('컨트롤러: 상담원 연결 요청·업무 답·취소의 요청 본문과 요청할 수 없는 상태', async () => {
  const { page, context, errors } = await open();
  try {
    const r = await page.evaluate(async () => {
      const tick = () => new Promise((res) => setTimeout(res, 0));
      const out = {};
      let { o, sent } = window.makeOwner('listening');
      o.dispatch({ type: 'RequestHandoff' }); await tick();
      out.request = sent.slice();
      const q = { kind: 'handoff_goal', requestId: 'req_7', message: 'm', reason: null, expiresAt: new Date(Date.now() + 60000).toISOString(), options: [], label: null, arsPrompt: null, terminator: null };
      ({ o, sent } = window.makeOwner('awaiting_user', q));
      o.dispatch({ type: 'Respond', requestId: 'req_7', value: ' 카드 발급 ', selection: false }); await tick();
      o.dispatch({ type: 'CancelHandoff', requestId: 'req_7' }); await tick();
      out.answerThenCancel = sent.slice();
      ({ o, sent } = window.makeOwner('awaiting_user', q));
      o.dispatch({ type: 'Respond', requestId: 'req_7', value: '1', selection: true }); await tick();
      o.dispatch({ type: 'CancelHandoff', requestId: 'req_7' }); await tick();
      out.selectIgnoredThenCancel = sent.filter((a) => a.type !== 'client_note');
      out.blocked = {};
      for (const [name, status, req] of [
        ['dialing', 'dialing', null], ['waiting_agent', 'waiting_agent', null], ['connected_to_agent', 'connected_to_agent', null],
        ['secure_input', 'awaiting_user', { ...q, kind: 'secure_input', requestId: 'req_s' }],
        ['critical_action', 'awaiting_user', { ...q, kind: 'clarification', reason: 'critical_action', requestId: 'req_c', options: [{ key: '1', label: '송금', category: null, confidence: null }] }],
        ['handoff_open', 'awaiting_user', q]]) {
        ({ o, sent } = window.makeOwner(status, req));
        o.dispatch({ type: 'RequestHandoff' }); await tick();
        out.blocked[name] = sent.length;
      }
      ({ o, sent } = window.makeOwner('awaiting_user', { ...q, kind: 'clarification', reason: 'ambiguous', requestId: 'req_q', options: [{ key: '1', label: '조회', category: null, confidence: null }] }));
      o.dispatch({ type: 'RequestHandoff' }); await tick();
      out.duringQuestion = sent.slice();
      ({ o, sent } = window.makeOwner('listening', null, 'sensitive_input_blocked'));
      o.dispatch({ type: 'RequestHandoff' }); await tick(); await tick();
      out.refused = { error: o.server.state.error, uncertain: o.server.state.actionUncertain, busy: o.server.state.busy };
      ({ o, sent } = window.makeOwner('listening', null, 'invalid_state'));
      o.dispatch({ type: 'RequestHandoff' }); await tick(); await tick();
      out.refusedState = { error: o.server.state.error, uncertain: o.server.state.actionUncertain };
      return out;
    });
    assert.deepEqual(r.request, [{ type: 'request_handoff' }]);
    assert.deepEqual(r.answerThenCancel, [{ type: 'answer', request_id: 'req_7', text: '카드 발급' }], '답을 보낸 요청은 취소하지 않는다');
    assert.deepEqual(r.selectIgnoredThenCancel, [{ type: 'cancel_handoff', request_id: 'req_7' }], '업무 입력은 번호 고르기로 답하지 않는다');
    assert.deepEqual(r.blocked, { dialing: 0, waiting_agent: 0, connected_to_agent: 0, secure_input: 0, critical_action: 0, handoff_open: 0 });
    assert.deepEqual(r.duringQuestion, [{ type: 'request_handoff' }], '일반 재질문 중에는 요청할 수 있다 (서버가 재질문을 닫고 취소하면 다시 연다)');
    assert.equal(r.refused.error, '개인정보 입력 중에는 상담원 연결로 바꿀 수 없어요. 입력을 마친 뒤 다시 눌러 주세요.');
    assert.equal(r.refused.uncertain, false, '서버 거절(409)은 전달 불확실로 보지 않는다');
    assert.equal(r.refused.busy, false);
    assert.match(r.refusedState.error, /지금은 상담원 연결을 요청할 수 없어요/);
    assert.equal(r.refusedState.uncertain, false);
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }]) {
  test(`실제 통화의 네 버튼은 화면 안에서 겹치지 않는다 (${viewport.width}px)`, async () => {
    const { page, context, errors } = await open(viewport);
    try {
      await handoffButton(page).waitFor();
      const boxes = await page.locator('.agent-controls > button').evaluateAll(els => els.map(e => { const r = e.getBoundingClientRect(); return { l: r.left, r: r.right, t: r.top, b: r.bottom }; }));
      assert.equal(boxes.length, 4);
      for (const b of boxes) assert(b.l >= 0 && b.r <= viewport.width, '버튼이 화면 밖으로 나가지 않는다');
      for (let i = 1; i < boxes.length; i++) assert(boxes[i].l >= boxes[i - 1].r - 0.5, '버튼이 겹치지 않는다');
      assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('실제 API 클라이언트로 상담원 연결 요청·취소·업무 답이 서버 주소로 나가고, 거절·전달 실패를 구분한다', async () => {
  const { page, context, errors } = await open();
  try {
    const r = await page.evaluate(async () => {
      const settle = () => new Promise((res) => setTimeout(res, 20));
      const q = { kind: 'handoff_goal', requestId: 'req_7', message: 'm', reason: null, expiresAt: new Date(Date.now() + 60000).toISOString(), options: [], label: null, arsPrompt: null, terminator: null };
      const out = {};
      let { o, sent } = window.makeRealOwner('listening');
      o.dispatch({ type: 'RequestHandoff' }); await settle();
      out.request = sent.map((x) => ({ path: new URL(x.url).pathname, body: x.body }));
      out.requestError = o.server.state.error;
      ({ o, sent } = window.makeRealOwner('awaiting_user', q));
      o.dispatch({ type: 'CancelHandoff', requestId: 'req_7' }); await settle();
      out.cancel = sent.map((x) => x.body);
      ({ o, sent } = window.makeRealOwner('listening'));
      window.fetchReply = () => ({ status: 409, body: { error: { code: 'sensitive_input_blocked' } } });
      o.dispatch({ type: 'RequestHandoff' }); await settle();
      out.refused = { error: o.server.state.error, uncertain: o.server.state.actionUncertain };
      ({ o, sent } = window.makeRealOwner('awaiting_user', q));
      window.fetchReply = () => ({ status: 409, body: { error: { code: 'invalid_state' } } });
      o.dispatch({ type: 'Respond', requestId: 'req_7', value: '카드 발급', selection: false }); await settle();
      out.lateAnswer = { body: sent.map((x) => x.body), error: o.server.state.error, uncertain: o.server.state.actionUncertain };
      ({ o, sent } = window.makeRealOwner('awaiting_user', q));
      window.fetchReply = () => 'network';
      o.dispatch({ type: 'CancelHandoff', requestId: 'req_7' }); await settle();
      out.cancelLost = { uncertain: o.server.state.actionUncertain };
      window.fetchReply = null;
      o.server.store.update((st) => ({ ...st, actionUncertain: false, error: null }));   // 상태 다시 연결로 같은 요청이 그대로 확인된 경우
      o.dispatch({ type: 'CancelHandoff', requestId: 'req_7' }); await settle();
      out.cancelRetry = sent.map((x) => x.body);
      return out;
    });
    assert.deepEqual(r.request, [{ path: '/v1/sessions/ses_t/actions', body: { type: 'request_handoff' } }]);
    assert.equal(r.requestError, null, '행동 검사에서 막히지 않는다');
    assert.deepEqual(r.cancel, [{ type: 'cancel_handoff', request_id: 'req_7' }]);
    assert.equal(r.refused.error, '개인정보 입력 중에는 상담원 연결로 바꿀 수 없어요. 입력을 마친 뒤 다시 눌러 주세요.');
    assert.equal(r.refused.uncertain, false);
    assert.deepEqual(r.lateAnswer.body, [{ type: 'answer', request_id: 'req_7', text: '카드 발급' }]);
    assert.match(r.lateAnswer.error, /업무 입력 시간이 지났거나 요청이 바뀌어/);
    assert.equal(r.lateAnswer.uncertain, false, '기한 직전 답의 409 는 전달 불확실이 아니다');
    assert.equal(r.cancelLost.uncertain, true, '전달 여부를 모르는 실패는 불확실로 표시');
    assert.deepEqual(r.cancelRetry, [{ type: 'cancel_handoff', request_id: 'req_7' }, { type: 'cancel_handoff', request_id: 'req_7' }], '다시 연결한 뒤 같은 요청을 다시 취소할 수 있다');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});
