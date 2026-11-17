import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { chromium, webkit } from 'playwright';

// Real call components, local fixtures only. No backend, microphone or outbound call.
const fixture = `
const fx = new Store({ large: false, demo: false, phase: "Secure", secureCount: 0 });
const request = { kind: "secure_input", requestId: "secure_test", label: "휴대폰 번호",
  message: "휴대폰 번호를 직접 눌러 주세요. 입력하신 번호는 저장되지 않습니다.",
  terminator: "#", options: [], expiresAt: new Date(Date.now() + 300000).toISOString() };
window.fxSet = patch => fx.update(s => ({ ...s, ...patch }));
window.secureCommands = [];
function Fixture() {
  const s = useStore(fx);
  const activeRequest = { ...request, ...s.request };
  const dispatch = a => { window.secureCommands.push(a); window.fxSet(reduce({ ...appState0(), screen: "Call", agent: true, number: "15889999", ...s }, a)); };
  const controls = { ...liveControls0(), elapsedSeconds: 157, endEnabled: true, progressConnected: true,
    extras: () => s.extra ? html\`<div class="red">ARS 소리를 재생하지 못했어요. 상태 확인 후 다시 켜 주세요.</div>\` : null };
  return html\`<\${TextSizeContext.Provider} value=\${s.large}><div class=\${"app" + (s.large ? " large-text" : "") + (s.anonymous ? " anon" : "")}>
    <header class="app-toolbar"><\${TextSizeControl} largeText=\${s.large} onToggle=\${() => window.fxSet({ large: !s.large })}/></header>
    <div class="app-viewport">\${s.demo
      ? html\`<div class="screen grad"><\${CallScreen} state=\${{ ...appState0(), screen: "Call", agent: true, number: "15889999", ...s }} dispatch=\${dispatch}/></div>\`
      : html\`<\${BServerCallScreen} number="15889999" snapshot=\${mkSnap("awaiting_user", "직접 입력이 필요합니다.", { pendingRequest: activeRequest })}
          controls=\${controls} onEnd=\${() => window.secureCommands.push({ type: "end" })} onBack=\${() => {}}
          secureInput=\${{ canSend: true, retryVersion: 0,
            onKey: (requestId, key) => window.secureCommands.push({ type: "key", requestId, key }),
            onDone: requestId => window.secureCommands.push({ type: "done", requestId }) }}/>
        \`}</div></div><//>\`;
}
render(html\`<\${Fixture}/>\`, document.getElementById("root"));`;

let browser, server, origin;
before(async () => {
  let source = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const entry = 'render(html`<${Root}/>`, document.getElementById("root"));';
  assert(source.includes(entry));
  source = source.replace(entry, fixture);
  const fonts = new Map();
  if (process.env.WEB_TEST_NOTO_FONT) {
    fonts.set('/noto.ttf', await readFile(process.env.WEB_TEST_NOTO_FONT));
    source = source.replace('</head>', '<style>@font-face{font-family:"Noto Sans KR";src:url("/noto.ttf");font-weight:100 900;font-display:block}</style></head>');
  }
  server = createServer((req, res) => {
    if (req.method === 'GET' && fonts.has(req.url)) { res.writeHead(200, { 'Content-Type': 'font/ttf' }); return res.end(fonts.get(req.url)); }
    if (req.method === 'GET' && req.url === '/app') { res.writeHead(200, { 'Content-Type': 'text/html;charset=utf-8' }); return res.end(source); }
    res.writeHead(403); res.end();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = process.env.WEB_TEST_BROWSER === 'webkit' ? await webkit.launch({ timeout: 10000 })
    : await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH, timeout: 10000 });
});
after(async () => { await browser?.close(); if (server?.listening) await new Promise(resolve => server.close(resolve)); });

async function open({ viewport = { width: 390, height: 844 }, large = false, demo = false, reducedMotion = 'no-preference' } = {}) {
  const context = await browser.newContext({ viewport, reducedMotion });
  const page = await context.newPage(), errors = [], calls = [];
  await context.route('**/*', route => {
    const url = route.request().url();
    if (!url.startsWith(origin) || url.includes('/v1/') || route.request().method() !== 'GET') { calls.push(url); return route.abort(); }
    return route.continue();
  });
  await page.addInitScript(() => {
    window.keyAnimations = [];
    const animate = Element.prototype.animate;
    Element.prototype.animate = function (frames, options) {
      if (this.matches('button.key')) window.keyAnimations.push({ frames, options });
      return animate.call(this, frames, options);
    };
  });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(origin + '/app', { waitUntil: 'domcontentloaded' });
  await page.evaluate(patch => window.fxSet(patch), { large, demo });
  await page.evaluate(() => document.fonts.ready.then(() => true));
  await page.getByRole('button', { name: '키 #', exact: true }).waitFor();
  return { page, context, errors, calls };
}

test('입력 배지·지우기·중복 설명 없이 개인정보 안내를 두 문단으로 표시한다', async () => {
  const { page, context, errors } = await open();
  try {
    assert.equal(await page.getByText('입력해 주세요', { exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: /지우기|⌫/ }).count(), 0);
    assert.equal(await page.getByText(/누르는 번호는 바로 ARS/).count(), 0);
    const instruction = page.getByText('휴대폰 번호를 직접 눌러주세요.', { exact: true });
    const privacy = page.getByText('입력하신 번호는 저장되지 않습니다.', { exact: true });
    const a = await instruction.boundingBox(), b = await privacy.boundingBox();
    assert(a && b && b.y >= a.y + a.height - 1, '저장하지 않는다는 안내는 별도 줄이어야 한다');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 768, height: 1024 }, { width: 1024, height: 768 }]) {
  for (const large of [false, true]) for (const demo of [false, true]) {
    test(`12개 키·입력 완료·종료를 스크롤 없이 표시 (${viewport.width}x${viewport.height}, ${large ? '큰글씨' : '기본'}, ${demo ? '데모' : '서버'})`, async () => {
      const { page, context, errors } = await open({ viewport, large, demo });
      try {
        const result = await page.evaluate(() => {
          const app = document.querySelector('.app').getBoundingClientRect();
          return [...document.querySelectorAll('.keypad button, .call-view .fbtn.wide, .call-view button[aria-label="종료"]')].map(el => {
            const r = el.getBoundingClientRect();
            let clipped = false;
            for (let p = el.parentElement; p; p = p.parentElement) {
              const style = getComputedStyle(p), b = p.getBoundingClientRect();
              if (['auto', 'scroll', 'hidden', 'clip'].includes(style.overflowY) && (r.top < b.top - 1 || r.bottom > b.bottom + 1)) clipped = true;
            }
            return { name: el.getAttribute('aria-label') || el.textContent.trim(), width: r.width, height: r.height,
              visible: r.top >= app.top - 1 && r.bottom <= app.bottom + 1 && !clipped };
          });
        });
        assert.equal(result.length, 14);
        assert(result.every(r => r.visible), JSON.stringify(result.filter(r => !r.visible)));
        assert(result.every(r => r.width >= 44 && r.height >= 44), '최소 44px 터치 영역');
        const regions = await page.evaluate(() => ['.secure-header', '.secure-copy', '.secure-count', '.secure-input-panel .keypad', '.secure-submit', 'button[aria-label="종료"]'].map(selector => {
          const r = document.querySelector(selector).getBoundingClientRect(); return { selector, top: r.top, bottom: r.bottom };
        }));
        for (let i = 1; i < regions.length; i++) assert(regions[i].top >= regions[i - 1].bottom - 1, JSON.stringify(regions));
        assert.equal(await page.getByRole('button', { name: /지우기|⌫/ }).count(), 0);
        assert.equal(await page.getByText('입력해 주세요', { exact: true }).count(), 0);
        assert.deepEqual(errors, []);
        if (process.env.WEB_KEYPAD_SCREENSHOTS && !demo) {
          await mkdir(process.env.WEB_KEYPAD_SCREENSHOTS, { recursive: true });
          await page.screenshot({ path: `${process.env.WEB_KEYPAD_SCREENSHOTS}/${viewport.width}x${viewport.height}-${large ? 'large' : 'standard'}.png` });
        }
      } finally { await context.close(); }
    });
  }
}

for (const reducedMotion of ['no-preference', 'reduce']) {
  test(`젤리 효과가 입력을 지연·중복시키지 않고 모션 감소를 존중한다 (${reducedMotion})`, async () => {
    const { page, context, errors } = await open({ reducedMotion });
    try {
      for (const digit of '123') await page.getByRole('button', { name: '키 ' + digit, exact: true }).click();
      assert.deepEqual(await page.evaluate(() => window.secureCommands), [...'123'].map(key => ({ type: 'key', requestId: 'secure_test', key })));
      const animations = await page.evaluate(() => window.keyAnimations);
      assert.equal(animations.length, reducedMotion === 'reduce' ? 0 : 3);
      if (animations.length) assert(animations.every(a => a.frames.length >= 3 && a.options.duration <= 350));
      await page.getByRole('button', { name: '입력 완료', exact: true }).click();
      assert.equal((await page.evaluate(() => window.secureCommands)).filter(a => a.type === 'done').length, 1);
      assert.equal(await page.getByRole('button', { name: '입력 완료', exact: true }).isEnabled(), false);
      assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

test('익명 모드에서는 개인정보 키에 젤리 효과를 주지 않아 누른 위치를 드러내지 않는다', async () => {
  const { page, context } = await open();
  try {
    await page.evaluate(() => window.fxSet({ anonymous: true }));
    await page.getByRole('button', { name: '키 1', exact: true }).click();
    assert.equal(await page.evaluate(() => window.keyAnimations.length), 0);
    assert.equal(await page.getByRole('img', { name: '개인정보 1자리 입력됨', exact: true }).count(), 1);
  } finally { await context.close(); }
});

for (const state of ['warning', 'submitted', 'extra']) {
  test(`긴 안내·상태 갱신 후에도 키패드가 완료·종료 버튼과 겹치지 않는다 (${state})`, async () => {
    const { page, context, errors } = await open({ viewport: { width: 320, height: 568 }, large: true });
    try {
      if (state === 'warning') await page.evaluate(() => window.fxSet({ request: {
        message: "12번을 누른 뒤 '상담 메뉴' 안내 없이 바로 입력 안내가 나왔습니다. 번호가 일부만 들어가 다른 메뉴로 갔을 수 있으니 ARS 안내를 확인한 뒤 입력해 주세요. 휴대폰 번호를 직접 눌러 주세요. 입력하신 번호는 저장되지 않습니다."
      } }));
      else if (state === 'extra') await page.evaluate(() => window.fxSet({ extra: true }));
      else {
        await page.getByRole('button', { name: '키 1', exact: true }).click();
        await page.getByRole('button', { name: '입력 완료', exact: true }).click();
        await page.getByText('입력을 전달했어요. 응답을 기다려 주세요.', { exact: true }).waitFor();
      }
      const rects = await page.evaluate(() => ['.secure-input-panel .keypad', '.secure-submit', 'button[aria-label="종료"]'].map(selector => {
        const b = document.querySelector(selector).getBoundingClientRect(); return { selector, top: b.top, bottom: b.bottom };
      }));
      for (let i = 1; i < rects.length; i++) assert(rects[i].top >= rects[i - 1].bottom - 1, JSON.stringify(rects));
      const keys = page.locator('.secure-input-panel .key');
      assert.equal(await keys.count(), 12);
      for (const key of await keys.all()) assert(await key.isVisible());
      assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}
