import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

let browser, server, origin;
before(async () => {
  let html = await readFile(new URL('../../static/app.html', import.meta.url), 'utf8');
  const fonts = new Map();
  for (const [path, file, family, weight] of [
    ['/_test/noto.ttf', process.env.WEB_TEST_NOTO_FONT, 'Noto Sans KR', '100 900'],
    ['/_test/gowun.ttf', process.env.WEB_TEST_GOWUN_FONT, 'Gowun Batang', '400'],
  ]) {
    if (!file) continue;
    fonts.set(path, await readFile(file));
    html = html.replace('</head>', `<style>@font-face{font-family:'${family}';src:url('${path}');font-weight:${weight};font-display:block}</style></head>`);
  }
  server = createServer((req, res) => {
    if (fonts.has(req.url) && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'font/ttf' }); res.end(fonts.get(req.url));
    } else if (req.url === '/app' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(html);
    } else { res.writeHead(403); res.end('Preview only: no backend'); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH });
});
after(async () => { await browser?.close(); await new Promise(resolve => server?.close(resolve)); });

async function open(viewport = { width: 390, height: 844 }, blockedStorage = false) {
  const context = await browser.newContext({ viewport });
  const calls = [], errors = [];
  await context.route('**/*', route => {
    const req = route.request();
    if (req.method() !== 'GET') calls.push(req.url());
    if (!req.url().startsWith(origin) || req.url().includes('/v1/')) return route.abort();
    return route.continue();
  });
  if (blockedStorage) await context.addInitScript(() => {
    Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('Blocked', 'SecurityError'); } });
  });
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(origin + '/app');
  await page.getByRole('button', { name: '실제 ARS', exact: true }).waitFor();
  await page.evaluate(() => document.fonts.ready);
  return { page, context, calls, errors };
}
const size = locator => locator.evaluate(el => parseFloat(getComputedStyle(el).fontSize));
const toggle = page => page.getByRole('button', { name: /큰글씨 보기|기본 글씨로/ });
async function demo(page, scenario, number = '131') {
  await page.getByRole('button', { name: '실제 ARS', exact: true }).click();
  await page.getByRole('button', { name: scenario, exact: true }).click();
  await page.keyboard.type(number);
  await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).click();
  await page.getByRole('textbox').fill('수원 오늘 날씨 알려줘');
}
async function withinScreen(locator, page) {
  const r = await locator.boundingBox();
  assert(r && r.width > 0 && r.height > 0);
  const v = page.viewportSize();
  assert(r.x >= 0 && r.y >= 0 && r.x + r.width <= v.width + 1 && r.y + r.height <= v.height + 1, JSON.stringify(r));
}

test('큰글씨를 키보드로 전환하고 번호와 크기를 유지하며 새로고침 후에도 복원한다', async () => {
  const { page, context, calls, errors } = await open();
  try {
    await page.keyboard.type('07012345678');
    const label = page.getByRole('tab', { name: '연락처' }).locator('.t-label');
    const base = await size(label);
    const digit = await size(page.locator('.key-digit').first());
    assert.equal(await toggle(page).count(), 1, '공통 큰글씨 전환 버튼이 있어야 한다');
    await toggle(page).focus(); await page.keyboard.press('Space');
    assert.equal(await toggle(page).getAttribute('aria-pressed'), 'true');
    assert((await size(label)) >= base * 1.2);
    assert.equal(await size(page.locator('.key-digit').first()), digit);
    await page.getByText('070-1234-5678', { exact: true }).waitFor();
    await page.reload();
    assert.equal(await toggle(page).getAttribute('aria-pressed'), 'true');
    assert((await size(label)) >= base * 1.2);
    await toggle(page).click();
    assert.equal(await size(label), base);
    await page.reload(); assert.equal(await toggle(page).getAttribute('aria-pressed'), 'false');
    assert.deepEqual(calls, []); assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('브라우저 저장소를 쓸 수 없어도 확대와 원상 복귀가 가능하다', async () => {
  const { page, context, errors } = await open(undefined, true);
  try {
    await toggle(page).click(); assert.equal(await toggle(page).getAttribute('aria-pressed'), 'true');
    await toggle(page).click(); assert.equal(await toggle(page).getAttribute('aria-pressed'), 'false');
    assert.deepEqual(errors, []);
  } finally { await context.close(); }
});

test('연락처 제목과 검색 입력도 확대된다', async () => {
  const { page, context } = await open();
  try {
    await page.getByRole('tab', { name: '연락처', exact: true }).click();
    const heading = page.locator('.t-headline');
    const search = page.getByPlaceholder('이름 또는 전화번호 검색');
    const titleSize = await size(heading), inputSize = await size(search);
    await toggle(page).click();
    assert((await size(heading)) >= titleSize * 1.2);
    assert((await size(search)) >= inputSize * 1.2);
  } finally { await context.close(); }
});

test('상단 안전 영역이 있는 화면에서도 큰글씨 버튼이 가려지지 않는다', async () => {
  const { page, context } = await open({ width: 390, height: 844 });
  try {
    await page.addStyleTag({ content: '.app{--safe-top:44px;--safe-left:12px;--safe-right:12px}' });
    const box = await toggle(page).boundingBox();
    assert(box.y >= 44);
    await toggle(page).click(); await withinScreen(toggle(page), page);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 390, height: 844 }, { width: 320, height: 568 }]) {
test(`목적 입력 중 크기를 바꿔도 입력칸과 값이 유지된다 (${viewport.width}px)`, async () => {
  const { page, context, calls } = await open(viewport);
  try {
    await page.keyboard.type('131');
    await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).click();
    const input = page.getByRole('textbox');
    await input.fill('카드 발급 안내');
    await input.evaluate(el => { el.dataset.retained = 'yes'; });
    const base = await size(input);
    await toggle(page).click();
    assert.equal(await input.inputValue(), '카드 발급 안내');
    assert.equal(await input.getAttribute('data-retained'), 'yes');
    assert((await size(input)) >= base * 1.2);
    assert.deepEqual(calls, []);
  } finally { await context.close(); }
});

}

test('종료 화면의 돌아가기 버튼도 확대되고 작은 화면의 대화상자를 닫을 수 있다', async () => {
  const { page, context } = await open({ width: 320, height: 568 });
  try {
    await page.keyboard.type('01012345678');
    await page.getByRole('button', { name: '일반 전화', exact: true }).click();
    await page.getByRole('button', { name: '종료', exact: true }).click();
    const back = page.getByRole('button', { name: '키패드로 돌아가기', exact: true });
    const base = await size(back);
    await toggle(page).click();
    assert((await size(back)) >= base * 1.2);
    await withinScreen(back, page); await back.click();
    await page.getByRole('button', { name: '실제 ARS', exact: true }).click();
    const close = page.getByRole('button', { name: '닫기', exact: true });
    await withinScreen(close, page); await close.click();
    assert.equal(await page.getByRole('dialog').count(), 0);
  } finally { await context.close(); }
});

test('작은 화면의 일반 전화 데모에서도 큰글씨 전환 후 번호를 확인할 수 있다', async () => {
  const { page, context } = await open({ width: 320, height: 568 });
  try {
    await page.keyboard.type('01012345678');
    await page.getByRole('button', { name: '일반 전화', exact: true }).click();
    await toggle(page).click();
    assert.equal(await page.locator('.top-label').count(), 1);
    assert((await page.locator('.top-label').innerText()).includes('010-1234-5678'));
  } finally { await context.close(); }
});

test('긴 목적의 발신 확인창에서도 번호 크기는 유지하고 취소 버튼에 접근할 수 있다', async () => {
  const { page, context, calls } = await open({ width: 320, height: 568 });
  try {
    await page.keyboard.type('07012345678');
    await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).click();
    await page.getByRole('textbox').fill('카드 발급 방법을 알고 싶어요. '.repeat(20));
    await page.getByRole('button', { name: '발신 확인', exact: true }).click();
    const dialog = page.getByRole('dialog');
    const number = dialog.getByText('070-1234-5678', { exact: true });
    const base = await size(number);
    await toggle(page).click();
    assert.equal(await size(number), base);
    const cancel = dialog.getByRole('button', { name: '취소', exact: true });
    await withinScreen(cancel, page); await withinScreen(dialog.getByRole('button', { name: '실제 발신', exact: true }), page);
    await cancel.click(); assert.equal(await dialog.count(), 0);
    assert.deepEqual(calls, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 390, height: 844 }, { width: 1440, height: 900 }]) {
  test(`후보 선택과 민감정보 데모에서 크게 봐도 종료·입력 완료에 접근 가능 (${viewport.width}px)`, async () => {
    const { page, context, calls, errors } = await open(viewport);
    try {
      await toggle(page).click();
      await demo(page, '후보 선택');
      await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor({ timeout: 12000 });
      const option = page.getByRole('button', { name: '오늘·내일·모레 단기예보 1번' });
      assert((await size(option.locator('span').first())) >= 16);
      await option.scrollIntoViewIfNeeded(); await withinScreen(option, page);
      assert.equal(await option.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
      if (viewport.width === 390) {
        await mkdir(new URL('../test-results/', import.meta.url), { recursive: true });
        await page.screenshot({ path: fileURLToPath(new URL('../test-results/large-text-choice.png', import.meta.url)) });
        await toggle(page).click();
        await page.screenshot({ path: fileURLToPath(new URL('../test-results/default-text-choice.png', import.meta.url)) });
        await toggle(page).click();
      }
      const end = page.getByRole('button', { name: '종료', exact: true });
      await withinScreen(end, page); await end.click();
      assert.equal(await page.getByText('131', { exact: true }).count(), 1, '종료 화면에서 번호가 중복되면 안 된다');
      await page.getByRole('button', { name: '키패드로 돌아가기', exact: true }).click();
      await page.getByRole('button', { name: /UI 데모 · 후보 선택/, exact: true }).click();
      await page.getByRole('button', { name: '민감정보 입력', exact: true }).click();
      await page.keyboard.type('131');
      await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).click();
      await page.getByRole('textbox').fill('수원 오늘 날씨 알려줘');
      await page.getByText('생년월일 6자리 입력', { exact: true }).waitFor({ timeout: 12000 });
      const keys = page.locator('[data-testid="call-keypad"]');
      for (const digit of '123456') await keys.getByRole('button', { name: '키 ' + digit, exact: true }).click();
      const submit = page.getByRole('button', { name: '입력 완료', exact: true });
      await withinScreen(submit, page); assert.equal(await submit.isEnabled(), true);
      await withinScreen(page.getByRole('button', { name: '종료', exact: true }), page);
      assert.deepEqual(calls, []); assert.deepEqual(errors, []);
    } finally { await context.close(); }
  });
}

for (const viewport of [{ width: 400, height: 688 }, { width: 327, height: 688 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
  test(`큰글씨 정보 카드 공간을 확보하고 종료 원의 위치를 유지한다 (${viewport.width}x${viewport.height})`, async () => {
    const { page, context, calls } = await open(viewport);
    try {
      await demo(page, '후보 선택');
      await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor({ timeout: 12000 });
      const card = page.locator('.card');
      const end = page.getByRole('button', { name: '종료', exact: true }).locator('.round');
      const normalCard = await card.boundingBox(), normalEnd = await end.boundingBox();
      const normalFont = await size(page.locator('.option .t-bodym').first());
      await toggle(page).click();
      const largeCard = await card.boundingBox(), largeEnd = await end.boundingBox();
      assert(largeCard.height >= normalCard.height - 1, JSON.stringify({ normalCard, largeCard }));
      assert((await size(page.locator('.option .t-bodym').first())) >= normalFont * 1.2);
      assert(Math.abs(largeEnd.y - normalEnd.y) <= 3, JSON.stringify({ normalEnd, largeEnd }));
      assert.equal(largeEnd.width, normalEnd.width); assert.equal(largeEnd.height, normalEnd.height);
      const mode = page.locator('.app-toolbar .toolbar-mode');
      assert.equal(await mode.innerText(), 'UI 데모');
      const modeBox = await mode.boundingBox(), toggleBox = await toggle(page).boundingBox();
      assert(Math.abs(modeBox.y + modeBox.height / 2 - toggleBox.y - toggleBox.height / 2) <= 2);
      if (viewport.height === 688) {
        const middle = await page.getByRole('button', { name: '키패드', exact: true }).locator('.round').boundingBox();
        const circleGap = largeEnd.y - middle.y - middle.height;
        assert(circleGap <= 44, JSON.stringify({ circleGap, middle, largeEnd }));
        const endTarget = await page.getByRole('button', { name: '종료', exact: true }).boundingBox();
        for (const name of ['ARS 소리 끄기', '키패드', '오디오']) {
          const button = page.getByRole('button', { name, exact: true });
          const circle = await button.locator('.round').boundingBox();
          assert.equal(circle.width, 60); assert.equal(circle.height, 60);
          const target = await button.boundingBox();
          assert(target.x + target.width <= endTarget.x || target.x >= endTarget.x + endTarget.width || target.y + target.height <= endTarget.y, '보조 버튼과 종료 버튼의 터치 영역이 겹치면 안 된다');
        }
        // 커진 후보는 카드 내부에서 스크롤하며 하단 종료 버튼은 고정한다.
        const last = page.getByRole('button', { name: '중기 10일 예보 5번' });
        await last.scrollIntoViewIfNeeded();
        const lastOption = await last.boundingBox();
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/choice-scroll-${viewport.width}.png`, import.meta.url)) });
        assert(lastOption.y + lastOption.height <= largeCard.y + largeCard.height - 8, JSON.stringify({ message: '마지막 후보에 카드 스크롤로 접근할 수 있어야 한다', largeCard, lastOption }));
        assert(lastOption.y >= largeCard.y);
        await page.screenshot({ path: fileURLToPath(new URL(`../test-results/large-text-layout-${viewport.width}.png`, import.meta.url)) });
      }
      await page.getByRole('button', { name: '중기 10일 예보 5번' }).scrollIntoViewIfNeeded();
      await withinScreen(page.getByRole('button', { name: '종료', exact: true }), page);
      await toggle(page).click();
      const restoredCard = await card.boundingBox(), restoredEnd = await end.boundingBox();
      assert.equal(restoredCard.height, normalCard.height);
      assert.equal(restoredEnd.y, normalEnd.y);
      assert.deepEqual(calls, []);
    } finally { await context.close(); }
  });
}

test('긴 전화번호는 큰글씨 헤더에서 가로로 잘리지 않는다', async () => {
  const { page, context, calls } = await open({ width: 320, height: 568 });
  try {
    await toggle(page).click();
    await demo(page, '후보 선택', '07012345678');
    await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor({ timeout: 12000 });
    const header = page.locator('.call-header-tight');
    const number = header.getByText('070-1234-5678', { exact: true });
    await number.scrollIntoViewIfNeeded(); await withinScreen(number, page);
    assert.equal(await size(number), 28);
    assert.equal(await header.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
    assert.deepEqual(calls, []);
  } finally { await context.close(); }
});

test('기관명을 복원하고 큰글씨에서는 번호·시간 아래 한 줄로 표시한다', async () => {
  const { page, context, calls } = await open();
  try {
    await page.keyboard.type('131');
    await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).click();
    assert.equal(await page.getByText(/기상청 기상콜센터/).count(), 1);
    await page.getByRole('button', { name: '취소', exact: true }).click();
    await page.getByRole('button', { name: '전체 지우기', exact: true }).click();
    await demo(page, '후보 선택');
    await page.getByText('어떤 정보를 원하시나요?', { exact: true }).waitFor({ timeout: 12000 });
    for (const large of [false, true]) {
      if (large) await toggle(page).click();
      assert.equal(await page.getByText(/기상청 기상콜센터/).count(), 1);
      assert.match(await page.locator('.call-number + .sec').innerText(), /\d{2}:\d{2}$/);
      if (large) {
        const institution = page.locator('.call-institution');
        assert.equal(await institution.innerText(), '기상청 기상콜센터');
        const numberBox = await page.locator('.call-number').boundingBox();
        const nameBox = await institution.boundingBox();
        assert(nameBox.y >= numberBox.y + numberBox.height);
        assert.equal(await institution.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
      }
    }
    assert.deepEqual(calls, []);
  } finally { await context.close(); }
});

for (const viewport of [{ width: 320, height: 568 }, { width: 327, height: 688 }, { width: 390, height: 844 }]) {
  test(`010-5555-5555 번호를 기본·큰글씨 키패드와 통화에서 잘림 없이 표시한다 (${viewport.width}px)`, async () => {
    const { page, context, calls } = await open(viewport);
    try {
      await page.keyboard.type('01055555555');
      const dialNumber = page.getByText('010-5555-5555', { exact: true });
      await withinScreen(dialNumber, page);
      await toggle(page).click(); await withinScreen(dialNumber, page);
      await toggle(page).click();
      assert.equal(await page.getByRole('button', { name: 'ARS Agent 전화', exact: true }).count(), 0);
      await page.getByRole('button', { name: '일반 전화', exact: true }).click();
      for (const large of [false, true]) {
        if (large) await toggle(page).click();
        const number = page.locator('.call-number,.top-label').filter({ hasText: '010-5555-5555' });
        assert.equal(await number.count(), 1);
        await withinScreen(number, page);
        assert.equal(await number.evaluate(el => el.scrollWidth <= el.clientWidth + 1), true);
      }
      await page.screenshot({ path: fileURLToPath(new URL(`../test-results/long-phone-${viewport.width}.png`, import.meta.url)) });
      assert.deepEqual(calls, []);
    } finally { await context.close(); }
  });
}
