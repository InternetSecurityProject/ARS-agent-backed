import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createPreviewServer } from '../preview.mjs';

test('로컬 미리보기는 화면만 제공하고 세션 생성·서버 API·파일 접근을 차단한다', async () => {
  const server = createPreviewServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  try {
    assert.equal((await fetch(origin + '/app')).status, 200);
    for (const [path, method] of [['/v1/sessions', 'POST'], ['/v1/health', 'GET'], ['/app', 'POST'], ['/.env', 'GET']]) {
      assert.equal((await fetch(origin + path, { method })).status, 403);
    }
  } finally { await new Promise(resolve => server.close(resolve)); }
});
