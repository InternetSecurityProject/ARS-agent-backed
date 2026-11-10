// Static frontend preview only. Deliberately does not import or proxy the backend.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export function createPreviewServer() {
  return createServer(async (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    const path = new URL(req.url, 'http://localhost').pathname;
    if (!['GET', 'HEAD'].includes(req.method) || !['/', '/app'].includes(path)) {
      res.writeHead(403, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('화면 확인 전용입니다. API 연결과 실제 발신은 차단되어 있어요.');
      return;
    }
    try {
      const html = await readFile(new URL('../static/app.html', import.meta.url));
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(req.method === 'HEAD' ? undefined : html);
    } catch {
      res.writeHead(500, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('먼저 웹 빌드를 실행해 주세요.');
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const port = Number(process.env.WEB_PREVIEW_PORT ?? 8013);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid WEB_PREVIEW_PORT');
  const server = createPreviewServer();
  server.on('error', error => { console.error(error.message); process.exitCode = 1; });
  server.listen(port, '127.0.0.1', () => {
    console.log(`화면 확인: http://127.0.0.1:${port}/app`);
    console.log('상단 실제 ARS → UI 데모 또는 원하는 시나리오를 선택하세요. API·실제 발신은 차단됩니다.');
  });
}
