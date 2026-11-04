// Playwright: run demo_flat.cir in eecircuit-engine, dump waveforms to eecircuit_tran.json
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '../../..');
const srv = http.createServer((q, r) => {
  const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  const ext = path.extname(f); const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
  r.writeHead(200, { 'content-type': types[ext] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r);
}).listen(0, '127.0.0.1', async () => {
  const port = srv.address().port;
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on('console', m => console.log('[console]', m.text().slice(0, 200)));
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  const t0 = Date.now();
  await page.goto(`http://127.0.0.1:${port}/hw/p0/browser/eecircuit.html`);
  await page.waitForFunction('window.__done === true', null, { timeout: 120000 });
  const res = await page.evaluate('window.__result');
  res.wall_ms = Date.now() - t0;
  fs.writeFileSync(path.join(__dirname, '..', 'eecircuit_tran.json'), JSON.stringify(res));
  console.log('vars', res.variableNames.join(' '), 'points', res.numPoints, 'init_ms', res.init_ms.toFixed(0), 'run_ms', res.run_ms.toFixed(0), 'wall_ms', res.wall_ms);
  await browser.close(); srv.close();
});
