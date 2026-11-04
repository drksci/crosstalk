// Headless Chromium: KiCanvas opens demo .kicad_sch + .kicad_pcb; three.js loads demo.glb. Screenshots -> ../shots/
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '../../..'); const SHOTS = path.join(__dirname, '..', 'shots'); fs.mkdirSync(SHOTS, { recursive: true });
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary' };
const srv = http.createServer((q, r) => {
  const f = path.join(ROOT, decodeURIComponent(q.url.split('?')[0]));
  if (!f.startsWith(ROOT) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); fs.createReadStream(f).pipe(r);
}).listen(0, '127.0.0.1', async () => {
  const base = `http://127.0.0.1:${srv.address().port}`;
  const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const out = {};
  // three.js + GLB
  let page = await browser.newPage({ viewport: { width: 1200, height: 700 } });
  const errs = []; page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 160)); });
  await page.goto(base + '/hw/p0/browser/three.html');
  await page.waitForFunction('window.__done === true', null, { timeout: 60000 });
  out.three = await page.evaluate('window.__result'); await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(SHOTS, 'three_glb.png') }); await page.close();
  // KiCanvas
  page = await browser.newPage({ viewport: { width: 1200, height: 1000 } });
  page.on('response', r => { if (r.status() === 404) errs.push('404 ' + r.url()); }); page.on('pageerror', e => errs.push('kc:' + e.message)); page.on('console', m => { if (m.type() === 'error') errs.push('kc:' + m.text().slice(0, 160)); });
  await page.goto(base + '/hw/p0/browser/kicanvas.html');
  await page.waitForTimeout(6000);
  out.kicanvas = await page.evaluate(() => [...document.querySelectorAll('kicanvas-embed')].map(e => ({ id: e.id, loaded: e.loaded ?? null, hasShadow: !!e.shadowRoot, canvases: e.shadowRoot ? e.shadowRoot.querySelectorAll('canvas').length : 0 })));
  await page.screenshot({ path: path.join(SHOTS, 'kicanvas_sch_pcb.png') });
  out.errors = errs; console.log(JSON.stringify(out, null, 1));
  await browser.close(); srv.close();
});
