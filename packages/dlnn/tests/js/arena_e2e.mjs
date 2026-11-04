// usage: node tests/js/arena_e2e.mjs [baseUrl]
// Drives the arena in headless Chromium: all four input sources (mic through Chromium's fake audio device), step mode, the
// challenge, console errors, and the frame rate unthrottled and with the CPU throttled 4x (CDP Emulation.setCPUThrottlingRate).
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const base = process.argv[2] || "http://localhost:8765/article/index.html";
const out = resolve(root, "docs/artifacts/article/_screens");
const res = { errors: [], checks: {} };
const browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"] });
const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 }, permissions: ["microphone"] });
const page = await ctx.newPage();
page.on("console", (m) => { if (["error", "warning"].includes(m.type())) res.errors.push(m.type() + ": " + m.text()); });
page.on("pageerror", (e) => res.errors.push("pageerror: " + e.message));
await page.goto(base + "?debug", { waitUntil: "networkidle" });
await page.evaluate(() => document.getElementById("w0-arena").scrollIntoView({ behavior: "instant" }));
await page.waitForFunction(() => window.__arena && window.__arena.stats.ready, null, { timeout: 15000 });
await page.evaluate(() => document.getElementById("w0-arena").scrollIntoView({ behavior: "instant" }));
const state = () => page.evaluate(() => { const a = window.__arena.arena; return { n: a.nSamples, cls: a.classifications, filled: a.filled, truth: a.truth, fps: window.__arena.stats.fps, preds: a.models.map((m) => [m.id, m.pred, m.totalAll, +(a.accuracy(m) ?? -1).toFixed(2)]), env: a.source.env, paused: a.paused, skipped: a.skipped }; });
await page.waitForTimeout(4000);
let s = await state();
res.checks.cycle = { classifications: s.cls, labelled_scored: s.preds.map((p) => p[0] + ":" + p[2]).join(" "), acc: s.preds.map((p) => p[0] + ":" + p[3]).join(" ") };
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-cycle-desktop.png` });

// step mode
await page.click("#ar-play"); const n0 = (await state()).n; await page.click("#ar-step"); await page.click("#ar-step"); await page.click("#ar-step");
res.checks.step = { samples_advanced: (await state()).n - n0, paused: (await state()).paused };
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-step-desktop.png` });
await page.click("#ar-play");

// key
await page.click("#ar-tab-key"); await page.waitForTimeout(1500);
const idle = await page.evaluate(() => { const a = window.__arena.arena; return Math.hypot(a.cleanI, a.cleanQ); });
await page.keyboard.down("Space"); await page.waitForTimeout(1200);
const held = await page.evaluate(() => { const a = window.__arena.arena; return { env: Math.hypot(a.cleanI, a.cleanQ), down: window.__arena.sources.key.down }; });
await page.waitForTimeout(500);
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-key-desktop.png` });
await page.keyboard.up("Space"); await page.waitForTimeout(800);
const rel = await page.evaluate(() => Math.hypot(window.__arena.arena.cleanI, window.__arena.arena.cleanQ));
res.checks.key = { amplitude_idle: +idle.toFixed(3), amplitude_held: +held.env.toFixed(3), key_down_flag: held.down, amplitude_released: +rel.toFixed(3) };
// pointer keying + FSK
await page.click("#ar-tab-key"); await page.click('button[data-m="fsk"]');
const kb = await page.locator("#ar-key").boundingBox(); await page.mouse.move(kb.x + 20, kb.y + 20); await page.mouse.down(); await page.waitForTimeout(600);
res.checks.key_fsk_pointer = await page.evaluate(() => window.__arena.sources.key.down); await page.mouse.up();

// mic (Chromium fake audio device: periodic beep)
await page.click("#ar-tab-mic"); await page.click("#ar-mic-en"); await page.waitForTimeout(2500);
const mic1 = await page.evaluate(() => { const m = window.__arena.sources.mic; return { state: m.state, error: m.error }; });
let maxLevel = 0, pitches = [], ons = 0;
for (let i = 0; i < 40; i++) { const r = await page.evaluate(() => { const m = window.__arena.sources.mic; return { l: m.level, p: m.pitch, on: m.on }; }); maxLevel = Math.max(maxLevel, r.l); if (r.p) pitches.push(Math.round(r.p)); if (r.on) ons++; await page.waitForTimeout(100); }
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-mic-desktop.png` });
await page.click('#ar-tab-mic'); await page.click('.seg button[data-m="fsk"]'); await page.waitForTimeout(2500);
res.checks.mic = { state: mic1.state, error: mic1.error, max_rms: +maxLevel.toFixed(3), samples_gate_on: ons + "/40", detected_pitch_hz: pitches.slice(0, 6), fsk_levels_used: await page.evaluate(() => window.__arena.sources.mic.fsk) };

// draw
await page.click("#ar-tab-draw"); await page.waitForTimeout(500);
const pad = await page.locator("#ar-pad").boundingBox();
await page.mouse.move(pad.x + pad.width * 0.3, pad.y + pad.height * 0.5); await page.mouse.down();
for (let t = 0; t <= 24; t++) await page.mouse.move(pad.x + pad.width * (0.3 + 0.4 * t / 24), pad.y + pad.height * (0.5 + 0.35 * Math.sin(t / 24 * 6.28)), { steps: 2 });
await page.mouse.up(); await page.waitForTimeout(2500);
res.checks.draw = await page.evaluate(() => ({ points: window.__arena.sources.draw.pts.length, path: !!window.__arena.sources.draw.path }));
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-draw-desktop.png` });

// challenge on the cycling stream with a hard channel
await page.click("#ar-tab-cycle"); await page.evaluate(() => { const i = document.getElementById("ar-snr"); i.value = 3; i.dispatchEvent(new Event("input")); const t = document.getElementById("ar-tol"); t.value = 1.5; t.dispatchEvent(new Event("input")); });
await page.click("#ch-start"); await page.waitForTimeout(9000);
res.checks.challenge = await page.evaluate(() => ({ score: window.__arena.arena.challenge.score, active: window.__arena.arena.challenge.active }));
await page.locator("#w0-arena").screenshot({ path: `${out}/arena-challenge-desktop.png` });
await page.click("#ch-start");
await page.evaluate(() => { const i = document.getElementById("ar-snr"); i.value = 12; i.dispatchEvent(new Event("input")); const t = document.getElementById("ar-tol"); t.value = 0; t.dispatchEvent(new Event("input")); });

// frame rate
async function fps(label) {
  await page.evaluate(() => { window.__arena.arena.models.forEach((m) => (m.enabled = true)); window.__arena.stats.workMsMax = 0; window.__arena.stats.workMsAvg = 0; window.__f = { n: 0, t0: performance.now(), worst: 0, last: performance.now(), dts: [] }; const loop = (t) => { const w = window.__f; w.n++; w.worst = Math.max(w.worst, t - w.last); w.dts.push(t - w.last); w.last = t; if (!w.stop) requestAnimationFrame(loop); }; requestAnimationFrame(loop); });
  const c0 = (await state()).cls; await page.waitForTimeout(6000);
  const r = await page.evaluate(() => { const w = window.__f; w.stop = true; const d = w.dts.slice(1).sort((a, b) => a - b); return { fps: w.n / ((performance.now() - w.t0) / 1000), worst_frame_ms: w.worst, median_ms: d[d.length >> 1], p95_ms: d[Math.floor(d.length * 0.95)], over33: d.filter((x) => x > 33.5).length, n: d.length }; });
  const s2 = await state();
  res.checks["fps_" + label] = { fps: +r.fps.toFixed(1), worst_frame_ms: +r.worst_frame_ms.toFixed(1), median_frame_ms: +r.median_ms.toFixed(1), p95_frame_ms: +r.p95_ms.toFixed(1), frames_over_33ms: r.over33 + "/" + r.n, models_running: s2.preds.length, classifications_in_6s: s2.cls - c0, skipped_windows: s2.skipped, ...(await page.evaluate(() => ({ js_work_ms_per_frame_avg: +window.__arena.stats.workMsAvg.toFixed(2), js_work_ms_per_frame_max: +window.__arena.stats.workMsMax.toFixed(1) }))) };
}
await page.click("#ar-tab-cycle");
await fps("unthrottled_all_models");
const cdp = await ctx.newCDPSession(page); await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
await fps("cpu_4x");
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
res.checks.final = (await state()).fps;
await browser.close();
console.log(JSON.stringify(res, null, 1));
