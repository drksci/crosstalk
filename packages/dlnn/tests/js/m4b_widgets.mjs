// usage: node tests/js/m4b_widgets.mjs [baseUrl]  -> loads every M4b widget (hero, W5, W6, W3, W7, W9), reports console errors, writes screenshots
// into docs/artifacts/article/_screens/m4b-*.png (1440 and 390 px, light and dark for the board / arena / degrade shots).
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const base = process.argv[2] || "http://localhost:8765/article/index.html";
const out = resolve(root, "docs/artifacts/article/_screens");
const browser = await chromium.launch({ args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const errors = [], info = {};
for (const [name, w, h] of [["desktop", 1440, 900], ["mobile", 390, 844]]) for (const scheme of ["light", "dark"]) {
  const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: scheme, deviceScaleFactor: name === "mobile" ? 2 : 1 });
  const page = await ctx.newPage(); const tag = `${name}/${scheme}`;
  page.on("console", (m) => { if ((m.type() === "error" || m.type() === "warning") && !/GL Driver Message/.test(m.text())) errors.push(`${tag}: ${m.type()}: ${m.text()}`); });
  page.on("pageerror", (e) => errors.push(`${tag}: pageerror: ${e.message}`));
  page.on("requestfailed", (r) => errors.push(`${tag}: requestfailed ${r.url()}`));
  console.error("STEP "+tag); await page.goto(base, { waitUntil: "networkidle" });
  const shot = async (sel, file, wait = 1500) => { await page.locator(sel).first().scrollIntoViewIfNeeded(); await page.waitForTimeout(wait); await page.locator(sel).first().screenshot({ path: `${out}/m4b-${name}-${scheme}-${file}.png` }); };
  await shot("#hero-sketch", "hero", 1800);
  await shot("#w5-degrade", "degrade", 1200);
  await page.locator("#w6-board").scrollIntoViewIfNeeded();
  await page.waitForFunction(() => { const p = document.querySelector(".bd-prog"); return !p || p.hidden || /unavailable|Could not/.test(p.textContent); }, null, { timeout: 90000 }).catch(() => {});
  info[tag] = { boardProg: await page.evaluate(() => (document.querySelector(".bd-prog") || {}).textContent, null) };
  await shot("#w6-board", "board", 2500);
  if (scheme === "light") { await page.getByRole("button", { name: "Bottom" }).click(); await page.waitForTimeout(1800); await shot("#w6-board .bd-canvas", "board-bottom", 500); await page.getByRole("button", { name: "Auto-turn" }).click(); }
  await shot("#w7-stitch", "stitch", 800); await shot("#w9-scale", "scale", 800);
  if (scheme === "light") {
    await page.locator("#w3-sandbox").scrollIntoViewIfNeeded(); await page.waitForTimeout(500);
    await page.click("#sb-run"); await page.waitForFunction(() => /Done in|failed|pending/i.test(document.querySelector(".sb-msg")?.textContent || ""), null, { timeout: 180000 }).catch(() => {});
    info[tag].sandbox = await page.evaluate(() => document.querySelector(".sb-msg")?.textContent);
    await shot("#w3-sandbox", "sandbox", 500);
  }
  await shot("#w0-arena", "arena", 2500);
  await ctx.close();
}
await browser.close();
console.log(JSON.stringify({ errors, info }, null, 1));
process.exit(errors.length ? 1 : 0);
