// usage: node tests/js/shots.mjs [baseUrl]  -> screenshots into docs/artifacts/article/_screens, console errors, page weight
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { chromium } = require("playwright");
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const base = process.argv[2] || "http://localhost:8765/article/index.html";
const out = resolve(root, "docs/artifacts/article/_screens");
const browser = await chromium.launch();
const summary = { errors: [], weights: {} };
for (const [name, w, h] of [["desktop", 1440, 900], ["mobile", 390, 844]]) {
  for (const scheme of ["light", "dark"]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: h }, colorScheme: scheme, deviceScaleFactor: name === "mobile" ? 2 : 1, hasTouch: name === "mobile" });
    const page = await ctx.newPage();
    let bytes = 0, reqs = 0;
    page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") summary.errors.push(`${name}/${scheme}: ${m.type()}: ${m.text()}`); });
    page.on("pageerror", (e) => summary.errors.push(`${name}/${scheme}: pageerror: ${e.message}`));
    page.on("requestfailed", (r) => summary.errors.push(`${name}/${scheme}: requestfailed ${r.url()}`));
    page.on("response", async (r) => { try { const b = await r.body(); bytes += b.length; reqs++; } catch (e) {} });
    await page.goto(base, { waitUntil: "networkidle" });
    await page.waitForTimeout(1500);
    summary.weights[`${name}/${scheme}/initial`] = { kb: Math.round(bytes / 1024), requests: reqs };
    await page.screenshot({ path: `${out}/${name}-${scheme}-top.png` });
    // scroll through the whole page so lazy widgets load
    const H = await page.evaluate(() => document.documentElement.scrollHeight);
    for (let y = 0; y < H; y += 700) { await page.evaluate((yy) => window.scrollTo({top:yy,behavior:'instant'}), y); await page.waitForTimeout(250); }
    await page.waitForTimeout(1000);
    summary.weights[`${name}/${scheme}/after-scroll`] = { kb: Math.round(bytes / 1024), requests: reqs };
    await page.evaluate(() => window.scrollTo(0, 0));
    for (const id of ["arena", "lineage", "idea", "maths", "refs"]) {
      await page.evaluate((i) => document.getElementById(i).scrollIntoView({behavior:'instant'}), id);
      await page.waitForTimeout(900);
      await page.screenshot({ path: `${out}/${name}-${scheme}-${id}.png` });
    }
    if (scheme === "light") await page.screenshot({ path: `${out}/${name}-${scheme}-full.png`, fullPage: true });
    const sw = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    if (sw) summary.errors.push(`${name}/${scheme}: horizontal page scroll`);
    await ctx.close();
  }
}
await browser.close();
console.log(JSON.stringify(summary, null, 1));
