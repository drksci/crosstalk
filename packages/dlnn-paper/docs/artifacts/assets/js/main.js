// main.js: page boot. Theme toggle, section nav (scroll-spy, mobile menu), evidence chips, lazy widget loading.
import { store, loadD3 } from "./common.js";
import { resolveChips } from "./chips.js";

const $ = (s, r = document) => r.querySelector(s);

// ---- theme
const root = document.documentElement;
function currentTheme() { const t = root.getAttribute("data-theme"); if (t) return t; return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"; }
$("#themebtn").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next); store.set("dlnn-theme", next);
  window.dispatchEvent(new Event("dlnn-theme"));
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => window.dispatchEvent(new Event("dlnn-theme")));

// ---- nav
const nav = $("#secnav"), secs = [...document.querySelectorAll("section.sec[data-nav]")];
for (const s of secs) { const a = document.createElement("a"); a.href = "#" + s.id; a.textContent = s.dataset.nav; nav.append(a); }
const top = $("#topbar"), mb = $("#menubtn");
mb.addEventListener("click", () => { const o = top.dataset.open !== "true"; top.dataset.open = o; mb.setAttribute("aria-expanded", o); });
nav.addEventListener("click", () => { top.dataset.open = "false"; mb.setAttribute("aria-expanded", "false"); });
const links = [...nav.children];
const spy = new IntersectionObserver((es) => {
  for (const e of es) if (e.isIntersecting) { links.forEach((l) => l.removeAttribute("aria-current")); const l = links.find((x) => x.hash === "#" + e.target.id); if (l) { l.setAttribute("aria-current", "true"); if (window.innerWidth > 900) l.scrollIntoView({ block: "nearest", inline: "center" }); } }
}, { rootMargin: "-30% 0px -60% 0px" });
secs.forEach((s) => spy.observe(s));

// ---- prose: content/sections.json = {section_id: html}, written by the prose worker; absent means the placeholder prose stays
fetch(new URL("content/sections.json", import.meta.url.replace("/assets/js/", "/article/"))).then((r) => (r.ok ? r.json() : null)).catch(() => null).then((sec) => {
  if (sec && typeof sec === "object") {
    for (const [id, html] of Object.entries(sec)) {
      const s = document.getElementById(id); if (!s || s.tagName !== "SECTION") continue;
      let p = s.querySelector(":scope > .prose");
      if (!p) { p = document.createElement("div"); p.className = "prose"; const h = s.querySelector(".sec-head"); h ? h.after(p) : s.prepend(p); }
      p.innerHTML = String(html);
    }
  }
  resolveChips();
});

// ---- lazy widgets
const loaders = {
  arena: () => import("./arena-ui.js").then((m) => m.mountArena(document.getElementById("w0-arena"))),
  timeline: () => loadD3().then(() => import("./viz.js")).then((m) => m.mountTimeline(document.getElementById("w8-timeline"))),
  taps: () => loadD3().then(() => import("./viz.js")).then((m) => m.mountTaps(document.getElementById("w1-taps"))),
  infer: () => import("./viz.js").then((m) => m.mountInference(document.getElementById("w2-infer"))),
  hero: () => import("./hero.js").then((m) => m.mountHero(document.getElementById("hero-sketch"))),
  mercury: () => import("./hero.js").then((m) => m.mountMercury(document.getElementById("merc-sketch"))),
  degrade: () => import("./degrade.js").then((m) => m.mountDegrade(document.getElementById("w5-degrade"))),
  board: () => import("./board.js").then((m) => m.mountBoard(document.getElementById("w6-board"))),
  sandbox: () => import("./sandbox.js").then((m) => m.mountSandbox(document.getElementById("w3-sandbox"))),
  stitch: () => import("./scale.js").then((m) => m.mountStitch(document.getElementById("w7-stitch"))),
  scale: () => import("./scale.js").then((m) => m.mountScale(document.getElementById("w9-scale"))),
  refs: () => import("./viz.js").then((m) => m.mountRefs(document.getElementById("refs-list"))),
};
const io = new IntersectionObserver((es) => {
  for (const e of es) {
    if (!e.isIntersecting) continue;
    io.unobserve(e.target);
    const k = e.target.dataset.widget;
    Promise.resolve(loaders[k] && loaders[k]()).catch((err) => {
      console.error(k, err);
      e.target.replaceChildren(Object.assign(document.createElement("div"), { className: "err", textContent: "This widget failed to load: " + err.message }));
    });
  }
}, { rootMargin: "700px 0px" });
document.querySelectorAll("[data-widget]").forEach((w) => io.observe(w));
window.__dlnnReady = true;
