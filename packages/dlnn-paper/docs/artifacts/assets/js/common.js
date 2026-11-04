// common.js: shared helpers for the article page (paths, cached JSON loader, lazy d3, small DOM utilities).
export const DATA = new URL("../../data/", import.meta.url);
export const VENDOR = new URL("../../vendor/", import.meta.url);
const cache = new Map();
export function loadJSON(name) {
  if (!cache.has(name)) cache.set(name, fetch(new URL(name, DATA)).then((r) => { if (!r.ok) throw new Error(name + ": HTTP " + r.status); return r.json(); }));
  return cache.get(name);
}
/** Like loadJSON but resolves to null when the file is absent (optional data such as energy_refs.json). */
export const loadOptional = (name) => loadJSON(name).catch(() => null);

let d3p = null;
export function loadD3() {
  if (window.d3) return Promise.resolve(window.d3);
  if (!d3p) d3p = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = new URL("d3/d3.min.js", VENDOR).href;
    s.onload = () => res(window.d3); s.onerror = () => rej(new Error("d3 failed to load"));
    document.head.appendChild(s);
  });
  return d3p;
}


let p5p = null;
export function loadP5() {
  if (window.p5) return Promise.resolve(window.p5);
  if (!p5p) p5p = new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = new URL("p5/p5.min.js", VENDOR).href;
    s.onload = () => res(window.p5); s.onerror = () => rej(new Error("p5 failed to load"));
    document.head.appendChild(s);
  });
  return p5p;
}

export const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : document.createTextNode(c));
  return e;
};
export const reducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
export const isNarrow = () => window.matchMedia && window.matchMedia("(max-width: 640px)").matches;
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const store = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage blocked: fine */ } },
};

/** Floating tooltip positioned near a pointer/target, kept inside the viewport. */
export function makeTooltip(cls = "tt") {
  const t = el("div", { class: cls, role: "tooltip", hidden: true });
  document.body.append(t);
  return {
    show(html, x, y) {
      t.innerHTML = html; t.hidden = false;
      const r = t.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight;
      t.style.left = Math.max(8, Math.min(W - r.width - 8, x + 12)) + "px";
      t.style.top = Math.max(8, Math.min(H - r.height - 8, y + 14)) + "px";
    },
    hide() { t.hidden = true; },
    el: t,
  };
}
export const CLASS_COLORS_NOTE = "class identity is always shown as text";
