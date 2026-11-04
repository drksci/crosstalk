// chips.js: evidence chips. <a class="chip" data-claim="p1.w32.acc.C" [data-show="value"]> resolves against data/claims.json.
// The chip shows the claim id (or its value with data-show="value"); hover/focus shows the claim, its fidelity label and source.
import { loadJSON, makeTooltip } from "./common.js";

const fmt = (c) => {
  const v = c.value;
  if (typeof v !== "number") return String(v);
  if (c.unit === "fraction") return v.toFixed(3);
  if (Math.abs(v) < 1e-3 && v !== 0) return v.toExponential(2);
  return Number.isInteger(v) ? String(v) : v.toPrecision(4);
};

export async function resolveChips(root = document) {
  let claims = {};
  try { const j = await loadJSON("claims.json"); for (const c of j.claims) claims[c.id] = c; } catch (e) { /* chips stay unresolved */ }
  const tip = makeTooltip("chip-pop");
  root.querySelectorAll("[data-claim]").forEach((a) => {
    if (a.dataset.resolved) return;
    a.dataset.resolved = "1";
    const id = a.dataset.claim, c = claims[id];
    a.classList.add("chip");
    if (!c) { a.dataset.missing = "1"; a.textContent = id + " (unresolved)"; a.title = "claim id not found in claims.json"; return; }
    a.dataset.fid = c.fidelity;
    a.replaceChildren();
    if (a.dataset.show === "value") { const b = document.createElement("b"); b.textContent = fmt(c); a.append(b); }
    else a.append(id);
    a.setAttribute("aria-label", `${id}: ${c.text} Value ${fmt(c)}. ${c.fidelity}.`);
    const html = () => `<b>${id}</b> &middot; <b>${fmt(c)}</b> ${c.unit === "fraction" ? "" : c.unit}<br>${c.text}<br><span style="opacity:.8">${c.fidelity} &middot; ${c.status}</span><br><code>${c.source.file} : ${c.source.key}</code>`;
    const place = () => { const r = a.getBoundingClientRect(); tip.show(html(), r.left, r.bottom); };
    a.addEventListener("mouseenter", place); a.addEventListener("focus", place);
    a.addEventListener("mouseleave", tip.hide); a.addEventListener("blur", tip.hide);
    a.addEventListener("click", (e) => { if (!a.getAttribute("href")) e.preventDefault(); });
    if (!a.getAttribute("href")) a.setAttribute("tabindex", "0");
  });
  return claims;
}
