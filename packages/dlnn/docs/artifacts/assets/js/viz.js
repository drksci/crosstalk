// viz.js: W8 lineage timeline, W1 network/tap explorer, W2 live inference with a layer-by-layer trace, and the reference list.
// All data comes from data/*.json (the same files the notebooks read). D3 v7 is vendored (vendor/d3) and loaded lazily.

import { loadJSON, el, cssVar, makeTooltip, reducedMotion } from "./common.js";
import { decodeModel, forward, softmax, argmax } from "./tdlnet.js";
import { decodeArenaExamples } from "./models.js";
import { CLASSES } from "./signals.js";

const STRANDS = {
  "delay-memory": "Delay-line memory", "analog-sp": "Analog signal processing", "neural-adaptive": "Neural and adaptive nets", reservoir: "Delay reservoirs",
  "analog-ai": "Analog AI hardware", ternary: "Ternary and low-bit nets", "rf-ml": "RF machine learning",
};
const LANE_ORDER = ["delay-memory", "analog-sp", "neural-adaptive", "reservoir", "analog-ai", "ternary", "rf-ml"];
const CLOSE = new Set(["pour2025", "merton2026", "woo2024", "tank1987"]);

// ================================================================================================ W8 timeline
export async function mountTimeline(host) {
  const d3 = window.d3, lit = await loadJSON("literature.json");
  const refs = Object.fromEntries(lit.references.map((r) => [r.key, r]));
  const events = lit.timeline.map((t) => ({ ...t, ref: refs[t.ref_key] })).sort((a, b) => a.year - b.year);
  const W = 960, laneH = 62, top = 34, left = 150, right = 24, H = top + LANE_ORDER.length * laneH + 10;
  const x = d3.scaleLinear().domain([1938, 2028]).range([left, W - right]);
  const pop = el("div", { class: "card-pop", hidden: true, role: "dialog", "aria-label": "Reference" });
  document.body.append(pop);
  let pinned = null;
  const showPop = (d, ev, pin) => {
    const r = d.ref;
    pop.innerHTML = `<div class="y">${d.year} &middot; ${STRANDS[d.strand] || d.strand}${CLOSE.has(d.ref_key) ? " &middot; CLOSE PRIOR WORK" : ""}</div><h5>${d.label}</h5><div>${r.authors} (${r.year}). <i>${r.title}</i>. <span class="src">${r.venue}</span></div><div style="margin-top:6px">${r.relevance}</div><div style="margin-top:6px"><a href="${r.url}" target="_blank" rel="noopener noreferrer">Open source${r.doi ? " (DOI)" : ""}</a> <span class="src">&middot; ${r.verified_source}</span></div>`;
    pop.hidden = false; pop.style.pointerEvents = pin ? "auto" : "none";
    const b = pop.getBoundingClientRect(), vw = innerWidth, vh = innerHeight;
    pop.style.left = Math.max(8, Math.min(vw - b.width - 8, ev.x + 14)) + "px"; pop.style.top = Math.max(8, Math.min(vh - b.height - 8, ev.y + 14)) + "px";
  };
  const hide = () => { pinned = null; pop.hidden = true; };
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Lineage: from delay-line memory to ternary delay-line nets" }), el("span", { class: "w-tag", text: "W8 · " + events.length + " events · " + lit.references.length + " verified references" })),
    el("p", { class: "w-sub", text: "One lane per strand of ideas. Hover or focus a dot for the reference, click to pin the card and open the source. Ringed dots are the closest prior work." }));
  const sc = el("div", { class: "lanes-scroll", tabindex: 0, role: "group", "aria-label": "Timeline, scrolls sideways on narrow screens" });
  host.append(sc);
  const svg = d3.select(sc).append("svg").attr("class", "tl-svg").attr("viewBox", `0 0 ${W} ${H}`);
  LANE_ORDER.forEach((s, i) => {
    const y = top + i * laneH;
    svg.append("rect").attr("x", left - 8).attr("y", y).attr("width", W - left - right + 16).attr("height", laneH - 4).attr("rx", 8).attr("fill", i % 2 ? "transparent" : "var(--paper)");
    svg.append("text").attr("class", "lane-lab").attr("x", 8).attr("y", y + laneH / 2).attr("dy", "0.32em").text(STRANDS[s]);
    svg.append("line").attr("x1", left).attr("x2", W - right).attr("y1", y + laneH / 2 - 2).attr("y2", y + laneH / 2 - 2).attr("stroke", `var(--lane-${i + 1})`).attr("stroke-opacity", 0.35).attr("stroke-dasharray", "1 5");
  });
  svg.append("g").selectAll("g").data(d3.range(1940, 2030, 10)).join("g").call((g) => { g.append("line").attr("x1", (d) => x(d)).attr("x2", (d) => x(d)).attr("y1", top - 6).attr("y2", H - 8).attr("stroke", "var(--rule-soft)"); g.append("text").attr("x", (d) => x(d)).attr("y", 18).attr("text-anchor", "middle").attr("font-family", "var(--font-mono)").text((d) => d); });
  // vertical stagger of events that fall close together in the same lane
  const lastX = {}; const placed = events.map((d) => {
    const lane = LANE_ORDER.indexOf(d.strand), px = x(d.year);
    const key = d.strand; const near = (lastX[key] || []).filter((q) => Math.abs(q.px - px) < 16).length;
    (lastX[key] = lastX[key] || []).push({ px });
    return { ...d, lane, px, py: top + lane * laneH + laneH / 2 - 2 + [0, -15, 15, -26, 26][near % 5] };
  });
  const dots = svg.append("g").selectAll("circle").data(placed).join("circle").attr("class", "dot").attr("cx", (d) => d.px).attr("cy", (d) => d.py)
    .attr("r", (d) => (CLOSE.has(d.ref_key) ? 7 : 5.5)).attr("fill", (d) => `var(--lane-${d.lane + 1})`).attr("tabindex", 0).attr("role", "button")
    .attr("aria-label", (d) => `${d.year}: ${d.label}`);
  dots.filter((d) => CLOSE.has(d.ref_key)).attr("stroke", "var(--ink)").attr("stroke-width", 2.5);
  dots.on("mouseenter", (e, d) => { if (!pinned) showPop(d, e, false); }).on("mousemove", (e, d) => { if (!pinned) showPop(d, e, false); }).on("mouseleave", () => { if (!pinned) pop.hidden = true; })
    .on("focus", (e, d) => { if (!pinned) { const r = e.target.getBoundingClientRect(); showPop(d, { x: r.right, y: r.top }, false); } }).on("blur", () => { if (!pinned) pop.hidden = true; })
    .on("click", (e, d) => { e.stopPropagation(); pinned = d; showPop(d, e, true); })
    .on("keydown", (e, d) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); const r = e.target.getBoundingClientRect(); pinned = d; showPop(d, { x: r.right, y: r.top }, true); } else if (e.key === "Escape") hide(); });
  document.addEventListener("click", (e) => { if (pinned && !pop.contains(e.target)) hide(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
  svg.append("g").selectAll("text.lbl").data(placed.filter((d) => CLOSE.has(d.ref_key))).join("text").attr("x", (d) => d.px + (d.year > 2015 ? -10 : 10)).attr("y", (d) => d.py - 11).attr("text-anchor", (d) => (d.year > 2015 ? "end" : "start")).attr("font-size", 10.5).attr("font-weight", 700).attr("fill", "var(--ink)").text((d) => d.ref_key.replace(/(\d{4})/, " $1"));
  host.append(el("details", { class: "fallback" }, el("summary", { text: "Text version of the timeline" }),
    el("ul", {}, events.map((d) => el("li", {}, `${d.year}, ${STRANDS[d.strand] || d.strand}: ${d.label} `, el("a", { href: d.ref.url, target: "_blank", rel: "noopener noreferrer", text: "[source]" }))))));
}

// ================================================================================================ W1 tap explorer
const TS_US = 0.96; // board T_s (docs/phase2_plan.md section 1.1: 8 x 120 ns), a design value
export async function mountTaps(host) {
  const d3 = window.d3, am = await loadJSON("arena_models.json");
  const entry = am.models.find((m) => m.id === "dlnn"), model = decodeModel(entry.data);
  let layer = 0, selO = 0;
  const tip = makeTooltip("tt");
  host.replaceChildren(el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "The network, tap by tap" }), el("span", { class: "w-tag", text: "W1 · the real trained weights of the arena DLNN · designed, not fabricated" })),
    el("p", { class: "w-sub", text: "Each small matrix is one output line: rows are the input lines, columns the eight taps. Warm is a tap wired as +1, cool is -1, grey is not populated. Hover a cell for its physical meaning on the board." }));
  const tabs = el("div", { class: "seg", role: "tablist", "aria-label": "Layer" }, model.layers.map((l, i) => el("button", { type: "button", role: "tab", "aria-selected": i === 0, "data-l": i, text: `Layer ${i + 1}` })));
  const stats = el("div", { class: "ar-hint", "aria-live": "polite" });
  const grid = el("div", { class: "tapgrid" });
  const detail = el("div", { class: "w1-detail" });
  host.append(el("div", { class: "ar-opts", style: "margin-bottom:10px" }, tabs, stats), grid, detail);
  const pol = (w) => (w > 0 ? "var(--pos)" : w < 0 ? "var(--neg)" : "var(--zero)");
  function draw() {
    const L = model.layers[layer], cell = 11, pad = 2;
    const nz = L.W.reduce((a, w) => a + (w !== 0), 0), np = L.W.reduce((a, w) => a + (w > 0), 0);
    stats.textContent = `${L.I} input lines × ${L.O} outputs × ${L.K} taps = ${L.W.length} slots: ${np} wired +1, ${nz - np} wired −1, ${L.W.length - nz} left off. Delay between taps: ${L.dilation} × T_s.`;
    grid.replaceChildren();
    for (let o = 0; o < L.O; o++) {
      const w = L.K * (cell + pad) + pad, h = L.I * (cell + pad) + pad;
      const btn = el("button", { class: "mini", type: "button", "aria-pressed": o === selO, "aria-label": `Layer ${layer + 1} output ${o}`, "data-o": o });
      const svg = d3.select(btn).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("aria-hidden", "true");
      for (let i = 0; i < L.I; i++) for (let k = 0; k < L.K; k++) {
        const wt = L.W[(o * L.I + i) * L.K + k];
        svg.append("rect").attr("x", pad + k * (cell + pad)).attr("y", pad + i * (cell + pad)).attr("width", cell).attr("height", cell).attr("rx", 2).attr("fill", pol(wt)).attr("opacity", wt ? 1 : 0.28)
          .on("mousemove", (e) => tip.show(`<b>Layer ${layer + 1}, output ${o}</b><br><span class="k">input line</span> ${i}${layer === 0 ? (i ? " (Q)" : " (I)") : ""} &nbsp; <span class="k">tap</span> ${k}<br><span class="k">polarity</span> <b>${wt > 0 ? "+1" : wt < 0 ? "−1" : "0 (off)"}</b><br><span class="k">delay</span> ${k * L.dilation} T_s = ${(k * L.dilation * TS_US).toFixed(2)} µs<br><span class="k">board element</span> LC-ladder tap at section ${8 * k}; ${wt ? "10 kΩ 0402 populated to the " + (wt > 0 ? "+" : "−") + " summing node" : "footprint left unpopulated (DNP)"}`, e.clientX, e.clientY))
          .on("mouseleave", tip.hide);
      }
      btn.append(el("div", { class: "cap" }, el("span", { text: "o" + o }), el("span", { text: Array.from({ length: L.I * L.K }, (_, q) => L.W[o * L.I * L.K + q]).filter(Boolean).length + " taps" })));
      btn.addEventListener("click", () => { selO = o; draw(); });
      grid.append(btn);
    }
    detail.replaceChildren(el("div", { class: "ar-hint", html: `<b>Output o${selO} of layer ${layer + 1}</b> sums ${[...Array(L.I * L.K).keys()].filter((q) => L.W[selO * L.I * L.K + q]).length} wired taps, adds a bias of ${L.bias[selO].toFixed(2)}, scales by 1/V<sub>sat</sub> = ${(1 / L.vsat).toFixed(3)} and passes the result through tanh (the differential-pair soft limiter). Physical delay line, resistor crossbar and diff-pair are <span class="status des"><i>&#9650;</i> designed, not fabricated</span>; the weights shown are <span class="status sim"><i>&#9679;</i> simulated</span> results of training.` }));
  }
  tabs.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; layer = +b.dataset.l; selO = 0; tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", x === b)); draw(); });
  draw();
}

// ================================================================================================ W2 inference
function heat(canvas, y, O, upto, colors) {
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(128, O);
  for (let o = 0; o < O; o++) for (let n = 0; n < 128; n++) {
    const p = (o * 128 + n) * 4;
    if (n >= upto) { img.data[p] = colors.bg[0]; img.data[p + 1] = colors.bg[1]; img.data[p + 2] = colors.bg[2]; img.data[p + 3] = 255; continue; }
    const v = Math.max(-1, Math.min(1, y[o * 128 + n])), a = Math.abs(v), c = v >= 0 ? colors.pos : colors.neg;
    img.data[p] = colors.bg[0] + (c[0] - colors.bg[0]) * a; img.data[p + 1] = colors.bg[1] + (c[1] - colors.bg[1]) * a; img.data[p + 2] = colors.bg[2] + (c[2] - colors.bg[2]) * a; img.data[p + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
const hex = (h) => { h = h.replace("#", ""); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); };

export async function mountInference(host) {
  const [am, exj] = await Promise.all([loadJSON("arena_models.json"), loadJSON("examples_arena.json")]);
  const model = decodeModel(am.models.find((m) => m.id === "dlnn").data), ex = decodeArenaExamples(exj);
  const snrs = [...new Set(ex.snr)].sort((a, b) => a - b);
  let cls = 0, snr = snrs[snrs.length - 2], t = 128, timer = null;
  host.replaceChildren(el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Run the layers yourself" }), el("span", { class: "w-tag", text: "W2 · tier V (ideal math) in JavaScript · simulated" })),
    el("p", { class: "w-sub", text: "Pick a stored test example. The browser runs the exported ternary weights layer by layer; the readout compares the result with the logits numpy computed for the same input." }));
  const selC = el("select", { id: "w2-class", "aria-label": "Modulation" }, CLASSES.map((c, i) => el("option", { value: i, text: c })));
  const selS = el("select", { id: "w2-snr", "aria-label": "SNR" }, snrs.map((s) => el("option", { value: s, selected: s === snr, text: s + " dB" })));
  const scrub = el("input", { type: "range", id: "w2-t", min: 1, max: 128, value: 128, "aria-label": "Time step revealed" });
  const play = el("button", { class: "btn", id: "w2-play", type: "button", text: "Replay" });
  const tlabel = el("span", { class: "mono", id: "w2-tl", text: "t = 128" });
  host.append(el("div", { class: "w2-ctl" }, selC, selS, play, el("label", { style: "display:flex;gap:8px;align-items:center;flex:1 1 200px" }, "Time", scrub, tlabel)));
  const cin = el("canvas", { width: 128, height: 2 });
  const strips = model.layers.map((l) => el("canvas", { width: 128, height: l.O }));
  const sc = el("canvas", { class: "scope", style: "height:110px", id: "w2-iq", role: "img", "aria-label": "Input I and Q" });
  const left = el("div", {}, el("div", { class: "ar-h", text: "Input and layer traces" }), sc, ...strips.map((c, i) => el("div", { class: "w2-strip" }, el("span", { text: "L" + (i + 1) }), c)));
  const probs = el("div", {}), verify = el("div", { class: "verify", id: "w2-verify" });
  host.append(el("div", { class: "w2-out" }, left, el("div", {}, el("div", { class: "ar-h", text: "Class probabilities (softmax of the readout)" }), probs, verify, el("p", { class: "ar-hint", text: "Traces are causal: the first samples see zeros from before the window, so the early columns differ from the settled behaviour." }))));
  function render() {
    const e = ex.y.findIndex((y, i) => y === cls && ex.snr[i] === snr), colors = { pos: hex(cssVar("--pos")), neg: hex(cssVar("--neg")), bg: hex(cssVar("--scope-bg")) };
    const r = forward(model, ex.x[e], 128);
    r.layers.forEach((y, l) => heat(strips[l], y, model.layers[l].O, t, colors));
    // IQ scope
    const d = devicePixelRatio || 1, rc = sc.getBoundingClientRect(); sc.width = Math.round(rc.width * d); sc.height = Math.round(rc.height * d);
    const ctx = sc.getContext("2d"), W = sc.width, H = sc.height; ctx.clearRect(0, 0, W, H); ctx.strokeStyle = cssVar("--scope-grid"); ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
    [[0, "--scope-i"], [128, "--scope-q"]].forEach(([off, col]) => { ctx.strokeStyle = cssVar(col); ctx.lineWidth = 1.5 * d; ctx.beginPath(); for (let n = 0; n < t; n++) { const yv = H / 2 - (ex.x[e][off + n] / 3.2) * (H / 2); n ? ctx.lineTo((n / 127) * W, yv) : ctx.moveTo(0, yv); } ctx.stroke(); });
    const p = softmax(r.out), top = argmax(r.out);
    probs.replaceChildren(...CLASSES.map((c, i) => el("div", { class: "probrow" + (i === top ? " top" : "") }, el("span", { text: c }), el("div", { class: "pb" }, el("i", { style: `width:${(p[i] * 100).toFixed(1)}%` })), el("span", { text: (p[i] * 100).toFixed(1) + "%" }))));
    let maxd = 0; for (let i = 0; i < 8; i++) maxd = Math.max(maxd, Math.abs(r.out[i] - ex.logits.dlnn[e][i]));
    verify.textContent = `JS vs numpy logits, this example: max |diff| = ${maxd.toExponential(1)} · predicted ${CLASSES[top]}, true ${CLASSES[ex.y[e]]}`;
    tlabel.textContent = "t = " + t;
  }
  selC.addEventListener("change", () => { cls = +selC.value; render(); }); selS.addEventListener("change", () => { snr = +selS.value; render(); });
  scrub.addEventListener("input", () => { t = +scrub.value; render(); });
  play.addEventListener("click", () => { clearInterval(timer); if (reducedMotion()) { t = 128; scrub.value = 128; render(); return; } t = 1; timer = setInterval(() => { t += 2; if (t >= 128) { t = 128; clearInterval(timer); } scrub.value = t; render(); }, 30); });
  addEventListener("dlnn-theme", render);
  render();
}

// ================================================================================================ references
export async function mountRefs(host) {
  const lit = await loadJSON("literature.json");
  const rows = [...lit.references].sort((a, b) => a.year - b.year);
  host.replaceChildren(el("div", { class: "w-head" }, el("h3", { class: "w-title", text: `${rows.length} verified references` }), el("span", { class: "w-tag", text: "from data/literature.json" })),
    el("div", { style: "overflow-x:auto" }, el("table", { class: "refs" }, el("tbody", {}, rows.map((r) => el("tr", { id: "ref-" + r.key }, el("td", { text: r.key }), el("td", {}, `${r.authors} (${r.year}). `, el("i", { text: r.title }), `. ${r.venue}. `, el("a", { href: r.url, target: "_blank", rel: "noopener noreferrer", text: r.doi ? "DOI" : "link" }), CLOSE.has(r.key) ? el("b", { text: "  close prior work" }) : null)))))));
}
