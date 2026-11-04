// degrade.js: W5 degradation lab (D3). Five tolerance / degradation sliders over the precomputed 5x5x5x5x5 tier-P grid
// (data/sweeps_grid.json); between grid nodes the value is a MULTILINEAR INTERPOLATION and is marked as such. Also the P-tier Monte Carlo
// headline (data/p_tier.json) and the gain-trim finding chart. Simulation only; nothing measured.
import { loadJSON, loadD3, el, cssVar } from "./common.js";

const AX_LABEL = { l_tol_pct: "Inductor tolerance", clock_err_pct: "Clock error", stage_snr_db: "Stage SNR", crosstalk_k: "Crosstalk k", vbe_mismatch_mv: "V_BE mismatch" };
const AX_UNIT = { l_tol_pct: "%", clock_err_pct: "%", stage_snr_db: " dB", crosstalk_k: "", vbe_mismatch_mv: " mV" };
const pct1 = (v) => (100 * v).toFixed(1) + "%";

export async function mountDegrade(host) {
  const [grid, pt] = await Promise.all([loadJSON("sweeps_grid.json"), loadJSON("p_tier.json"), loadD3()]);
  const d3 = window.d3;
  const M = grid.models.tdlnet_b_w8_aware2_fro, axes = M.axes, acc = M.accuracy, nA = axes.length;
  const dims = M.shape, strides = dims.map((_, i) => dims.slice(i + 1).reduce((a, b) => a * b, 1)); // C order, last axis fastest
  // per axis: ascending-sorted values with original index (stage_snr_db is stored descending)
  const sorted = axes.map((a) => a.values.map((v, i) => ({ v, i })).sort((p, q) => p.v - q.v));
  const lo = axes.map((a) => Math.min(...a.values)), hi = axes.map((a) => Math.max(...a.values));
  const state = axes.map((a) => (a.name === "stage_snr_db" ? hi[axes.indexOf(a)] : 0)); // start at the nominal corner (60 dB, no tolerances)
  const bracket = (ax, x) => {
    const s = sorted[ax]; let k = 0;
    while (k < s.length - 2 && x > s[k + 1].v) k++;
    const t = (x - s[k].v) / (s[k + 1].v - s[k].v);
    return { a: s[k].i, b: s[k + 1].i, t: Math.min(1, Math.max(0, t)), exactA: Math.abs(x - s[k].v) < 1e-9, exactB: Math.abs(x - s[k + 1].v) < 1e-9 };
  };
  function interp(x) { // multilinear over the 2^5 surrounding nodes; returns value + whether every coordinate sits on a node
    const br = x.map((v, ax) => bracket(ax, v));
    let val = 0, exact = true;
    for (let m = 0; m < 1 << nA; m++) {
      let w = 1, idx = 0;
      for (let ax = 0; ax < nA; ax++) { const up = (m >> ax) & 1, b = br[ax]; w *= up ? b.t : 1 - b.t; idx += (up ? b.b : b.a) * strides[ax]; }
      if (w > 0) val += w * acc[idx];
    }
    for (const b of br) if (!(b.exactA || b.exactB)) exact = false;
    return { val, exact };
  }
  const nominal = interp(axes.map((a, i) => (a.name === "stage_snr_db" ? hi[i] : 0)));

  // --------------------------------------------------------------------- DOM
  const knobs = el("div", { class: "knobs" }), outs = [];
  axes.forEach((a, i) => {
    const step = (hi[i] - lo[i]) / 200;
    const out = el("output", { text: "" }), inp = el("input", { type: "range", min: lo[i], max: hi[i], step, value: state[i], "aria-label": AX_LABEL[a.name] || a.name, id: "dg-" + a.name });
    inp.addEventListener("input", () => { state[i] = +inp.value; update(); });
    knobs.append(el("div", { class: "knob dg-knob" }, el("label", { for: "dg-" + a.name, text: AX_LABEL[a.name] || a.name, title: a.unit }), inp, out));
    outs.push({ out, inp, i });
  });
  const big = el("div", { class: "dg-big" }), badge = el("span", { class: "dg-badge" });
  const snap = el("button", { class: "btn", type: "button", text: "Snap to nearest grid point" }), reset = el("button", { class: "btn", type: "button", text: "Reset to nominal" });
  snap.addEventListener("click", () => { axes.forEach((a, i) => { state[i] = a.values.reduce((b, v) => (Math.abs(v - state[i]) < Math.abs(b - state[i]) ? v : b)); }); update(true); });
  reset.addEventListener("click", () => { axes.forEach((a, i) => { state[i] = a.name === "stage_snr_db" ? hi[i] : 0; }); update(true); });
  const sm = el("div", { class: "dg-small" });
  const mcBox = el("div", { class: "dg-mc" }), trimBox = el("div", { class: "dg-trim" });
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Degradation lab: how much can the board be wrong?" }), el("span", { class: "w-tag", text: "W5 · simulated (tier P) · designed board, not built" })),
    el("p", { class: "w-sub", text: `Test accuracy of the board-trained width-8 net as five non-idealities grow, from a precomputed ${dims.join("x")} grid of tier-P runs (${M.n_points ? M.n_points : acc.length} points, ${grid.meta.n_examples_per_point} stratified examples each, one fixed golden board). Between grid points the number is interpolated, and says so.` }),
    el("div", { class: "dg-grid" }, el("div", { class: "dg-left" }, knobs, big, el("div", { class: "dg-btns" }, snap, reset)), sm),
    el("div", { class: "dg-row2" }, mcBox, trimBox),
    el("p", { class: "w-foot", text: grid.note + " Accuracy here is on a 800-example subset, so the standard error is about 1.2 points; differences smaller than that are not meaningful. The effect of gain level is the opposite: see the trim chart." }),
  );

  const cs = () => ({ ink: cssVar("--ink"), muted: cssVar("--muted"), rule: cssVar("--rule"), acc: cssVar("--accent"), pos: cssVar("--pos"), neg: cssVar("--neg"), bad: cssVar("--bad"), surf: cssVar("--surface-2") });

  // --------------------------------------------------------------------- small multiples (one per axis, others held at the sliders)
  const cells = axes.map((a, ax) => {
    const wrap = el("div", { class: "dg-cell" });
    sm.append(wrap);
    const svg = d3.select(wrap).append("svg").attr("viewBox", "0 0 260 150").attr("role", "img").attr("aria-label", `Accuracy against ${AX_LABEL[a.name]} with the other sliders held`);
    return { ax, svg, wrap };
  });
  const X0 = 38, X1 = 248, Y0 = 20, Y1 = 118;
  function drawCells() {
    const c = cs();
    for (const { ax, svg } of cells) {
      svg.selectAll("*").remove();
      const a = axes[ax], xs = d3.scaleLinear([lo[ax], hi[ax]], [X0, X1]), ys = d3.scaleLinear([0, 1], [Y1, Y0]);
      svg.append("text").attr("x", X0).attr("y", 12).attr("fill", c.ink).attr("class", "dg-t").text(AX_LABEL[a.name]);
      for (const v of [0, 0.5, 1]) { svg.append("line").attr("x1", X0).attr("x2", X1).attr("y1", ys(v)).attr("y2", ys(v)).attr("stroke", c.rule).attr("stroke-width", 0.6); svg.append("text").attr("x", X0 - 4).attr("y", ys(v) + 3).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", c.muted).text(v * 100); }
      const pts = d3.range(0, 41).map((k) => { const x = lo[ax] + ((hi[ax] - lo[ax]) * k) / 40, s = state.slice(); s[ax] = x; return [x, interp(s).val]; });
      svg.append("path").attr("d", d3.line().x((d) => xs(d[0])).y((d) => ys(d[1]))(pts)).attr("fill", "none").attr("stroke", c.acc).attr("stroke-width", 1.8).attr("stroke-dasharray", "4 3").attr("opacity", 0.9);
      // grid nodes = exact simulated values (only when the other sliders sit on grid nodes would these be exact; mark them hollow otherwise)
      const others = state.every((v, j) => j === ax || axes[j].values.some((g) => Math.abs(g - v) < 1e-9));
      for (const g of a.values) { const s = state.slice(); s[ax] = g; const v = interp(s).val; svg.append("circle").attr("cx", xs(g)).attr("cy", ys(v)).attr("r", 3.2).attr("fill", others ? c.acc : "none").attr("stroke", c.acc).attr("stroke-width", 1.4); }
      svg.append("circle").attr("cx", xs(state[ax])).attr("cy", ys(interp(state).val)).attr("r", 5.5).attr("fill", "none").attr("stroke", c.ink).attr("stroke-width", 2);
      svg.append("text").attr("x", X0).attr("y", 134).attr("class", "dg-ax").attr("fill", c.muted).text(lo[ax] + (AX_UNIT[a.name] || ""));
      svg.append("text").attr("x", X1).attr("y", 134).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", c.muted).text(hi[ax] + (AX_UNIT[a.name] || ""));
    }
  }
  const legend = el("p", { class: "dg-legend" }, "Dashed line: interpolated. Filled dot: simulated grid point (exact). Hollow dot: grid position, but other sliders are off-grid so it is interpolated. Ring: where the sliders are now.");
  sm.append(legend);

  function update(setSliders) {
    for (const { out, inp, i } of outs) {
      if (setSliders) inp.value = state[i];
      out.textContent = (axes[i].name === "crosstalk_k" ? state[i].toFixed(4) : state[i].toFixed(axes[i].name === "stage_snr_db" ? 0 : 1)) + (AX_UNIT[axes[i].name] || "");
    }
    const r = interp(state);
    big.replaceChildren(el("span", { class: "dg-num", text: pct1(r.val) }), el("span", { class: "dg-lab", text: "accuracy" }), el("span", { class: "dg-d", text: `${r.val - nominal.val >= 0 ? "+" : ""}${((r.val - nominal.val) * 100).toFixed(1)} pt vs nominal` }));
    big.dataset.exact = r.exact;
    badge.textContent = r.exact ? "simulated grid point" : "interpolated between grid points";
    big.append(badge); badge.dataset.exact = r.exact;
    drawCells();
  }

  // --------------------------------------------------------------------- Monte Carlo headline + histogram
  const mc = pt.monte_carlo;
  mcBox.append(el("h4", { class: "ar-h", text: "1000 simulated boards, datasheet tolerances" }),
    el("div", { class: "dg-stats" },
      ...[["mean accuracy", mc.accuracy_mean.toFixed(3)], ["sigma", mc.accuracy_std.toFixed(3)], ["yield", mc.yield.toFixed(2)], ["worst board", mc.accuracy_min.toFixed(3)]].map(([k, v]) => el("div", { class: "dg-stat" }, el("b", { text: v }), el("span", { text: k })))),
    el("div", { class: "dg-hist" }),
    el("p", { class: "w-foot", text: `${mc.n_trials} Monte Carlo boards, each with random part tolerances (L and C 5%, R 1%, clock 0.5%, V_BE 2 mV, phase 6 ns), ${mc.n_examples}-example stratified subset. Yield = ${mc.yield_rule} (threshold ${mc.yield_threshold}). Simulation of a designed board; none was built.` }));
  function drawHist() {
    const c = cs(), div = mcBox.querySelector(".dg-hist"); div.replaceChildren();
    const svg = d3.select(div).append("svg").attr("viewBox", "0 0 320 130").attr("role", "img").attr("aria-label", "Histogram of accuracy over 1000 simulated boards, centred at 0.856");
    const xs = d3.scaleLinear([0.83, 0.88], [30, 310]), bins = d3.bin().domain([0.83, 0.88]).thresholds(25)(mc.accuracies), ys = d3.scaleLinear([0, d3.max(bins, (b) => b.length)], [100, 8]);
    svg.selectAll("rect").data(bins).join("rect").attr("x", (b) => xs(b.x0) + 0.5).attr("width", (b) => Math.max(0, xs(b.x1) - xs(b.x0) - 1)).attr("y", (b) => ys(b.length)).attr("height", (b) => 100 - ys(b.length)).attr("fill", c.acc).attr("opacity", 0.85);
    svg.append("line").attr("x1", 30).attr("x2", 310).attr("y1", 100).attr("y2", 100).attr("stroke", c.muted);
    for (const v of [0.83, 0.84, 0.85, 0.86, 0.87, 0.88]) svg.append("text").attr("x", xs(v)).attr("y", 113).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text(v.toFixed(2));
    svg.append("text").attr("x", 170).attr("y", 127).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text("accuracy on the 800-example subset");
    svg.append("line").attr("x1", xs(mc.accuracy_mean)).attr("x2", xs(mc.accuracy_mean)).attr("y1", 6).attr("y2", 100).attr("stroke", c.ink).attr("stroke-dasharray", "3 3");
  }

  // --------------------------------------------------------------------- gain trim chart (finding F1)
  const gt = pt.gain_trim;
  trimBox.append(el("h4", { class: "ar-h", text: "The finding that matters: gain level" }), el("div", { class: "dg-trimsvg" }),
    el("p", { class: "w-foot", text: `Accuracy against the gain trim of the analog chain, board-trained net. As first designed (0 dB) it collapses to ${(gt.as_designed_0dB_full_test * 100).toFixed(0)}%: the tanh stages are under-driven by about 9 dB relative to what the net was trained on. The nominal trim (about ${gt.nominal_auto_db} dB, the level the net was trained at) gives ${(gt.nominal_accuracy_full_test * 100).toFixed(0)}%. The network needs the gain within roughly 2 dB. Line: sweep on a stratified subset; diamonds: the full ${gt.nominal_n_examples}-example test set.` }));
  function drawTrim() {
    const c = cs(), div = trimBox.querySelector(".dg-trimsvg"); div.replaceChildren();
    const svg = d3.select(div).append("svg").attr("viewBox", "0 0 340 190").attr("role", "img").attr("aria-label", "Accuracy against gain trim: 0 dB 0.18, 6 dB 0.46, 8 dB 0.84, about 9 dB 0.87");
    const xs = d3.scaleLinear([0, 10.5], [38, 328]), ys = d3.scaleLinear([0, 1], [150, 14]);
    for (const v of [0, 0.25, 0.5, 0.75, 1]) { svg.append("line").attr("x1", 38).attr("x2", 328).attr("y1", ys(v)).attr("y2", ys(v)).attr("stroke", c.rule).attr("stroke-width", 0.6); svg.append("text").attr("x", 33).attr("y", ys(v) + 3).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", c.muted).text(Math.round(v * 100)); }
    for (const v of [0, 2, 4, 6, 8, 10]) svg.append("text").attr("x", xs(v)).attr("y", 165).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text(v);
    svg.append("text").attr("x", 183).attr("y", 184).attr("text-anchor", "middle").attr("class", "dg-ax").attr("fill", c.muted).text("gain trim (dB)");
    svg.append("text").attr("x", 4).attr("y", 10).attr("class", "dg-ax").attr("fill", c.muted).text("accuracy %");
    const data = gt.values.map((v, i) => [v, gt.accuracy[i]]);
    svg.append("path").attr("d", d3.line().x((d) => xs(d[0])).y((d) => ys(d[1])).curve(d3.curveMonotoneX)(data)).attr("fill", "none").attr("stroke", c.acc).attr("stroke-width", 2);
    svg.selectAll("circle").data(data).join("circle").attr("cx", (d) => xs(d[0])).attr("cy", (d) => ys(d[1])).attr("r", 3.6).attr("fill", c.acc);
    svg.selectAll("text.v").data(data.filter((d) => [0, 6.0206, 8].includes(d[0]))).join("text").attr("class", "dg-ax").attr("x", (d) => xs(d[0]) + 5).attr("y", (d) => ys(d[1]) - 6).attr("fill", c.ink).text((d) => d[1].toFixed(2));
    const dia = [[0, gt.as_designed_0dB_full_test], [gt.nominal_auto_db, gt.nominal_accuracy_full_test]];
    svg.selectAll("path.dia").data(dia).join("path").attr("d", d3.symbol(d3.symbolDiamond, 70)).attr("transform", (d) => `translate(${xs(d[0])},${ys(d[1])})`).attr("fill", c.pos).attr("stroke", c.ink).attr("stroke-width", 0.8);
    svg.append("text").attr("x", xs(gt.nominal_auto_db) - 6).attr("y", ys(gt.nominal_accuracy_full_test) - 10).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", c.ink).text(`${gt.nominal_accuracy_full_test.toFixed(2)} nominal`);
    svg.append("text").attr("x", xs(0) + 8).attr("y", ys(gt.as_designed_0dB_full_test) + 14).attr("class", "dg-ax").attr("fill", c.ink).text(`${gt.as_designed_0dB_full_test.toFixed(2)} as designed`);
  }
  const redraw = () => { drawCells(); drawHist(); drawTrim(); };
  window.addEventListener("dlnn-theme", redraw);
  update(true); drawHist(); drawTrim();
}
