// sandbox.js: W3 circuit sandbox. Runs the TILE-8 unit cell (data/spice/unit_cell.template.cir: one 56-section LC delay line, 8 taps, summer,
// BJT diff pair) in ngspice-in-WebAssembly (eecircuit-engine, lazy-loaded from vendor/ on click, ~20 MB) with editable L, C, Q, tap weights and input
// pattern, and plots the traces with D3. If the template is absent the widget shows a "pending" state. SIMULATION of a design that is not built.
import { loadJSON, loadD3, el, cssVar, VENDOR, DATA } from "./common.js";
import { decodeModel } from "./tdlnet.js";

const TS_US = 0.8904;            // symbol clock of the board (models/board_aware/line_fir_m1a.json ts_us)
const RP_Q = { 10: 6350.9, 30: 21561.8 };   // Rp at L = 120 uH for Q = 10 / 30, from the template header
const L0 = 120, C0 = 120;        // template values (uH, pF)

export async function mountSandbox(host) {
  const tplRes = await fetch(new URL("spice/unit_cell.template.cir", DATA)).catch(() => null);
  if (!tplRes || !tplRes.ok) {
    host.replaceChildren(el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Circuit sandbox" }), el("span", { class: "w-tag", text: "W3 · pending" })),
      el("p", { class: "w-sub", text: "Pending: the unit-cell netlist (data/spice/unit_cell.template.cir) has not been produced yet. When it appears, this widget will run it in ngspice (WebAssembly) with editable L, C and tap weights." }));
    return;
  }
  const tpl = await tplRes.text();
  const [arena] = await Promise.all([loadJSON("arena_models.json"), loadD3()]);
  const d3 = window.d3;
  const bm = decodeModel(arena.models.find((m) => m.id === "dlnn_board").data);
  const l0 = bm.layers[0];
  const st = { L: L0, C: C0, Q: 10, amp: 0.4, w: [0, -1, -1, -1, -1, -1, 0, 0], pat: [1, 1, -1, 0, 1, -1, -1, 1], o: 1, i: 0 };
  const loadW = () => { for (let k = 0; k < 8; k++) st.w[k] = l0.W[(st.o * l0.I + st.i) * l0.K + k]; };
  loadW();

  const mkNum = (label, key, min, max, step, unit) => {
    const out = el("output", { text: st[key] + unit }), inp = el("input", { type: "range", min, max, step, value: st[key], "aria-label": label });
    inp.addEventListener("input", () => { st[key] = +inp.value; out.textContent = st[key] + unit; });
    return el("div", { class: "knob" }, el("label", { text: label }), inp, out);
  };
  const qSel = el("select", { "aria-label": "Inductor Q" }, el("option", { value: 10, text: "Q = 10 (datasheet minimum)" }), el("option", { value: 30, text: "Q = 30 (assumed typical)" }));
  qSel.addEventListener("change", () => { st.Q = +qSel.value; });
  const wRow = el("div", { class: "sb-weights", role: "group", "aria-label": "Tap weights" });
  const wBtns = [];
  const glyph = (v) => (v > 0 ? "+" : v < 0 ? "−" : "0");
  const drawW = () => wBtns.forEach((b, k) => { b.textContent = glyph(st.w[k]); b.dataset.w = st.w[k]; b.setAttribute("aria-label", `tap ${k} weight ${glyph(st.w[k])}`); });
  for (let k = 0; k < 8; k++) { const b = el("button", { type: "button", class: "sb-w", title: `tap ${k}: click to cycle +1, 0, -1` }); b.addEventListener("click", () => { st.w[k] = st.w[k] === 1 ? 0 : st.w[k] === 0 ? -1 : 1; drawW(); }); wBtns.push(b); wRow.append(el("div", { class: "sb-wc" }, b, el("small", { text: "t" + k }))); }
  const oSel = el("select", { "aria-label": "Load weights from trained layer 1" }, ...Array.from({ length: l0.O }, (_, o) => el("option", { value: o, text: `output o${o}` })));
  const iSel = el("select", { "aria-label": "Input channel" }, el("option", { value: 0, text: "channel I" }), el("option", { value: 1, text: "channel Q" }));
  oSel.value = st.o; iSel.value = st.i;
  const loadBtn = el("button", { class: "btn", type: "button", text: "Load trained weights" });
  loadBtn.addEventListener("click", () => { st.o = +oSel.value; st.i = +iSel.value; loadW(); drawW(); });
  const pRow = el("div", { class: "sb-weights", role: "group", "aria-label": "Input symbols" });
  const pBtns = [];
  const drawP = () => pBtns.forEach((b, n) => { b.textContent = glyph(st.pat[n]); b.dataset.w = st.pat[n]; });
  for (let n = 0; n < 8; n++) { const b = el("button", { type: "button", class: "sb-w", title: `symbol ${n}: click to cycle` }); b.addEventListener("click", () => { st.pat[n] = st.pat[n] === 1 ? 0 : st.pat[n] === 0 ? -1 : 1; drawP(); }); pBtns.push(b); pRow.append(el("div", { class: "sb-wc" }, b, el("small", { text: "s" + n }))); }
  drawW(); drawP();
  const run = el("button", { class: "btn primary", type: "button", id: "sb-run", text: "Run simulation (loads the 20 MB engine on first use)" });
  const msg = el("p", { class: "w-foot sb-msg", "aria-live": "polite", text: "Not run yet." });
  const plot = el("div", { class: "sb-plot" });
  host.replaceChildren(
    el("div", { class: "w-head" }, el("h3", { class: "w-title", text: "Circuit sandbox: one delay line, in real SPICE" }), el("span", { class: "w-tag", text: "W3 · simulated · ngspice in your browser" })),
    el("p", { class: "w-sub", text: "One line of the layer-1 tile: 56 LC sections, eight taps (every 8th section), a summing amplifier and a BJT differential pair. Change the inductance, capacitance, tap weights or the input pattern and run it. Op-amps are single-pole fits, not the vendor macro; the inductor is a datasheet-derived lumped model." }),
    el("div", { class: "sb-grid" },
      el("div", { class: "knobs" }, mkNum("Inductance L", "L", 60, 200, 5, " µH"), mkNum("Capacitance C", "C", 60, 200, 5, " pF"), mkNum("Input amplitude", "amp", 0.1, 1, 0.05, " V"), el("div", { class: "knob" }, el("label", { text: "Inductor Q" }), qSel, el("span"))),
      el("div", {}, el("h4", { class: "ar-h", text: "Tap weights (crossbar resistors)" }), wRow, el("div", { class: "ar-opts" }, oSel, iSel, loadBtn), el("h4", { class: "ar-h", text: "Input symbols (one per tap time)" }), pRow)),
    el("div", { class: "dg-btns" }, run), msg, plot);

  let sim = null;
  function netlist() {
    let n = tpl;
    const sL = st.L / L0;
    n = n.replace(/^(L\d+ \S+ \S+ )120u/gm, `$1${st.L}u`).replace(/^(C\d+ n\d+ 0 )120p/gm, `$1${st.C}p`)
      .replace(/^(C00 n0 0 )60p/m, `$1${st.C / 2}p`).replace(/^(C56 n56 0 )60p/m, `$1${st.C / 2}p`);
    // Rp scales with L (Rp ~ Q w L); Cp = 1/((2 pi f_srf)^2 L) scales with 1/L (SRF held at 3.3 MHz)
    n = n.replace("{{Q_RP}}", (RP_Q[st.Q] * sL).toFixed(1)).replace(/CP=19\.383p/, `CP=${(19.383 / sL).toFixed(3)}p`);
    for (let k = 0; k < 8; k++) { n = n.replace(`{{RXP${k}}}`, st.w[k] > 0 ? "10k" : "1G").replace(`{{RXN${k}}}`, st.w[k] < 0 ? "10k" : "1G"); }
    const tr = 1e-8, T = TS_US * 1e-6, pts = ["0 2.5"];
    st.pat.forEach((s, i) => { const t = (1 + i * TS_US) * 1e-6, v = 2.5 + s * st.amp; pts.push(`${(t).toExponential(4)} ${pts.length === 1 ? 2.5 : 2.5 + st.pat[i - 1] * st.amp}`, `${(t + tr).toExponential(4)} ${v}`); });
    const tEnd = (1 + st.pat.length * TS_US + 7) * 1e-6;
    pts.push(`${((1 + st.pat.length * TS_US) * 1e-6).toExponential(4)} ${2.5 + st.pat[st.pat.length - 1] * st.amp}`, `${((1 + st.pat.length * TS_US) * 1e-6 + tr).toExponential(4)} 2.5`, `${tEnd.toExponential(4)} 2.5`);
    n = n.replace("{{VIN_PWL}}", pts.join(" "));
    void T;
    return n.replace(/^\.end\s*$/m, `.tran 20n ${tEnd.toExponential(4)} 0 20n\n.end`);
  }
  async function ensureSim() {
    if (sim) return sim;
    msg.textContent = "Loading the circuit engine (about 20 MB, once)...";
    const mod = await import(new URL("eecircuit-engine/eecircuit-engine.mjs", VENDOR).href);
    sim = new mod.Simulation();
    await sim.start();
    return sim;
  }
  run.addEventListener("click", async () => {
    run.disabled = true;
    try {
      const s = await ensureSim();
      msg.textContent = "Simulating (transient, 56 LC sections)...";
      await new Promise((r) => setTimeout(r, 30));
      const t0 = performance.now();
      s.setNetList(netlist());
      const res = await s.runSim();
      const data = {}; for (const v of res.data) data[v.name.toLowerCase()] = v.type === "time" || v.name === "time" ? v.values : v.values.map((q) => (typeof q === "number" ? q : q.real ?? q));
      if (!data.time) throw new Error("no time vector in the result: " + Object.keys(data).slice(0, 6).join(", "));
      draw(data);
      msg.textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s: ${data.time.length} time points. Simulation of a design that is not built.`;
    } catch (e) { console.error("sandbox", e); msg.textContent = "Simulation failed: " + (e.message || e); } finally { run.disabled = false; }
  });

  function draw(d) {
    const c = { ink: cssVar("--ink"), muted: cssVar("--muted"), rule: cssVar("--rule"), acc: cssVar("--accent"), pos: cssVar("--pos"), neg: cssVar("--neg") };
    plot.replaceChildren();
    const t = d.time.map((x) => x * 1e6), W = 760, tmax = d3.max(t);
    const panels = [
      { title: "Input and the eight taps (change from their start value, volts)", y: 150, series: [["v(vin)", c.ink, 2], ...Array.from({ length: 8 }, (_, k) => [`v(t${k})`, k % 2 ? c.neg : c.pos, 1])] },
      { title: "Summed output of the differential pair, v(out) (change from start, volts)", y: 150, series: [["v(out)", c.acc, 2]] }];
    const svg = d3.select(plot).append("svg").attr("viewBox", `0 0 ${W} ${panels.length * 190 + 10}`).attr("role", "img").attr("aria-label", "Simulated waveforms: input, tap voltages and the summed output");
    const xs = d3.scaleLinear([0, tmax], [46, W - 12]);
    panels.forEach((p, pi) => {
      const g = svg.append("g").attr("transform", `translate(0,${pi * 190 + 10})`);
      const all = p.series.map(([name]) => (d[name] || []).map((v, i, a) => v - a[0]));
      const ext = d3.extent(all.flat()), pad = (ext[1] - ext[0]) * 0.08 || 0.01, ys = d3.scaleLinear([ext[0] - pad, ext[1] + pad], [p.y, 14]);
      g.append("text").attr("x", 46).attr("y", 8).attr("class", "dg-t").attr("fill", c.ink).text(p.title);
      g.append("g").call(d3.axisLeft(ys).ticks(4)).attr("transform", "translate(46,0)").call((a) => a.selectAll("text").attr("fill", c.muted).attr("class", "dg-ax")).call((a) => a.selectAll("path,line").attr("stroke", c.rule));
      g.append("g").call(d3.axisBottom(xs).ticks(8)).attr("transform", `translate(0,${p.y})`).call((a) => a.selectAll("text").attr("fill", c.muted).attr("class", "dg-ax")).call((a) => a.selectAll("path,line").attr("stroke", c.rule));
      g.append("text").attr("x", W - 12).attr("y", p.y + 28).attr("text-anchor", "end").attr("class", "dg-ax").attr("fill", c.muted).text("time (µs)");
      p.series.forEach(([name, col, w], si) => { if (!d[name]) return; g.append("path").attr("d", d3.line().x((_, i) => xs(t[i])).y((v) => ys(v))(all[si])).attr("fill", "none").attr("stroke", col).attr("stroke-width", w).attr("opacity", name === "v(vin)" || name === "v(out)" ? 1 : 0.6); });
    });
  }
}
